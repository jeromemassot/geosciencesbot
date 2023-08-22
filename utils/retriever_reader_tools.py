
from nemoguardrails import LLMRails, RailsConfig
import google.generativeai as palm
import sciencebasepy
import pinecone
import openai

from collections import defaultdict
import numpy as np
import json
import os


def extract_topics_from_query(query:str, resources:dict, llm:str, use_embeddings:str='No') -> list:
    """
    Extract topics (mainly mineral names) from user query
    :param query: user query as string
    :param resources: knowledge used to analyze the user query
    :param llm: encoding llm name
    :param use_embeddings: use the context embeddings option [No, Fuzzy, Strict]
    :return: set of topics
    """

    def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    topics = set()

    # case 1: the mineral is directly mentionned in the query
    for k in resources.keys():
        if query.lower().find(k) >= 0:
            topics.add(k)

    if len(topics)==0 and use_embeddings != "No":
        # case 2: the query does not specify a mineral but a possible application
        # in this case, the similarity of the resources knowledge and the query is
        # used to populate the topics list.
        encoded_query = palm.generate_embeddings(model=llm, text=query)['embedding']

        similarities = [(k, cos_sim(encoded_query, v)) for k, v in resources.items()]
        similarities.sort(key=lambda x:x[1])

        if use_embeddings == 'Strict':
            topics.add(similarities.pop()[0])
        else:
            for i in range(3):
                current_max = similarities.pop()
                topics.add(current_max[0])

    return topics


def extract_documents_info_from_usgs(query:str, resources:dict, llm:str, use_embeddings:str='Strict') -> str:
    """
    Extract USGS documentation for the given topic.
    :param query: user query
    :param resources: knowledge used to analyze the user query
    :param llm: encoding llm name
    :param use_embeddings: use the context embeddings option [No, Fuzzy, Strict]
    :return: contexts as string and retrieved content as json
    """

    # find the topics related for the user query
    topics = extract_topics_from_query(query, resources, llm, use_embeddings)

    # placeholder for the return content as json
    retrieved_content = defaultdict(list)

    # Establish a session.
    sb = sciencebasepy.SbSession()

    # get the USGS documentation for the given topics
    for topic in topics:

        # found documents
        found_documents = sb.find_items({'q':topic})

        for found_document in found_documents['items']:

            # get the summary of the document if available
            summary = found_document['summary'] if 'summary' in found_document.keys() else "No abstract available"

            found_document_dict = {
                "title": found_document['title'],
                "nature": 'document',
                "summary": summary,
                "url": found_document['link']['url'],
                "node_id": found_document['id']
            }

            # enrich the dict with origin and topic for serialization
            found_document_dict.update({'origin': 'USGS', 'topic': topic})
            retrieved_content[topic].append(found_document_dict)

    
    # reformat the context as string to be used in an enriched prompt
    context = "The following information can be given:\n"
    for k, documents in retrieved_content.items():
        context += f"For the mineral {k}: "
        for document in documents:
            if document['summary'] != "No abstract available":
                context += document['summary']
        context += '\n---------------------------\n'

    return context, retrieved_content


async def retrieve_context_from_usgs(query:str) -> str:
    """
    Retrieve USGS documentation for the given query
    :param query: user query
    :return: contexts as string and retrieved content as json
    """

    # resources knowledge and embedding model used
    resources_embeddings = json.load(open(os.path.join('resources', 'encoded_mineral_industrial_usages.json'), 'r'))
    embedding_models = [m.name for m in palm.list_models() if 'embedText' in m.supported_generation_methods]

    context, _ = extract_documents_info_from_usgs(
        query,
        resources_embeddings,
        embedding_models[0],
        'Strict'
    )

    return context


async def rag_usgs(query:str, context:str) -> str:
    """
    Answer the query using RAG if needed.
    :param query: user's question
    :param context: context string covering the user query
    :return: answer as string
    """


    # prompt template to be used by the LLM
    prompt = f"""You are a helpful assistant, below is a query from a user and
    some relevant context. Answer the question given the information in those
    context. If you cannot find the answer to the question, say "I don't know".

    Contexts:
    {context}

    Query: {query}

    Answer: """
    # generate answer
    res = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=100,
    )

    returned_msg = f"Using the USGS technical library, I can say that: {res['choices'][0]['text']}"

    return returned_msg


async def retrieve_context_from_slb(query:str, top_k:int=3) -> list:
    """
    Search the index
    :param query: the query
    :param top_k: the number of results to return
    :return: the list of results
    """

    def encode_query(query:str, embedding_model:str):
        """
        Encode the query
        :param query: the query
        :param embedding_model: the embedding model
        :return: the encoded query
        """
        # encode the query
        dense = palm.generate_embeddings(model=embedding_model, text=query)['embedding']
        return dense

    # get the pinecone index
    index = pinecone.Index("hybrid-slb-glossary")

    # encode the query into dense vector
    embedding_models = [m.name for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
    hdense = encode_query(query, embedding_models[0])

    # search the pinecone index
    res = index.query(top_k=top_k, vector=hdense, include_metadata=True)
    contexts = [x['metadata']['text'] for x in res['matches']]

    return contexts


async def rag_slb(query: str, contexts: list) -> str:
    """
    Answer the query using RAG if needed.
    :param query: user's question
    :param contexts: list of contexts retrieved from vector index
    :return: answer as string
    """

    # we merge the contexts as a single string object.
    context_str = "\n".join(contexts)

    # prompt template to be used by the LLM
    prompt = f"""You are a helpful assistant, below is a query from a user and
    some relevant contexts. Answer the question given the information in those
    contexts. If you cannot find the answer to the question, say "I don't know".

    Contexts:
    {context_str}

    Query: {query}

    Answer: """
    # generate answer
    res = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=100,
    )

    returned_msg = f"Using the SLB technical glossary, I can say that: {res['choices'][0]['text']}"

    return returned_msg


def init_guardrails() -> LLMRails:
    """
    Initialize the guardrails
    :return: rails object
    """

    yaml_content = """
    models:
      - type: main
        engine: openai
        model: text-davinci-003
    """

    with open("./resources/colang_content.txt", "r") as fp:
        rag_colang_content = fp.read()

    # initialize rails config
    config = RailsConfig.from_content(
        colang_content=rag_colang_content,
        yaml_content=yaml_content
    )
    
    # create rails
    rag_rails = LLMRails(config)

    # register actions
    rag_rails.register_action(action=retrieve_context_from_slb, name="retrieve_context_from_slb")
    rag_rails.register_action(action=rag_slb, name="rag_slb")

    rag_rails.register_action(action=retrieve_context_from_usgs, name="retrieve_context_from_usgs")
    rag_rails.register_action(action=rag_usgs, name="rag_usgs")

    return rag_rails
