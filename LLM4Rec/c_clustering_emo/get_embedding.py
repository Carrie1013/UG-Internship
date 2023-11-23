import json
from langchain.embeddings import OpenAIEmbeddings

def getEmbeddingDict(corpus_path, embedding_path):

    try:
        with open(embedding_path, 'r') as f:
            embedding_dictionary = json.load(f)

    except:
        with open(corpus_path, 'r') as f:
            tags = eval(f.read()) # load theme tags
        unique_tags = list(set(tags)) # unique tags

        # LLM-ada002 configuration
        embedding = OpenAIEmbeddings(
            deployment = "embedding-ada-002-2",
            model = "text-embedding-ada-002",
            openai_api_key = "xxx",
            openai_api_base = 'https://xxxtcl-ai.openai.azure.com/',
            openai_api_type = 'azure',
            openai_api_version = '2023-07-01-preview',
            chunk_size=1,
        )

        # generate LLM-ada002 embedding
        tag_documents = unique_tags
        tag_embeddings = embedding.embed_documents(tag_documents)

        # save LLM-ada002 embedding
        embedding_dictionary = {}
        for tagDoc, tagEmbedding in zip(tag_documents, tag_embeddings):
            embedding_dictionary[tagDoc] = (tagEmbedding)
        file_path = embedding_path
        with open(file_path, 'w') as json_file:
            json.dump(embedding_dictionary, json_file)
    
    return embedding_dictionary