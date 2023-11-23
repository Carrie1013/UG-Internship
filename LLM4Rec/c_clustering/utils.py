import ast
import math
import json
import openai
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from langchain.embeddings import OpenAIEmbeddings

openai.api_key ='xxx'
openai.api_base = 'https://xxxtcl-ai.openai.azure.com/' 
openai.api_type = 'azure'
openai.api_version = '2023-07-01-preview' 
model_name = "text-embedding-ada-002"
deployment_name = "embedding-ada-002-2"

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
            deployment = deployment_name,
            model = model_name,
            openai_api_key = "xxx",
            openai_api_base = 'https://xxxtcl-ai.openai.azure.com/',
            openai_api_type = 'azure',
            openai_api_version = '2023-07-01-preview',
            chunk_size = 1,
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

def get_theme_tags(theme_corpus_path, dataset_path):
    try:
        with open(theme_corpus_path, 'r') as f:
            theme_tags = eval(f.read())
    except:
        movie_tags = pd.read_csv(dataset_path)
        theme_tags = []
        emo_tags = []
        themes = [i for i in movie_tags['themes_cleaned'].tolist() if not (isinstance(i, float) and math.isnan(i))]
        emos = [i for i in movie_tags['emotion_tags_cut'].tolist() if not (isinstance(i, float) and math.isnan(i))]

        for th in themes:
            formatted_th = th.replace("' '", "', '")
            try:
                theme_list = ast.literal_eval(formatted_th)
                theme_tags.extend(theme_list)
            except ValueError as e:
                print(f"Error converting {formatted_th}: {e}")

        for th in emos:
            formatted_th = th.replace("' '", "', '")
            try:
                emos_list = ast.literal_eval(formatted_th)
                emo_tags.extend(emos_list)
            except ValueError as e:
                print(f"Error converting {formatted_th}: {e}")
        
        theme_tags.extend(emo_tags)
        with open(theme_corpus_path, 'w') as f:
            f.write(theme_tags) 

    return theme_tags

def data_augmentation(corpus, scale): # input: corpus_path, output: aug_corpus_path

    with open(corpus, 'r') as f:
        words = eval(f.read())
    aug_words = [word.strip() * scale for word in words]

    aug_path = f"corpus/{scale}_aug_map_list.txt"
    with open(aug_path, "w") as f:
        f.write(str(aug_words))
    
    print("the augmented data is stored in: ", aug_path)

def dim_reduction(n_components, embedding_dict): # input: dict; output: dict

    keys = list(embedding_dict.keys())
    embeddings = list(embedding_dict.values())

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings).tolist()
    pca_embedding_dict = dict(zip(keys, pca_embeddings))

    return pca_embedding_dict

def kmeans_cls(num_clusters, tags, embeddings): # K-means

    print('cluster=', num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    return dict(zip(tags, cluster_labels))