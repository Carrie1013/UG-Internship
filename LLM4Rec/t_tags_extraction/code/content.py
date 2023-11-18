import os
import ast
import math
import json
import torch
import openai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from langchain.embeddings import OpenAIEmbeddings
from transformers import BertTokenizer, BertModel
movie_tags = pd.read_csv('movie_tags.csv')

type_tags = []
types = movie_tags['types'].tolist()
types = [i for i in types if not (isinstance(i, float) and math.isnan(i))]

for item in types:
    formatted_item = item.replace("' '", "', '")
    try:
        tag_list = ast.literal_eval(formatted_item)
        type_tags.extend(tag_list)
    except ValueError as e:
        print(f"Error converting {formatted_item}: {e}")

seen = set()
counts = Counter(type_tags)
unique_type_tags = list(set(type_tags))
filtered_type_tags = [x for x in type_tags if counts[x] >= 5 and (x not in seen and not seen.add(x))]

embedding = OpenAIEmbeddings(
    deployment = "embedding-ada-002-2",
    model = "text-embedding-ada-002",
    openai_api_key = "6e25ec6fa59d44f8af091db59e6db6d7",
    openai_api_base = 'https://tcl-ai.openai.azure.com/',
    openai_api_type = 'azure',
    openai_api_version = '2023-07-01-preview',
    chunk_size=1,
)

tag_documents = unique_type_tags
tag_embeddings = embedding.embed_documents(tag_documents)

# K-means 聚类
def kmeans_cls(num_clusters):
    print('cluster=', num_clusters)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(tag_embeddings)
    cluster_labels = kmeans.labels_

    clusters = {}
    for word, cluster_label in zip(tag_documents, cluster_labels):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(word)

    for cluster_label, words in clusters.items():
        print(f"Cluster {cluster_label}: {words}")

kmeans_cls(5)
kmeans_cls(6)
kmeans_cls(7)
kmeans_cls(8)
kmeans_cls(9)
kmeans_cls(10)


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

def get_bert_embedding(label):
    inputs = tokenizer(label, return_tensors="pt", padding=True, truncation=True, max_length=20)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][:,0,:].numpy()

bert_tag_embeddings = [get_bert_embedding(tag) for tag in unique_type_tags]


bert_embeddings_array = np.vstack(bert_tag_embeddings)

num_clusters = 5 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(bert_embeddings_array)
cluster_labels = kmeans.labels_

clusters = {}
for tag, cluster_label in zip(unique_type_tags, cluster_labels):
    if cluster_label not in clusters:
        clusters[cluster_label] = []
    clusters[cluster_label].append(tag)

# 打印聚类结果
for cluster_label, tags in clusters.items():
    print(f"Cluster {cluster_label}: {tags}")




