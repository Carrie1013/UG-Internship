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
from sklearn.metrics import silhouette_score
from langchain.embeddings import OpenAIEmbeddings
from transformers import BertTokenizer, BertModel

def get_tags_from_movielist():
    movie_tags = pd.read_csv('LLM_Movie10000+/t_tags_extraction/code/movie_tags.csv')
    theme_tags = [] # movie_tags中theme集合
    emo_tags = [] # movie_tags中emotions集合

    themes = [i for i in movie_tags['themes_cleaned'].tolist() if not (isinstance(i, float) and math.isnan(i))]
    emos = [i for i in movie_tags['emotion_tags_cut'].tolist() if not (isinstance(i, float) and math.isnan(i))]

    # format the input
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

    unique_them_tags = list(set(theme_tags))
    unique_emo_tags = list(set(emo_tags))
    # required unique theme and emotion tags
    unique_them_tags.extend(unique_emo_tags)
    unique_theme_tags = list(set(unique_them_tags))

    seen_t, seen_e = set(), set()
    counts_t , counts_e = Counter(theme_tags), Counter(emo_tags)
    filtered_them_tags = [x for x in theme_tags if counts_t[x] >= 5 and (x not in seen_t and not seen_t.add(x))]
    filtered_emo_tags = [x for x in emo_tags if counts_e[x] >= 5 and (x not in seen_e and not seen_e.add(x))]
    # required filtered theme and emotion tags
    filtered_them_tags.extend(filtered_emo_tags)
    filtered_theme_tags = list(set(filtered_them_tags))

    return unique_theme_tags, filtered_theme_tags

def generate_embedngs_with_llm(file_path):
    # 导入LLM-ada002-theme嵌入
    embedding = OpenAIEmbeddings(
        deployment = "embedding-ada-002-2",
        model = "text-embedding-ada-002",
        # openai_api_key = "xxx",
        openai_api_base = 'https://xxxtcl-ai.openai.azure.com/',
        openai_api_type = 'azure',
        openai_api_version = '2023-07-01-preview',
        chunk_size=1,
    )

    # 生成ada_theme_embedding结果
    theme_documents, _ = get_tags_from_movielist()
    theme_embeddings = embedding.embed_documents(theme_documents)

    # 保存ada_theme_embedding结果
    theme_embeddings_dict = {}
    for themeTag, themeEmbedding in zip(theme_documents, theme_embeddings):
        theme_embeddings_dict[themeTag] = (themeEmbedding)
    with open(file_path, 'w') as json_file:
        json.dump(theme_embeddings_dict, json_file)

def get_embeddings(file_path):
    # 从json文件导入LLM-ada002-theme嵌入
    # file_path = "theme2_embeddings_ada002.json"
    try:
        with open(file_path, 'r') as json_file:
            theme_embeddings_dict = json.load(json_file)
        theme_embeddings = list(theme_embeddings_dict.values())
    except:
        generate_embedngs_with_llm(file_path)
        get_embeddings(file_path)
    return theme_embeddings, theme_embeddings_dict
