import json
import openai
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from utils import get_theme_tags, getEmbeddingDict

percentile = 50
dataset_path = 'data/movie_tags.csv'
theme_corpus_path = 'data/theme_map_list.txt'
theme_embeddings_path = 'data/thememap_embeddings_ada002.json'
output_path = f'data/data_thememap_{percentile}p.csv'

theme_tags = get_theme_tags(theme_corpus_path, dataset_path)
theme_embeddings_dict = getEmbeddingDict(theme_corpus_path, theme_embeddings_path)
unique_theme_tags = list(theme_embeddings_dict.keys()) # load unique tags
theme_embeddings = list(theme_embeddings_dict.values()) # load filtered tags

def create_initial_cluster(tags):

    df_tag = pd.DataFrame()
    df_tag['tag'] = pd.Series(list(set(tags)))
    tag_counts = {tag: tags.count(tag) for tag in df_tag['tag'].unique()}
    df_tag['count'] = df_tag['tag'].map(tag_counts)
    # 初始center：count>5, length<7
    df_tag['filter_tag'] = np.where((df_tag['count'] > 5) & (df_tag['tag'].str.len() <= 6), df_tag['tag'], np.nan)

    initial_cluster_centers_tags = df_tag['filter_tag'].dropna().unique()
    initial_cluster_centers_embeddings = np.array([theme_embeddings_dict[tag] for tag in initial_cluster_centers_tags])

    # 使用Filtered Tags作为center进行Initial Clustering
    num_initial_clusters = len(initial_cluster_centers_embeddings)
    kmeans = KMeans(n_clusters=num_initial_clusters, init=initial_cluster_centers_embeddings, n_init=1, max_iter=300)
    all_tags = df_tag['tag'].unique()
    all_embeddings = np.array([theme_embeddings_dict[tag] for tag in all_tags if tag in theme_embeddings_dict])
    kmeans.fit(all_embeddings)

    labels = kmeans.labels_
    tag_cluster_mapping = pd.DataFrame({'tag': all_tags, 'cluster': labels})
    return tag_cluster_mapping, kmeans, all_embeddings, labels

def create_new_clusters(tags_df, embeddings, threshold, kmeans):
    new_cluster_centers = [] # new cluster centers
    tags_df_updated = tags_df.copy() # Copy the DataFrame

    # no tags above threshold 的时候收敛
    while tags_df_updated['distance_to_center'].max() > threshold:

        max_distance_tag = tags_df_updated.loc[tags_df_updated['distance_to_center'].idxmax()]

        new_cluster_center = embeddings[tags_df['tag'] == max_distance_tag['tag']][0]
        new_cluster_centers.append(new_cluster_center)

        all_cluster_centers = np.vstack([kmeans.cluster_centers_, new_cluster_centers])
        all_distances = cdist(embeddings, all_cluster_centers, metric='euclidean')

        new_labels = np.argmin(all_distances, axis=1)
        new_distances_to_center = all_distances[np.arange(all_distances.shape[0]), new_labels]

        tags_df_updated['cluster'] = new_labels
        tags_df_updated['distance_to_center'] = new_distances_to_center

    return tags_df_updated, new_cluster_centers


def create_re_clustering(mapping, p, all_embeddings, kmeans, labels):

    distances = cdist(all_embeddings, kmeans.cluster_centers_[labels], metric='euclidean').diagonal()
    mapping['distance_to_center'] = distances
    threshold = np.percentile(distances, p) 
    tag_cluster_mapping_updated, _ = create_new_clusters(mapping, all_embeddings, threshold, kmeans)
    
    return tag_cluster_mapping_updated

if __name__ == "__main__":

    tag_cluster_mapping, kmeans, all_embeddings, labels = create_initial_cluster(theme_tags)
    current_df = create_re_clustering(tag_cluster_mapping, percentile, all_embeddings, kmeans, labels)
    current_df.columns = ['tag', 'new_cluster', 'new_distance']
    tag_cluster_mapping = pd.merge(tag_cluster_mapping, current_df, on='tag')
    output_df = tag_cluster_mapping

    new_centers_indices = output_df[output_df['new_distance'] == 0.0].index
    new_centers_tags = output_df.loc[new_centers_indices, 'tag']
    new_centers_embeddings = np.array([theme_embeddings_dict[tag] for tag in new_centers_tags])

    # Re-run KMeans with the new cluster centers
    kmeans = KMeans(n_clusters=len(new_centers_embeddings), init=new_centers_embeddings, n_init=1)
    all_embeddings = np.array([theme_embeddings_dict[tag] for tag in output_df['tag']])
    kmeans.fit(all_embeddings)

    # Assign new clusters and calculate distances
    output_df['re_cluster'] = kmeans.labels_
    output_df['re_distance_to_center'] = [np.min(np.linalg.norm(embedding - kmeans.cluster_centers_, axis=1)) for embedding in all_embeddings]
    output_df = output_df.sort_values('re_cluster')[['tag', 're_cluster']]
    output_df.columns = [['tag', 'cluster']]
    output_df.to_csv(output_path, index=False)
    print("The output table is stored in: ", output_path)
    print(output_df)