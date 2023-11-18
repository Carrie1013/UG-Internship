import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from themeEmbedding import get_embeddings, get_tags_from_movielist


def pca_dim_reduction(n_components, embeddings):
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca_embeddings


def kmeans_cls(num_clusters, tags, embeddings): # K-means 聚类

    print('cluster=', num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_

    clusters = {}
    for word, cluster_label in zip(tags, cluster_labels):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(word)

    # print to output cluster result
    for cluster_label, words in clusters.items():
        print(f"Cluster {cluster_label}: {words}")


def chooseK(embeddings, upper):
    bert_embeddings_array = np.vstack(embeddings)
    sum_of_squared_distances = []
    silhouette_scores = []
    K = range(2, upper)  # Test number of clusters from 2 to 10

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans = kmeans.fit(bert_embeddings_array)
        sum_of_squared_distances.append(kmeans.inertia_)

        # Silhouette score
        silhouette_avg = silhouette_score(bert_embeddings_array, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

    # Plotting the Elbow Curve
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    # Plotting the Silhouette Scores
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis For Optimal k')
    plt.show()

    # Choosing the k for the highest silhouette score
    optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
    # print(f"The optimal number of clusters is: {optimal_k}")
    return optimal_k

def DEC_cls(num_clusters, dim, tags, embeddings):
    # input data 处理
    a_theme_embeddings = np.array(embeddings)
    a_theme_embeddings = StandardScaler().fit_transform(a_theme_embeddings)
    input_dim = a_theme_embeddings.shape[1]
    encoding_dim = dim
    # data encoding 处理
    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(inputs) # encoder
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded) # decoder
    autoencoder = tf.keras.Model(inputs, decoded) # full autoencoder
    # 训练 autoencoder
    encoder = tf.keras.Model(inputs, encoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(a_theme_embeddings, a_theme_embeddings, epochs=50, batch_size=256, shuffle=True)
    # 使用训练后的autoencoder对数据编码
    encoded_embeddings = encoder.predict(a_theme_embeddings)

    # 定义Cluster数 - 这里Kmeans只用于初始化聚类中心
    K = num_clusters
    kmeans = KMeans(n_clusters=K, n_init=20)
    y_pred = kmeans.fit_predict(encoded_embeddings)
    cluster_centers = kmeans.cluster_centers_

    # 定义Layer, 对input data进行软分配
    class ClusteringLayer(tf.keras.layers.Layer):
        def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
            super(ClusteringLayer, self).__init__(**kwargs)
            self.n_clusters = n_clusters
            self.alpha = alpha
            self.initial_weights = weights
            self.input_spec = tf.keras.layers.InputSpec(ndim=2)

        def build(self, input_shape):
            assert len(input_shape) == 2
            input_dim = input_shape[1]
            self.cluster_centers = self.add_weight(name='cluster_centers',
                shape=(self.n_clusters, input_dim),
                initializer='glorot_uniform')
            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)
                del self.initial_weights
            self.built = True

        def call(self, inputs, **kwargs):
            q = 1.0 / (1.0 + (tf.keras.backend.sum(tf.keras.backend.square(
                tf.keras.backend.expand_dims(inputs, axis=1) - self.cluster_centers), axis=2) / self.alpha))
            q **= (self.alpha + 1.0) / 2.0
            q = tf.keras.backend.transpose(tf.keras.backend.transpose(q) / tf.keras.backend.sum(q, axis=1))
            return q

    # add clustering layer
    clustering_layer = ClusteringLayer(K, weights=[cluster_centers], name='clustering')(encoded)
    model = tf.keras.Model(inputs=inputs, outputs=[clustering_layer, decoded])
    # compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(0.1, 0.9), loss=['kld', 'mse'])

    #训练
    y_pred_last = np.copy(y_pred)
    for ite in range(100):  # 迭代次数
        q, _ = model.predict(a_theme_embeddings, verbose=0)
        p = np.zeros_like(q)
        p[y_pred] = q[y_pred]
        model.train_on_batch(x=a_theme_embeddings, y=[p, a_theme_embeddings])
        q = model.predict(a_theme_embeddings, verbose=0)[0]
        y_pred = q.argmax(1) # y_pred为最终聚类结果

    # 输出预测结果
    L = y_pred.tolist()
    print("cluster: ", max(L)+1)
    clusters = {}
    for i in range(len(L)):
        if L[i] not in clusters:
            clusters[L[i]] = []
        clusters[L[i]].append(tags[i])
    # output
    for cluster_label, words in clusters.items():
        print(f"Cluster {cluster_label}: {words}")


if __name__ =='__main__':

    unique_theme_tags, filtered_theme_tags = get_tags_from_movielist()
    theme_embeddings, theme_embeddings_dict = get_embeddings("LLM_Movie10000+/t_tags_extraction/code/theme2_embeddings_ada002.json")
    filtered_theme_embeddings = [theme_embeddings_dict[t] for t in filtered_theme_tags if t in theme_embeddings_dict]
    print(len(filtered_theme_embeddings))
    # kmeans_cls(53, unique_theme_tags, theme_embeddings)

    pca_theme_embeddings = pca_dim_reduction(n_components=64, embeddings=theme_embeddings)
    kmeans_cls(53, unique_theme_tags, pca_theme_embeddings)

    # pca_filtered_theme_embeddings =  pca_dim_reduction(n_components=128, embeddings=filtered_theme_embeddings)
    # kmeans_cls(53, filtered_theme_tags, pca_filtered_theme_embeddings)

    # DEC_cls(53, 128, unique_theme_tags, theme_embeddings)