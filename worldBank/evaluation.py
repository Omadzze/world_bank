import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


class DataEvaluation:
    def apply_pca(self, data: pd.DataFrame, n_components: int = 2, drop_columns: list = None, random_state: int = 42):
        """
        Applies PCA to the given DataFrame.

        Parameters:
            data (pd.DataFrame): The input DataFrame.
            n_components (int): Number of PCA components to extract.
            drop_columns (list): List of column names to drop before applying PCA.
            random_state (int): Random state for reproducibility.

        Returns:
            df_pca (pd.DataFrame): DataFrame containing the PCA-transformed data with columns 'PC1', 'PC2', ...
            pca_model (PCA): The fitted PCA model.
        """
        if drop_columns is not None:
            data_for_pca = data.drop(columns=drop_columns)
        else:
            data_for_pca = data.copy()

        pca_model = PCA(n_components=n_components, random_state=random_state)
        pca_components = pca_model.fit_transform(data_for_pca)
        df_pca = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(n_components)])

        return df_pca, pca_model

    def plot_pca_clusters(self, df_pca: pd.DataFrame, cluster_labels: np.ndarray, title: str = "Clustering Results",
                          centroids: np.ndarray = None, pca_model: PCA = None, noise_label: int = -1):
        """
        Plots clusters in the PCA space.

        Parameters:
            df_pca (pd.DataFrame): DataFrame with PCA components (e.g., 'PC1' and 'PC2').
            cluster_labels (np.ndarray): Cluster labels corresponding to each row in df_pca.
            title (str): Plot title, name of different algorithms that were used.
            centroids (np.ndarray): (Optional) Centroid coordinates in the original space.
            pca_model (PCA): (Optional) PCA model used to transform the original centroids into PCA space.
            noise_label (int): The label used to denote noise (default is -1).

        Returns:
            None. Displays the scatter plot.
        """
        plt.figure(figsize=(8, 6))

        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = (cluster_labels == label)
            if label == noise_label:
                plt.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'],
                            s=50, c='black', label='Noise', edgecolors='k')
            else:
                plt.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'],
                            s=50, c=[color], label=f'Cluster {label}', edgecolors='k')

        # If centroids are provided, transform them into PCA space and plot.
        if centroids is not None and pca_model is not None:
            centroids_pca = pca_model.transform(centroids)
            plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                        s=200, marker='X', c='black', label='Centroids')

        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.tight_layout()
        plt.show()