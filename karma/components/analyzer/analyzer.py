import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.manifold import TSNE
from collections import defaultdict


class Analyzer:
    def __init__(self, trajectory_dir="trajectories", output_dir="analyzer_output"):
        """
        Initialize the Analyzer module.

        Args:
            trajectory_dir (str): Directory where agent trajectories are stored.
            output_dir (str): Directory to store analysis results.
        """
        self.trajectory_dir = trajectory_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_trajectories(self):
        """
        Load trajectories from the specified directory.

        Returns:
            dict: A dictionary mapping agent IDs to their respective trajectories.
        """
        print("Loading trajectories...")
        trajectories = {}
        for file in os.listdir(self.trajectory_dir):
            if file.endswith(".json"):
                agent_id = os.path.splitext(file)[0]
                with open(os.path.join(self.trajectory_dir, file), "r") as f:
                    trajectories[agent_id] = json.load(f)
        print(f"Loaded {len(trajectories)} trajectories.")
        return trajectories

    def extract_features(self, trajectories):
        """
        Extract features from trajectories for clustering.

        Args:
            trajectories (dict): Agent trajectories.

        Returns:
            np.ndarray: Feature matrix where each row corresponds to a trajectory.
        """
        print("Extracting features from trajectories...")
        features = []
        for agent_id, trajectory in trajectories.items():
            # Example feature extraction: mean and variance of state observations
            states = np.array([step["state"] for step in trajectory])
            mean_state = np.mean(states, axis=0)
            var_state = np.var(states, axis=0)
            features.append(np.concatenate([mean_state, var_state]))
        features = np.array(features)
        print(f"Extracted features shape: {features.shape}")
        return features

    def cluster_trajectories(self, features, num_clusters=None):
        """
        Perform hierarchical clustering on trajectory features.

        Args:
            features (np.ndarray): Feature matrix.
            num_clusters (int): Optional. Number of clusters to extract.

        Returns:
            np.ndarray: Cluster labels for each trajectory.
        """
        print("Performing hierarchical clustering...")
        linkage_matrix = linkage(features, method="ward")
        if num_clusters:
            cluster_labels = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
        else:
            cluster_labels = fcluster(linkage_matrix, t=1.15, criterion="inconsistent")

        # Save dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix, truncate_mode="level", p=5)
        plt.title("Hierarchical Clustering Dendrogram")
        dendrogram_path = os.path.join(self.output_dir, "dendrogram.png")
        plt.savefig(dendrogram_path)
        print(f"Dendrogram saved to {dendrogram_path}")

        return cluster_labels

    def visualize_clusters(self, features, cluster_labels):
        """
        Visualize clustered trajectories in a 2D space using t-SNE.

        Args:
            features (np.ndarray): Feature matrix.
            cluster_labels (np.ndarray): Cluster labels for each trajectory.
        """
        print("Visualizing clusters with t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=cluster_labels,
            cmap="viridis",
            s=50,
        )
        plt.colorbar(scatter, label="Cluster")
        plt.title("t-SNE Visualization of Trajectory Clusters")
        tsne_path = os.path.join(self.output_dir, "tsne_clusters.png")
        plt.savefig(tsne_path)
        print(f"t-SNE visualization saved to {tsne_path}")

    def analyze(self):
        """
        Main analysis workflow: load trajectories, extract features, cluster, and visualize.
        """
        print("Starting trajectory analysis...")
        trajectories = self.load_trajectories()
        features = self.extract_features(trajectories)
        cluster_labels = self.cluster_trajectories(features)
        self.visualize_clusters(features, cluster_labels)

        # Save clustering results
        clustering_results = {
            "agent_ids": list(trajectories.keys()),
            "cluster_labels": cluster_labels.tolist(),
        }
        results_path = os.path.join(self.output_dir, "clustering_results.json")
        with open(results_path, "w") as f:
            json.dump(clustering_results, f, indent=4)
        print(f"Clustering results saved to {results_path}")


if __name__ == "__main__":
    analyzer = Analyzer(trajectory_dir="trajectories", output_dir="analyzer_output")
    analyzer.analyze()
