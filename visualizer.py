"""
Responsibility:
- Visualize training and experiment results.
- Includes:
  - Training loss curves
  - Validation metrics over epochs
  - Graph structure visualization
"""


import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

class Visualizer:
    def __init__(self, save_path="experiments/"):
        self.save_path = save_path

    def plot_training_loss(self, losses, exp_path):
        plt.figure()
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss Curve")
        plt.savefig(f"{exp_path}/training_loss.png")
        plt.close()

    def plot_validation_metrics(self, metrics, exp_path):
        plt.figure()
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend()
        plt.title("Validation Metrics")
        plt.savefig(f"{exp_path}/validation_metrics.png")
        plt.close()

    def plot_graph(self, graph_data, exp_path):
        G = to_networkx(graph_data, to_undirected=True)
        plt.figure(figsize=(6,6))
        nx.draw(G, node_size=20, alpha=0.6)
        plt.title("Graph Structure")
        plt.savefig(f"{exp_path}/graph_structure.png")
        plt.close()
