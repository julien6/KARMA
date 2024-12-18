import json
import networkx as nx
import matplotlib.pyplot as plt


def load_topology_and_visualize(json_file, output_image):
    # Charger la topologie depuis le fichier JSON
    with open(json_file, "r") as f:
        topology = json.load(f)

    # Initialiser un graphe orienté
    G = nx.DiGraph()

    # Ajouter les services comme nœuds avec leurs propriétés
    for service, properties in topology["services"].items():
        # Format des propriétés en texte pour affichage
        node_label = "\n".join(
            [f"{key}: {value}" for key, value in properties.items()])
        G.add_node(service, label=node_label)

    # Ajouter explicitement les nœuds "INPUT" et "OUTPUT" avec des labels clairs
    G.add_node("INPUT", label="INPUT")
    G.add_node("OUTPUT", label="OUTPUT")

    # Ajouter les connexions comme arêtes avec leurs propriétés
    for connection in topology["connections"]:
        source = connection["source"]
        destination = connection["destination"]
        throughput = connection["throughput"]
        G.add_edge(source, destination, throughput=throughput)

    # Définir les positions des nœuds (spring_layout pour une meilleure visibilité)
    pos = nx.spring_layout(G)

    # Dessiner les nœuds avec les labels formatés
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Dessiner les arêtes avec des flèches orientées
    nx.draw_networkx_edges(
        G,
        pos,
        # Courbure optionnelle pour éviter les chevauchements
        connectionstyle="arc3,rad=0.1",
        arrowstyle="->",  # Style des flèches
        arrowsize=35,
        edge_color="black",
        arrows=True
    )

    # Ajouter les labels des arêtes (débits)
    edge_labels = nx.get_edge_attributes(G, "throughput")
    edge_labels_formatted = {edge: f"{value} Mbps" if value != -
                             1 else "?" for edge, value in edge_labels.items()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels_formatted, font_size=8)

    # Afficher le graphe
    plt.title("Deployment Topology on Kubernetes")
    plt.axis("off")
    plt.savefig(output_image)  # Enregistrer l'image
    plt.show()


# Exemple d'utilisation
json_file = "topology.json"  # Remplacez par le chemin de votre fichier JSON
output_image = "topology_graph.png"
load_topology_and_visualize(json_file, output_image)
