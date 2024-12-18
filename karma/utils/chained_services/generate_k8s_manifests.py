import os
import yaml
import json
from collections import defaultdict


def create_kubernetes_manifests(json_file, output_dir):
    """
    Génère des manifests Kubernetes (Deployment et Service) à partir de la topologie JSON.

    Args:
        json_file (str): Chemin vers le fichier JSON de la topologie.
        output_dir (str): Dossier où les fichiers YAML seront générés.
    """
    # Charger la topologie depuis le fichier JSON
    with open(json_file, "r") as f:
        topology = json.load(f)

    # Calculer les services en aval à partir des connexions
    downstream_mapping = defaultdict(list)
    for connection in topology["connections"]:
        if connection["source"] != "INPUT" and connection["destination"] != "OUTPUT":
            downstream_mapping[connection["source"].lower()].append(
                connection["destination"].lower())

    # Créer le dossier de sortie pour les manifests
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir les services pour générer les manifests
    for service_name, properties in topology["services"].items():
        # Convertir le nom du service en minuscules
        service_name_lower = service_name.lower()

        image_name = f"{service_name_lower}_image"  # Nom de l'image Docker
        cpu_request = properties["cpu"]
        ram_request = f"{properties['ram']}Mi"
        max_replicas = properties["max_replicas"]
        min_replicas = properties["min_replicas"]
        downstream_services = ",".join(
            downstream_mapping[service_name_lower])  # Services en aval

        # Nom des fichiers YAML
        deployment_file = os.path.join(
            output_dir, f"{service_name_lower}_deployment.yaml")
        service_file = os.path.join(
            output_dir, f"{service_name_lower}_service.yaml")

        # Générer le Deployment
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": service_name_lower,
                "labels": {"app": service_name_lower}
            },
            "spec": {
                "replicas": min_replicas,
                "selector": {"matchLabels": {"app": service_name_lower}},
                "template": {
                    "metadata": {"labels": {"app": service_name_lower}},
                    "spec": {
                        "containers": [
                            {
                                "name": service_name_lower,
                                "image": f'{image_name}:latest',
                                "resources": {
                                    "requests": {"cpu": f"{cpu_request}m", "memory": ram_request}
                                },
                                "ports": [{"containerPort": 5000}],
                                "env": [
                                    {
                                        "name": "DOWNSTREAM_SERVICES",
                                        "value": downstream_services
                                    }
                                ],
                                "imagePullPolicy": "Never"
                            }
                        ]
                    }
                }
            }
        }

        # Générer le Service
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service_name_lower,
                "labels": {"app": service_name_lower}
            },
            "spec": {
                "selector": {"app": service_name_lower},
                "ports": [{"protocol": "TCP", "port": 5000, "targetPort": 5000}]
            }
        }

        # Sauvegarder le Deployment dans un fichier YAML
        with open(deployment_file, "w") as f:
            yaml.dump(deployment, f, default_flow_style=False)
        print(f"Deployment manifest created: {deployment_file}")

        # Sauvegarder le Service dans un fichier YAML
        with open(service_file, "w") as f:
            yaml.dump(service, f, default_flow_style=False)
        print(f"Service manifest created: {service_file}")


# Exemple d'utilisation
json_file = "topology.json"  # Chemin vers le fichier JSON
output_dir = "kubernetes_manifests"  # Dossier pour les fichiers YAML
create_kubernetes_manifests(json_file, output_dir)
