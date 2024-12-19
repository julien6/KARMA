from collections import defaultdict
import os
import subprocess
import json
import time

import yaml


class KarmaCluster:
    def __init__(self, topology_path, output_dir="cluster_artifacts", manifests_dir="kubernetes_manifests", cluster_name="karma-cluster"):
        """
        Initialise le cluster Karma avec les chemins et noms nécessaires.

        Args:
            topology_path (str): Chemin vers le fichier JSON décrivant la topologie.
            output_dir (str): Dossier pour les fichiers générés (par défaut: cluster_artifacts).
            manifests_dir (str): Dossier pour les manifests Kubernetes (par défaut: kubernetes_manifests).
            cluster_name (str): Nom du cluster Kubernetes Kind (par défaut: karma-cluster).
        """
        self.topology_path = topology_path
        self.output_dir = output_dir
        self.manifests_dir = manifests_dir
        self.cluster_name = cluster_name

    def check_prerequisites(self):
        """
        Vérifie la présence des outils nécessaires : docker, kind, kubectl.
        """
        print("Checking prerequisites...")
        required_tools = ["docker", "kind", "kubectl"]
        for tool in required_tools:
            try:
                subprocess.run([tool, "--help"],
                               stdout=subprocess.PIPE, check=True)
                print(f"{tool} is available.")
            except subprocess.CalledProcessError:
                raise EnvironmentError(
                    f"{tool} is not installed or accessible in PATH.")

    def generate_mocks(self):
        """
        Génère les services mock Python et leurs Dockerfiles à partir de la topologie.
        """
        print("Generating mock services and Dockerfiles...")
        if not os.path.exists(self.topology_path):
            raise FileNotFoundError(
                f"Topology file not found: {self.topology_path}")

        # Charger la topologie depuis le fichier JSON
        with open(self.topology_path, "r") as f:
            topology = json.load(f)

        # Calculer les services en aval à partir des connexions
        downstream_mapping = defaultdict(list)
        for connection in topology["connections"]:
            if connection["source"] != "INPUT" and connection["destination"] != "OUTPUT":
                downstream_mapping[connection["source"].lower()].append(
                    connection["destination"].lower())

        # Créer le dossier de sortie principal
        os.makedirs(self.output_dir, exist_ok=True)

        # Parcourir les services pour générer les mocks et les Dockerfiles
        for service_name, properties in topology["services"].items():
            service_name_lower = service_name.lower()
            computation_throughput = properties["computation_throughput"]
            cpu = properties["cpu"]
            ram = properties["ram"]
            # Adresses des services aval
            downstream_services = ",".join(
                downstream_mapping[service_name_lower])

            # Sous-dossier dédié pour chaque service dans cluster_artifacts
            service_dir = os.path.join(self.output_dir, service_name_lower)
            os.makedirs(service_dir, exist_ok=True)

            # Fichier mock Python
            mock_file = os.path.join(service_dir, "mock.py")
            with open(mock_file, "w") as f:
                f.write(f"""from flask import Flask, request, jsonify
import os
import time
import requests

app = Flask(__name__)

# Configuration des adresses des services en aval
DOWNSTREAM_SERVICES = os.getenv("DOWNSTREAM_SERVICES", "").split(",")

# Configuration du throughput et du délai de traitement
THROUGHPUT = int(os.getenv("THROUGHPUT", {computation_throughput}))  # En données par seconde
PROCESSING_TIME = 1 / THROUGHPUT  # Temps simulé pour chaque unité de donnée

def consume_resources(cpu_percent, ram_mb):
    \"\"\" Simule la consommation de CPU et de RAM \"\"\"
    busy_time = cpu_percent / 100
    idle_time = 1 - busy_time
    start_time = time.time()
    while time.time() - start_time < busy_time:
        pass  # Boucle active pour consommer du CPU
    time.sleep(idle_time)

    data = []
    try:
        for _ in range(int(ram_mb * 1024 / 64)):
            data.append(bytearray(64 * 1024))
        time.sleep(1)
    except MemoryError:
        print("Warning: Memory allocation failed. System may not have enough RAM.")
    finally:
        del data

@app.route("/process", methods=["POST"])
def process():
    \"\"\" Simule la réception, le traitement et la transmission des données \"\"\"
    data = request.json
    input_data = data.get("input", "0000")  # Données par défaut si rien n'est passé
    print(f"Received data: {{input_data}}")

    # Simuler le traitement
    size = len(input_data)
    time.sleep(size * PROCESSING_TIME)
    consume_resources(cpu_percent={cpu / 1000 * 100}, ram_mb={ram})

    # Envoyer des requêtes aux services en aval
    responses = []
    for service in DOWNSTREAM_SERVICES:
        if service:
            try:
                response = requests.post(
                    f"http://{{service}}:5000/process",
                    json={{"input": input_data}}
                )
                responses.append({{"service": service, "status": response.status_code}})
            except Exception as e:
                responses.append({{"service": service, "error": str(e)}})

    print(f"Processed data sent downstream: {{responses}}")
    return jsonify({{"status": "processed", "responses": responses}}), 200

if __name__ == "__main__":
    print("Starting mock service for {service_name_lower}...")
    app.run(host="0.0.0.0", port=5000)
""")
            print(
                f"Mock with HTTP API generated for service {service_name_lower}: {mock_file}")

            # Générer le Dockerfile
            dockerfile_path = os.path.join(service_dir, "Dockerfile")
            with open(dockerfile_path, "w") as dockerfile:
                dockerfile.write(f"""# Base Python image
    FROM python:3.10-slim

    WORKDIR /app

    COPY mock.py /app/mock.py

    RUN pip install --no-cache-dir flask requests

    CMD ["python", "/app/mock.py"]
    """)
            print(
                f"Dockerfile generated for service {service_name_lower}: {dockerfile_path}")

        print(f"Mocks generated in {self.output_dir}.")

    def build_docker_images(self):
        """
        Construit les images Docker pour chaque service généré.
        """
        print("Building Docker images for services...")

        # Vérifier si le dossier de sortie existe
        if not os.path.exists(self.output_dir):
            print(f"Error: The directory '{self.output_dir}' does not exist.")
            return

        # Parcourir les sous-dossiers des services
        for service_name in os.listdir(self.output_dir):
            service_dir = os.path.join(self.output_dir, service_name)

            # Vérifier que c'est bien un dossier
            if not os.path.isdir(service_dir):
                continue

            # Construire le chemin du Dockerfile
            dockerfile_path = os.path.join(service_dir, "Dockerfile")

            # Vérifier que le Dockerfile existe
            if not os.path.exists(dockerfile_path):
                print(
                    f"Warning: Dockerfile not found for service '{service_name}' in {service_dir}. Skipping...")
                continue

            # Nom de l'image Docker
            image_name = f"{service_name.lower()}_image"

            # Construire l'image Docker
            print(f"Building Docker image for service '{service_name}'...")
            try:
                subprocess.run(
                    ["docker", "build", "-t", image_name, service_dir],
                    check=True
                )
                print(f"Docker image built successfully: {image_name}")
            except subprocess.CalledProcessError as e:
                print(
                    f"Error: Failed to build Docker image for service '{service_name}'. {e}")
                continue

        print("Docker images built successfully.")

    def generate_manifests(self):
        """
        Génère les manifests Kubernetes pour les services.
        """
        print("Generating Kubernetes manifests...")
        if not os.path.exists(self.topology_path):
            raise FileNotFoundError(
                f"Topology file not found: {self.topology_path}")

        # Charger la topologie depuis le fichier JSON
        with open(self.topology_path, "r") as f:
            topology = json.load(f)

        # Calculer les services en aval à partir des connexions
        downstream_mapping = defaultdict(list)
        for connection in topology["connections"]:
            if connection["source"] != "INPUT" and connection["destination"] != "OUTPUT":
                downstream_mapping[connection["source"].lower()].append(
                    connection["destination"].lower())

        # Créer le dossier de sortie pour les manifests
        os.makedirs(self.manifests_dir, exist_ok=True)

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
                self.manifests_dir, f"{service_name_lower}_deployment.yaml")
            service_file = os.path.join(
                self.manifests_dir, f"{service_name_lower}_service.yaml")

            # Générer le Deployment
            deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": service_name_lower,
                    "namespace": self.cluster_name,
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
                    "namespace": self.cluster_name,
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

        print(f"Manifests generated in {self.manifests_dir}.")

    def create_cluster(self):
        """
        Crée le cluster Kubernetes avec Kind et applique les manifests.
        """
        print(f"Creating Kubernetes cluster '{self.cluster_name}'...")

        try:

            # Étape 1 : Créer le cluster Kind
            print(f"Creating Kind cluster '{self.cluster_name}'...")
            subprocess.run(
                ["kind", "create", "cluster", "--name", self.cluster_name],
                check=True
            )
            print(f"Cluster '{self.cluster_name}' created successfully.")

            # Étape 2 : Créer un namespace avec le nom du cluster
            print(
                f"Checking if namespace '{self.cluster_name}' already exists...")
            result = subprocess.run(
                ["kubectl", "get", "namespace", self.cluster_name],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode == 0:
                print(
                    f"Namespace '{self.cluster_name}' already exists. Skipping creation.")
            else:
                print(f"Creating namespace '{self.cluster_name}'...")
                subprocess.run(
                    ["kubectl", "create", "namespace", self.cluster_name],
                    check=True
                )
            print(f"Namespace '{self.cluster_name}' created successfully.")

            # Étape 3 : Charger les images Docker dans le cluster
            print("Loading Docker images into the Kind cluster...")
            images_dir = "cluster_artifacts"
            if not os.path.exists(images_dir):
                print(
                    f"Error: Directory '{images_dir}' does not exist. Make sure the images are built.")
                return

            for service_name in os.listdir(images_dir):
                service_dir = os.path.join(images_dir, service_name)
                image_name = f"{service_name.lower()}_image"

                # Vérifier si l'image Docker existe localement
                result = subprocess.run(
                    ["docker", "images", "-q", image_name],
                    stdout=subprocess.PIPE,
                    text=True
                )
                if not result.stdout.strip():
                    print(
                        f"Error: Docker image '{image_name}' not found. Build the image before proceeding.")
                    return

                # Charger l'image dans le cluster
                print(
                    f"Loading image '{image_name}' into the cluster '{self.cluster_name}'...")
                subprocess.run(
                    ["kind", "load", "docker-image", image_name,
                        "--name", self.cluster_name],
                    check=True
                )
                print(f"Image '{image_name}' loaded successfully.")

            # Étape 4 : Appliquer les manifests Kubernetes
            if not os.path.exists(self.manifests_dir):
                print(
                    f"Error: Directory '{self.manifests_dir}' does not exist. Generate the manifests first.")
                return

            # Étape 5 : Appliquer les manifests Kubernetes dans le namespace spécifique
            print("Applying Kubernetes manifests...")
            subprocess.run(
                ["kubectl", "apply", "-f", self.manifests_dir,
                    "-n", self.cluster_name],
                check=True
            )
            print(
                f"Kubernetes manifests applied successfully from directory '{self.manifests_dir}'.")

            # Étape 6 : Vérifier les pods et services
            print("Checking cluster status...")
            subprocess.run(["kubectl", "get", "pods", "--namespace", self.cluster_name], check=True)
            subprocess.run(["kubectl", "get", "services", "--namespace", self.cluster_name], check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error during setup: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def wait_for_pods_ready(self, timeout=300):
        """
        Attend que tous les pods soient prêts dans un namespace donné.

        Args:
            timeout (int): Durée maximale d'attente en secondes (par défaut: 300).
        """
        print(f"Waiting for all pods to be ready in namespace '{self.cluster_name}'...")
        for _ in range(timeout // 5):
            result = subprocess.run(
                ["kubectl", "get", "pods", "--namespace", self.cluster_name,
                    "--field-selector=status.phase!=Running"],
                stdout=subprocess.PIPE, text=True
            )
            if not result.stdout.strip():
                print("All pods are ready.")
                return
            time.sleep(5)
        raise TimeoutError("Timeout waiting for pods to be ready.")

    def delete_cluster(self):
        """
        Supprime le cluster Kind.
        """
        print(f"Deleting Kubernetes cluster '{self.cluster_name}'...")
        subprocess.run(["kind", "delete", "cluster", "--name",
                       self.cluster_name], check=True)
        print(f"Cluster '{self.cluster_name}' deleted successfully.")


if __name__ == '__main__':

    print("=" * 20)
    print("cluster_util_test.py")
    print("=" * 20)

    kc = KarmaCluster("install_topology.json")

    try:
        kc.check_prerequisites()
        kc.generate_mocks()
        kc.build_docker_images()
        kc.generate_manifests()
        kc.create_cluster()
        kc.wait_for_pods_ready()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Nettoyage optionnel
        kc.delete_cluster()
