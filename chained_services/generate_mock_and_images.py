import os
import json
import subprocess
from collections import defaultdict

def generate_mock_services_with_http(json_file, output_dir):
    """
    Génère des fichiers mock Python avec communication HTTP via Flask.

    Args:
        json_file (str): Chemin vers le fichier JSON de topologie.
        output_dir (str): Dossier unique "cluster_artifacts" où tous les fichiers seront générés.
    """
    # Charger la topologie depuis le fichier JSON
    with open(json_file, "r") as f:
        topology = json.load(f)

    # Calculer les services en aval à partir des connexions
    downstream_mapping = defaultdict(list)
    for connection in topology["connections"]:
        if connection["source"] != "INPUT" and connection["destination"] != "OUTPUT":
            downstream_mapping[connection["source"].lower()].append(connection["destination"].lower())

    # Créer le dossier de sortie principal
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir les services pour générer les mocks et les Dockerfiles
    for service_name, properties in topology["services"].items():
        service_name_lower = service_name.lower()
        computation_throughput = properties["computation_throughput"]
        cpu = properties["cpu"]
        ram = properties["ram"]
        downstream_services = ",".join(downstream_mapping[service_name_lower])  # Adresses des services aval

        # Sous-dossier dédié pour chaque service dans cluster_artifacts
        service_dir = os.path.join(output_dir, service_name_lower)
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
        print(f"Mock with HTTP API generated for service {service_name_lower}: {mock_file}")

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
        print(f"Dockerfile generated for service {service_name_lower}: {dockerfile_path}")



def build_docker_images(output_dir):
    """
    Construit les images Docker pour chaque service à partir des fichiers générés.

    Args:
        output_dir (str): Dossier contenant les sous-dossiers des services avec les Dockerfiles et mocks.
    """
    # Vérifier si le dossier de sortie existe
    if not os.path.exists(output_dir):
        print(f"Error: The directory '{output_dir}' does not exist.")
        return

    # Parcourir les sous-dossiers des services
    for service_name in os.listdir(output_dir):
        service_dir = os.path.join(output_dir, service_name)

        # Vérifier que c'est bien un dossier
        if not os.path.isdir(service_dir):
            continue

        # Construire le chemin du Dockerfile
        dockerfile_path = os.path.join(service_dir, "Dockerfile")

        # Vérifier que le Dockerfile existe
        if not os.path.exists(dockerfile_path):
            print(f"Warning: Dockerfile not found for service '{service_name}' in {service_dir}. Skipping...")
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
            print(f"Error: Failed to build Docker image for service '{service_name}'. {e}")
            continue






# Exemple d'utilisation
json_file = "topology.json"  # Chemin vers le fichier JSON
output_dir = "cluster_artifacts"  # Nom du dossier unique
generate_mock_services_with_http(json_file, output_dir)
build_docker_images(output_dir)