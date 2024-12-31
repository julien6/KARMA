import os
import subprocess


def setup_kubernetes_cluster_with_kind(cluster_name="chained-services", manifests_dir="kubernetes_manifests"):
    """
    Crée un cluster Kubernetes avec Kind, charge les images Docker générées et applique les manifests Kubernetes.

    Args:
        cluster_name (str): Nom du cluster Kind (par défaut "chained-services").
        manifests_dir (str): Chemin du dossier contenant les manifests Kubernetes (par défaut "kubernetes_manifests").
    """
    try:
        # Étape 1 : Créer le cluster Kind
        print(f"Creating Kind cluster '{cluster_name}'...")
        subprocess.run(
            ["kind", "create", "cluster", "--name", cluster_name],
            check=True
        )
        print(f"Cluster '{cluster_name}' created successfully.")

        # Étape 2 : Charger les images Docker dans le cluster
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
                f"Loading image '{image_name}' into the cluster '{cluster_name}'...")
            subprocess.run(
                ["kind", "load", "docker-image", image_name, "--name", cluster_name],
                check=True
            )
            print(f"Image '{image_name}' loaded successfully.")

        # Étape 3 : Appliquer les manifests Kubernetes
        if not os.path.exists(manifests_dir):
            print(
                f"Error: Directory '{manifests_dir}' does not exist. Generate the manifests first.")
            return

        print("Applying Kubernetes manifests...")
        subprocess.run(
            ["kubectl", "apply", "-f", manifests_dir],
            check=True
        )
        print(
            f"Kubernetes manifests applied successfully from directory '{manifests_dir}'.")

        # Étape 4 : Vérifier les pods et services
        print("Checking cluster status...")
        subprocess.run(["kubectl", "get", "pods"], check=True)
        subprocess.run(["kubectl", "get", "services"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# Exemple d'utilisation
setup_kubernetes_cluster_with_kind()
