import subprocess
import time
import requests

from utils.cluster_util import KarmaCluster, generate_and_deploy_cluster
from pprint import pprint


class Transferer:
    def __init__(self, prometheus_url, namespace: str = "karma-cluster"):
        self.prometheus_url = prometheus_url
        self.namespace = namespace

    def get_state_from_prometheus(self, retries=3, backoff=2):
        """
        Récupère les métriques actuelles depuis Prometheus pour définir l'état.
        Réessaie plusieurs fois en cas d'échec.

        Args:
            retries (int): Nombre maximum de tentatives.
            backoff (int): Temps d'attente entre chaque tentative (en secondes).
        """
        try:
            queries = {
                "cpu_usage": "sum(rate(container_cpu_usage_seconds_total[5m])) by (pod)",
                "memory_usage": "sum(container_memory_working_set_bytes) by (pod)",
                "replicas": "sum(kube_deployment_status_replicas) by (deployment)"
            }

            state = {}

            for attempt in range(retries):
                try:
                    # Collecter les résultats pour chaque métrique
                    for key, query in queries.items():
                        response = requests.get(
                            f"{self.prometheus_url}/api/v1/query",
                            params={"query": query},
                            timeout=5  # Timeout pour chaque requête
                        )
                        response.raise_for_status()
                        results = response.json()["data"]["result"]

                        # Structurer les données dans le dictionnaire state
                        state[key] = {
                            result["metric"].get("pod", result["metric"].get("deployment")): float(result["value"][1])
                            for result in results
                        }

                    return state  # Retourner l'état si tout fonctionne

                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(backoff)
                    else:
                        raise  # Relève l'exception après le nombre maximal de tentatives

        except Exception as e:
            print(f"Error retrieving metrics from Prometheus: {e}")
            return {}

    def scale_deployment(self, deployment_name, replicas, timeout=60, interval=5):
        """
        Met à jour le nombre de réplicas pour un déploiement donné et attend que le changement soit effectif.

        Args:
            deployment_name (str): Nom du déploiement Kubernetes.
            replicas (int): Nouveau nombre de réplicas.
            timeout (int): Temps maximum d'attente en secondes pour que le scaling prenne effet (par défaut : 60).
            interval (int): Intervalle de vérification en secondes (par défaut : 5).

        Returns:
            bool: True si le scaling a réussi et que le nombre attendu de réplicas est atteint, False sinon.
        """
        try:
            # Exécuter la commande kubectl pour mettre à jour les réplicas
            response = subprocess.run(
                [
                    "kubectl", "scale", "deployment", deployment_name,
                    f"--replicas={replicas}", "-n", self.namespace
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True  # Lever une exception si la commande échoue
            )
            print(
                f"Scaling deployment '{deployment_name}' to {replicas} replicas...")

            # Attente active pour vérifier le scaling
            elapsed_time = 0
            while elapsed_time < timeout:
                try:
                    # Obtenir l'état actuel des réplicas
                    state = self.get_state_from_prometheus()
                    current_replicas = state.get(
                        "replicas", {}).get(deployment_name, None)

                    if current_replicas == replicas:
                        print(
                            f"Deployment '{deployment_name}' successfully scaled to {replicas} replicas.")
                        return True

                except Exception as e:
                    print(
                        f"Error checking replicas for '{deployment_name}': {e}")

                time.sleep(interval)
                elapsed_time += interval

            print(
                f"Timeout waiting for deployment '{deployment_name}' to scale to {replicas} replicas.")
            return False

        except subprocess.CalledProcessError as e:
            print(
                f"Failed to scale deployment '{deployment_name}': {e.stderr}")
            return False


if __name__ == '__main__':

    print("=" * 20)
    print("transferer_test.py")
    print("=" * 20)

    kc = KarmaCluster("../../utils/install_topology.json")

    kc.validate_prometheus()

    kc = generate_and_deploy_cluster("../../utils/install_topology.json")

    time.sleep(5)

    tc = Transferer(prometheus_url="http://localhost:9090",
                    namespace="karma-cluster")

    try:

        pprint(tc.get_state_from_prometheus())

        tc.scale_deployment("a", 1)
        pprint(tc.get_state_from_prometheus())

        tc.scale_deployment("a", 3)
        pprint(tc.get_state_from_prometheus())

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Nettoyage optionnel
        kc.delete_cluster()
