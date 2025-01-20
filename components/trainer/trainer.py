import yaml
import torch
import os
import numpy as np
import shutil
import importlib

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, topology_path, model_path, reward_function, env_name="karma_cluster", map_id=0, num_agents=4, max_steps=200):
        """
        Initialisation de la classe Trainer.

        Args:
            topology_path (str): Chemin vers le fichier JSON de topologie.
            model_path (str): Chemin vers le fichier .pth du modèle de transition.
            reward_function (callable): Fonction de récompense.
            env_name (str): Nom de l'environnement généré.
            max_steps (int): Nombre maximal de pas dans un épisode.
        """
        self.topology_path = topology_path
        self.model_path = model_path
        self.reward_function = reward_function
        self.env_name = env_name
        self.max_steps = max_steps
        self.map_id = map_id
        self.num_agents = num_agents

        # Enregistrement de l'environnement dans MARLlib et patches
        self.integrate_cluster_env_in_MARLlib()

    def generate_yaml(self):
        """
        Génère et sauvegarde un fichier YAML de configuration pour MARLlib.
        """
        config = {
            "env": self.env_name,
            "env_args": {
                "map_name": f"cluster_{self.map_id}",
                "topology_path": str(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.topology_path))),
                "model_path": str(os.path.realpath(os.path.join(os.path.dirname(__file__), self.model_path))),
                "max_steps": self.max_steps,
                "num_agents": self.num_agents
            },
            "mask_flag": False,
            "global_state_flag": True,
            "opp_action_in_cc": False,
            "agent_level_batch_update": True,
        }

        path = os.path.join(os.path.dirname(
            marl.__file__), "../envs/base_env/config/{}.yaml".format(self.env_name))
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
        print(f"Fichier YAML généré et sauvegardé à {path}.")

    def integrate_cluster_env_in_MARLlib(self):

        # Generate and save KARMA Cluster Configuration in MARLlib
        self.generate_yaml()

        # Copy PettingZoo and MARLlib KARMA environments in MARLlib
        source = os.path.join(os.path.dirname(__file__), "karma_cluster.py")
        destination = os.path.join(os.path.dirname(
            marl.__file__), "../envs/base_env/karma_cluster.py")
        shutil.copy(source, destination)
        print("Environnement KARMA patché dans MARLlib")


        # Recharger le module pour inclure les modifications
        import marllib.envs.base_env
        importlib.reload(marllib)
        from marllib.envs.base_env.karma_cluster import RLlibKarmaCluster, RLlibKarmaCluster_FCOOP

        try:
            ENV_REGISTRY["karma_cluster"] = RLlibKarmaCluster
        except Exception as e:
            ENV_REGISTRY["karma_cluster"] = str(e)

        try:
            COOP_ENV_REGISTRY["karma_cluster"] = RLlibKarmaCluster_FCOOP
        except Exception as e:
            COOP_ENV_REGISTRY["karma_cluster"] = str(e)

        time.sleep(0.01)

        print("Environnement KARMA enregistré dans MARLlib")

    def train(self, algo="mappo", stop_criteria=None, model_config=None, results_path="results"):
        """
        Lance l'entraînement avec MARLlib.

        Args:
            algo (str): Algorithme MARLlib à utiliser (par exemple, "mappo").
            stop_criteria (dict): Critères d'arrêt pour l'entraînement.
            model_config (dict): Configuration du modèle (par exemple, architecture du réseau).
            results_path (str): Chemin pour sauvegarder les résultats.
        """

        # Charger l'environnement MARLlib
        env = marl.make_env(
            environment_name=self.env_name,
            map_name=f"cluster_{self.map_id}"
        )

        # Sélectionner l'algorithme MARLlib
        algorithm = getattr(marl.algos, algo)(hyperparam_source="test")

        # Configuration par défaut pour le modèle
        if model_config is None:
            model_config = {
                "core_arch": "mlp",  # Architecture centrale
                "encode_layer": "128-256",  # Couches d'encodage
            }

        # Construire le modèle
        model = marl.build_model(env, algorithm, model_config)

        # Critères d'arrêt par défaut
        if stop_criteria is None:
            stop_criteria = {"training_iteration": 50}

        # Démarrer l'entraînement
        results = algorithm.fit(
            env,
            model,
            stop=stop_criteria,
            local_mode=True,
            num_gpus=0,  # Désactiver l'utilisation des GPU
            num_workers=2,
            share_policy="group",
            checkpoint_freq=10,
        )

        print("ok4")

        # Sauvegarder le checkpoint final
        best_checkpoint = results.get_best_checkpoint()
        checkpoint_dir = os.path.join(results_path, algo, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pth")
        marl.save(best_checkpoint, checkpoint_path)

        print(
            f"Entraînement terminé. Checkpoint final sauvegardé à {checkpoint_path}.")


if __name__ == "__main__":

    import time
    from components.modeler.modeler import Modeler

    print("=" * 20)
    print("trainer_test.py")
    print("=" * 20)

    modeler_process = None

    try:
        # 1. Créer un singleton de la classe Modeler
        print("Initialisation du Modeler en tant que singleton...")
        db_name = "../modeler/transitions.db"
        topology_path = "../../utils/install_topology.json"
        model_path = "mlp_model.pth"

        # Créer l'objet Modeler
        modeler = Modeler(
            db_path=db_name, topology_path=topology_path, model_path=model_path)

        # Démarrer le réentraînement périodique dans un processus séparé
        modeler_process = modeler.start_retrain_loop(interval=300)

        # Pause pour laisser le Modeler générer `mlp_model.pth`
        print("Pause pour permettre au Modeler de générer le fichier `mlp_model.pth`...")
        time.sleep(5)

        # Vérifier que le fichier `mlp_model.pth` existe
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Le fichier {model_path} n'a pas été généré par le Modeler.")

        print(f"Fichier {model_path} détecté avec succès.")

        # 2. Initialiser le Trainer
        print("Initialisation du Trainer...")

        def reward_function(state, action, next_state):
            """Exemple de fonction de récompense pour les tests."""
            return -np.sum(np.abs(next_state - state))

        trainer = Trainer(
            topology_path=topology_path,
            model_path=model_path,
            reward_function=reward_function,
        )

        from marllib import marl

        # 3. Lancer l'entraînement avec MARLlib
        print("Lancement de l'entraînement avec MARLlib...")
        stop_criteria = {
            "episode_reward_mean": 500,
            "training_iteration": 100,
        }
        model_config = {
            "core_arch": "mlp",
            "encode_layer": "128-128",
        }
        trainer.train(algo="mappo", stop_criteria=stop_criteria,
                      model_config=model_config)

    except KeyboardInterrupt:
        print("Arrêt manuel du processus...")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
    finally:
        # Arrêter proprement le processus du Modeler
        if modeler_process.is_alive():
            print("Arrêt du processus Modeler...")
            modeler_process.terminate()
            modeler_process.join()
