import numpy as np
import os
import torch

from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
# Assurez-vous que le Modeler est disponible et bien importé
from modeler import Modeler
from marllib.envs.pettingzoo import PettingZooEnvWrapper
from marllib import marl

class Trainer:
    def __init__(self, modeler: Modeler, reward_function, algo_name="ppo"):
        """
        Initialisation du composant Trainer.

        Args:
            modeler (Modeler): Instance du composant Modeler pour prédire les transitions d'état.
            reward_function (callable): Fonction de récompense globale qui calcule la récompense à partir d'un état et d'une action.
            algo_name (str): Nom de l'algorithme MARLlib à utiliser (par défaut : "ppo").
        """
        self.modeler = modeler
        self.reward_function = reward_function
        self.env = None  # Environnement PettingZoo, créé dynamiquement
        self.agents = self._initialize_agents()  # Liste des noms d'agents
        self.algo_name = algo_name
        self.marl_algorithm = None  # Algorithme MARLlib configuré

    def _initialize_agents(self):
        """
        Initialise les noms des agents en fonction du nombre de services.

        Returns:
            list: Liste des noms d'agents.
        """
        return [f"agent_{i}" for i in range(self.modeler.num_services)]

    def create_pettingzoo_env(self):
        """
        Crée un environnement PettingZoo basé sur la fonction de transition d'état et la fonction de récompense.
        """
        class CustomEnv(ParallelEnv):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer
                self.agents = trainer.agents.copy()
                self.possible_agents = trainer.agents.copy()

                # Définir les espaces d'état et d'action
                self.observation_spaces = {
                    agent: spaces.Box(
                        low=0,
                        high=np.inf,
                        shape=(trainer.modeler.num_metrics,),
                        dtype=np.float32,
                    )
                    for agent in self.agents
                }

                self.action_spaces = {
                    agent: spaces.Box(
                        low=-5,
                        high=5,
                        shape=(1,),
                        dtype=np.float32,
                    )
                    for agent in self.agents
                }

                # Initialiser les états
                self.states = None
                self.reset()

            def reset(self, seed=None, options=None):
                """
                Réinitialise l'environnement et renvoie les observations initiales.

                Returns:
                    dict: Observations initiales pour chaque agent.
                """
                self.agents = self.possible_agents.copy()
                self.states = {
                    agent: np.random.rand(self.trainer.modeler.num_metrics).astype(
                        np.float32
                    )
                    for agent in self.agents
                }
                return self.states

            def step(self, actions):
                """
                Exécute une étape dans l'environnement.

                Args:
                    actions (dict): Dictionnaire des actions effectuées par chaque agent.

                Returns:
                    tuple: observations, rewards, terminations, truncations, infos
                """
                # Initialiser les dictionnaires
                next_states = {}
                rewards = {}
                terminations = {agent: False for agent in self.agents}
                truncations = {agent: False for agent in self.agents}
                infos = {agent: {} for agent in self.agents}

                # Calculer l'état suivant et les récompenses
                for agent in self.agents:
                    current_state = self.states[agent]
                    action = actions[agent]
                    next_state = self.trainer.modeler.predict_next_state(
                        state=current_state, action=action
                    )
                    next_states[agent] = next_state
                    rewards[agent] = self.trainer.reward_function(
                        state=current_state, action=action, next_state=next_state
                    )

                # Mettre à jour les états
                self.states = next_states
                return self.states, rewards, terminations, truncations, infos

        # Créer une instance de l'environnement
        self.env = CustomEnv(self)
        print("Environnement PettingZoo créé avec succès.")

    def get_env(self):
        """
        Retourne l'environnement PettingZoo.

        Returns:
            ParallelEnv: L'environnement PettingZoo créé.
        """
        if self.env is None:
            self.create_pettingzoo_env()
        return self.env

    def configure_marl_algorithm(self, env):
        """
        Configure l'algorithme MARLlib avec l'environnement donné.

        Args:
            env (PettingZooEnvWrapper): Environnement PettingZooWrapper pour MARLlib.
        """
        algo = marl(self.algo_name)
        model_config = {
            "core_arch": "mlp",  # Architecture pour le modèle
            "encoder": "mlp",    # Type d'encodeur
        }

        self.marl_algorithm = algo(env, model_config)
        print(f"Algorithme {self.algo_name} configuré avec succès.")

    def train_agents(self, train_episodes=1000, test_episodes=100, reward_threshold=100.0, std_threshold=10.0):
        """
        Entraîne les agents en utilisant MARLlib jusqu'à atteindre les conditions de convergence.

        Args:
            train_episodes (int): Nombre d'épisodes d'entraînement.
            test_episodes (int): Nombre d'épisodes de test pour évaluer la convergence.
            reward_threshold (float): Seuil de récompense moyenne pour arrêter l'entraînement.
            std_threshold (float): Seuil de l'écart-type de la récompense pour arrêter l'entraînement.
        """
        if self.env is None:
            self.create_pettingzoo_env()

        # Envelopper l'environnement avec PettingZooWrapper de MARLlib
        wrapped_env = PettingZooEnvWrapper(self.env)

        # Configurer l'algorithme MARLlib
        self.configure_marl_algorithm(wrapped_env)

        print("Début de l'entraînement...")
        results = self.marl_algorithm.train(
            train_episodes=train_episodes, test_episodes=test_episodes, reward_threshold=reward_threshold, std_threshold=std_threshold
        )

        # Sauvegarder les politiques entraînées
        policies_dir = "./trained_policies"
        os.makedirs(policies_dir, exist_ok=True)
        for agent, policy in results["policies"].items():
            policy_path = os.path.join(policies_dir, f"{agent}_policy.pth")
            torch.save(policy, policy_path)
            print(f"Politique de {agent} sauvegardée à : {policy_path}")

        print("Entraînement terminé.")


if __name__ == '__main__':

    from modeler import Modeler, create_and_populate_database

    # Configuration de base
    DB_NAME = "transitions.db"
    TOPOLOGY_PATH = "./topology.json"  # Fichier JSON avec les informations sur le cluster
    MODEL_PATH = "mlp_model.pth"
    NUM_TRANSITIONS = 5000  # Nombre de transitions générées
    NUM_SERVICES = 4        # Nombre de services
    NUM_METRICS = 5         # Nombre de métriques par service

    # Création et peuplement de la base de données
    if not os.path.exists(DB_NAME):
        print("Création et peuplement de la base de données...")
        create_and_populate_database(
            db_name=DB_NAME,
            num_transitions=NUM_TRANSITIONS,
            num_services=NUM_SERVICES,
            num_metrics=NUM_METRICS
        )

    # Instanciation du Modeler
    modeler = Modeler(db_path=DB_NAME, topology_path=TOPOLOGY_PATH, model_path=MODEL_PATH)

    # Création de l'environnement PettingZoo
    class SimpleClusterEnv:
        def __init__(self, modeler):
            self.modeler = modeler
            self.num_services = modeler.num_services
            self.num_metrics = modeler.num_metrics
            self.observation_space = marl.env.observation_space(
                shape=(self.num_services, self.num_metrics), low=0, high=1
            )
            self.action_space = marl.env.action_space(
                shape=(self.num_services,), low=-5, high=5
            )

        def reset(self):
            # Générer un état initial aléatoire
            self.state = np.random.rand(self.num_services, self.num_metrics)
            return self.state

        def step(self, actions):
            try:
                # Prédire l'état suivant avec le Modeler
                next_state = self.modeler.predict_next_state(self.state, actions)
            except ValueError:
                print("Transition inconnue, état aléatoire généré.")
                next_state = np.random.rand(self.num_services, self.num_metrics)

            # Calcul de la récompense (par exemple, minimiser une métrique)
            reward = -np.sum(next_state)

            # Déterminer si l'épisode est terminé
            done = np.random.random() < 0.05  # Probabilité fixe pour terminer l'épisode

            self.state = next_state
            return next_state, reward, done, {}

    # Instanciation de l'environnement
    env = SimpleClusterEnv(modeler)

    # Configuration de MARLlib
    algo_name = "ppo"  # Algorithme choisi parmi ceux supportés par MARLlib
    team = marl.make_env(environment=env)
    model = marl.build_model(team, algo_name)
    algo = marl.build_algo(model=model, algo_name=algo_name)

    # Entraînement des agents
    print("Entraînement des agents MARL en cours...")
    algo.fit(env=env, total_timesteps=10000)

    # Sauvegarde des politiques entraînées
    policy_path = "trained_policy.pth"
    algo.save(policy_path)
    print(f"Politique entraînée sauvegardée à : {policy_path}")