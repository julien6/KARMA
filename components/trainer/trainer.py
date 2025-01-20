import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
# Assurez-vous que le Modeler est disponible et bien importé
from modeler import Modeler


class Trainer:
    def __init__(self, modeler: Modeler, reward_function):
        """
        Initialisation du composant Trainer.

        Args:
            modeler (Modeler): Instance du composant Modeler pour prédire les transitions d'état.
            reward_function (callable): Fonction de récompense globale qui calcule la récompense à partir d'un état et d'une action.
        """
        self.modeler = modeler
        self.reward_function = reward_function
        self.env = None  # Environnement PettingZoo, créé dynamiquement
        self.agents = self._initialize_agents()  # Liste des noms d'agents

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


if __name__ == '__main__':

    # Fonction de récompense globale pour le test
    def test_reward_function(state, action, next_state):
        """
        Calcule une récompense simplifiée pour le test.

        Args:
            state (np.ndarray): État actuel.
            action (np.ndarray): Action prise.
            next_state (np.ndarray): État suivant.

        Returns:
            float: Récompense calculée.
        """
        # Exemple : récompense basée sur la réduction de la différence entre les métriques
        return -np.sum(np.abs(next_state - state))

    print("=" * 20)
    print("trainer_test.py")
    print("=" * 20)

    # 1. Configurer le Modeler
    print("Initialisation du Modeler...")
    db_name = "../modeler/transitions.db"
    topology_path = "../../utils/install_topology.json"
    modeler = Modeler(db_path=db_name, topology_path=topology_path)

    # 2. Créer le Trainer
    print("Initialisation du Trainer...")
    trainer = Trainer(modeler=modeler, reward_function=test_reward_function)

    # 3. Créer l'environnement PettingZoo
    print("Création de l'environnement PettingZoo...")
    trainer.create_pettingzoo_env()
    env = trainer.get_env()

    # 4. Tester les fonctions de base de l'environnement
    print("Test du reset()...")
    observations = env.reset()
    print("Observations initiales :", observations)

    print("Test du step()...")
    actions = {agent: np.random.randint(-5, 6, size=(1,))
               for agent in env.agents}
    print("Actions prises :", actions)

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("Observations après step :", observations)
    print("Récompenses :", rewards)
    print("Terminations :", terminations)
    print("Infos :", infos)
