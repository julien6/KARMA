import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Process
import os
import time


class Modeler:
    def __init__(self, db_path, topology_path):
        """
        Classe pour modéliser la fonction de transition et entraîner un MLP.

        Args:
            db_path (str): Chemin vers la base de données SQLite contenant les transitions.
            topology_path (str): Chemin vers le fichier JSON contenant la topologie du cluster.
        """
        self.db_path = os.path.abspath(db_path)
        self.topology_path = os.path.abspath(topology_path)

        # Charger la topologie depuis le fichier JSON
        if not os.path.exists(self.topology_path):
            raise FileNotFoundError(
                f"Topology file not found: {self.topology_path}")

        with open(self.topology_path, "r") as f:
            self.topology = json.load(f)

        self.num_services = len(self.topology["services"])
        self.num_metrics = len(self.topology["metrics"])
        self.mlp = None  # Modèle MLP (sera défini et entraîné)

    def fetch_transitions(self, limit=None):
        """
        Récupère les transitions (état, action, état suivant) depuis la base de données.

        Args:
            limit (int, optional): Nombre maximal de transitions à récupérer. Par défaut, récupère tout.

        Returns:
            list: Liste de tuples (state, action, next_state).
        """
        query = "SELECT state, action, next_state FROM transitions"
        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            data = cursor.fetchall()

        transitions = [
            (
                np.array(json.loads(row[0]), dtype=np.float32),  # state
                np.array(json.loads(row[1]), dtype=np.float32),  # action
                np.array(json.loads(row[2]), dtype=np.float32)   # next_state
            )
            for row in data
        ]
        return transitions

    def partial_transition(self, state, action):
        """
        Modélise la fonction de transition partielle exacte en recherchant l'état suivant directement dans la base de données.

        Args:
            state (np.ndarray): État courant.
            action (np.ndarray): Action appliquée.

        Returns:
            np.ndarray: État suivant correspondant.

        Raises:
            ValueError: Si aucune transition correspondante n'est trouvée.
        """
        state_json = json.dumps(state.tolist())
        action_json = json.dumps(action.tolist())

        query = """
            SELECT next_state
            FROM transitions
            WHERE state = ? AND action = ?
            LIMIT 1
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (state_json, action_json))
            result = cursor.fetchone()

        if result:
            next_state_json = result[0]
            return np.array(json.loads(next_state_json), dtype=np.float32)

        raise ValueError("Transition non trouvée dans la base de données.")

    def predict_next_state(self, state, action):
        """
        Prédit l'état suivant synthétiquement : recherche dans la base de données, sinon utilise le MLP.

        Args:
            state (np.ndarray): État courant.
            action (np.ndarray): Action appliquée.

        Returns:
            np.ndarray: État suivant prédit ou exact.
        """
        exact_next_state = self.partial_transition(state, action)
        if exact_next_state is not None:
            return exact_next_state  # Transition trouvée

        # Utiliser le MLP si la transition exacte n'est pas trouvée
        if self.mlp is None:
            raise ValueError("Le MLP n'a pas été entraîné.")

        self.mlp.eval()
        with torch.no_grad():
            input_data = np.concatenate((state, action), dtype=np.float32)
            input_tensor = torch.tensor(input_data).unsqueeze(
                0)  # Ajouter une dimension batch
            next_state = self.mlp(input_tensor).squeeze(
                0).numpy()  # Supprimer la dimension batch
            return next_state

    def train_mlp(self, transitions, hidden_dim=128, epochs=50, batch_size=32, learning_rate=1e-3):
        """
        Entraîne un MLP pour prédire l'état suivant à partir d'un état et d'une action.

        Args:
            transitions (list): Liste de transitions (state, action, next_state).
            hidden_dim (int): Dimension des couches cachées.
            epochs (int): Nombre d'époques d'entraînement.
            batch_size (int): Taille des lots.
            learning_rate (float): Taux d'apprentissage.
        """
        # Préparer les données
        states = np.array([np.concatenate((s, a))
                          for s, a, _ in transitions], dtype=np.float32)
        next_states = np.array(
            [next_s for _, _, next_s in transitions], dtype=np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            states, next_states, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(
            torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Définir le modèle
        input_dim = states.shape[1]
        output_dim = next_states.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Définir la perte et l'optimiseur
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)

        # Entraînement
        self.mlp.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.mlp(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            val_loss = 0.0
            self.mlp.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = self.mlp(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        print("Entraînement terminé.")

    def retrain_mlp_loop(self, interval=300):
        """
        Boucle de réentraînement périodique du MLP.

        Args:
            interval (int): Intervalle de réentraînement en secondes.
        """
        while True:
            try:
                # Cloner les transitions et le MLP
                transitions = self.fetch_transitions()
                cloned_mlp = nn.Sequential(*[layer for layer in self.mlp])

                # Réentraîner le modèle cloné
                self.train_mlp(transitions, hidden_dim=128,
                               epochs=10, batch_size=32, learning_rate=1e-3)

                # Remplacer le modèle actuel par le modèle réentraîné
                self.mlp = cloned_mlp

                print("MLP réentraîné et mis à jour.")
            except Exception as e:
                print(f"Erreur lors du réentraînement du MLP : {e}")

            time.sleep(interval)

    def start_retrain_loop(self, interval=300):
        """
        Lance le processus de réentraînement périodique du MLP.

        Args:
            interval (int): Intervalle de réentraînement en secondes.
        """
        process = Process(target=self.retrain_mlp_loop, args=(interval,))
        process.start()
        print(
            f"Processus de réentraînement MLP démarré avec PID {process.pid}.")
        return process


if __name__ == '__main__':
    print("=" * 20)
    print("modeler_test.py")
    print("=" * 20)

    try:
        modeler = Modeler(db_path="transitions.db",
                          topology_path="../../utils/install_topology.json")
        transitions = modeler.fetch_transitions(limit=1000)
        modeler.train_mlp(transitions, hidden_dim=128,
                          epochs=10, batch_size=32, learning_rate=1e-3)

        # Démarrer la boucle de réentraînement
        modeler.start_retrain_loop(interval=300)

    except Exception as e:
        print(f"An error occurred: {e}")
