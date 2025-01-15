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
import signal
import random


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, transition):
        """Ajoute une transition au buffer."""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Supprime l'élément le plus ancien
        self.buffer.append(transition)

    def sample(self, batch_size):
        """Échantillonne un lot de transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def size(self):
        """Retourne la taille actuelle du buffer."""
        return len(self.buffer)


class Modeler:
    def __init__(self, db_path, topology_path, model_path="mlp_model.pth", buffer_capacity=10000):
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

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.num_services = len(self.topology["services"])
        self.num_metrics = len(self.topology["metrics"])
        self.input_dim = self.num_services * self.num_metrics + self.num_services
        self.output_dim = self.num_services * self.num_metrics

        self.state_action_scaler = None  # Will store normalization parameters for input
        self.next_state_scaler = None    # Will store normalization parameters for output

        self.model_path = model_path
        # Initialiser ou charger le modèle
        if os.path.exists(self.model_path):
            self.load_mlp()
        else:
            self.initialize_mlp(hidden_dim=128)

    def initialize_mlp(self, hidden_dim=128):
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        print("Modèle MLP initialisé.")

    def normalize(self, data, scaler=None):
        """
        Normalizes data using the provided scaler or calculates a new scaler.

        Args:
            data (np.ndarray): Data to normalize.
            scaler (dict, optional): Scaler containing 'mean' and 'std'.

        Returns:
            np.ndarray: Normalized data.
            dict: Scaler used for normalization.
        """
        if scaler is None:
            mean = data.mean(axis=0)
            std = data.std(axis=0) + 1e-8  # Avoid division by zero
            scaler = {'mean': mean, 'std': std}
        normalized_data = (data - scaler['mean']) / scaler['std']
        return normalized_data, scaler

    def denormalize(self, normalized_data, scaler):
        """
        Denormalizes data using the provided scaler.

        Args:
            normalized_data (np.ndarray): Normalized data.
            scaler (dict): Scaler containing 'mean' and 'std'.

        Returns:
            np.ndarray: Denormalized data.
        """
        return normalized_data * scaler['std'] + scaler['mean']

    def load_mlp(self):
        """Charge le modèle MLP depuis un fichier .pth."""
        self.initialize_mlp()  # Initialiser pour définir la structure
        self.mlp.load_state_dict(torch.load(self.model_path))
        print(f"Modèle MLP chargé depuis {self.model_path}.")

    def save_mlp(self):
        """Sauvegarde le modèle MLP dans un fichier .pth."""
        torch.save(self.mlp.state_dict(), self.model_path)
        print(f"Modèle MLP sauvegardé à {self.model_path}.")

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
        Prédit l'état suivant en utilisant le MLP, en appliquant la normalisation et la dénormalisation.

        Args:
            state (np.ndarray): État courant (2D : [num_services, num_metrics]).
            action (np.ndarray): Action appliquée (1D : [num_services]).

        Returns:
            np.ndarray: État suivant prédit (2D : [num_services, num_metrics]).
        """
        if self.mlp is None:
            raise ValueError("Le MLP n'a pas été entraîné.")

        self.mlp.eval()
        with torch.no_grad():
            # Combine state (flattened) and action
            input_data = np.concatenate(
                (state.flatten(), action), dtype=np.float32)

            # Normalize the input data
            normalized_input, _ = self.normalize(
                input_data, scaler=self.state_action_scaler)

            # Convert to tensor and predict
            input_tensor = torch.tensor(normalized_input).unsqueeze(
                0)  # Add batch dimension
            normalized_next_state = self.mlp(input_tensor).squeeze(0).numpy()

            # Denormalize the predicted next state
            next_state_flat = self.denormalize(
                normalized_next_state, self.next_state_scaler)

            # Reshape the flat next state back to 2D
            next_state = next_state_flat.reshape(
                self.num_services, self.num_metrics)
            return next_state

    def incremental_train_with_buffer(self, new_transitions, epochs=10, batch_size=32, learning_rate=1e-4):
        """
        Fine-tune le MLP en utilisant un replay buffer avec des nouvelles transitions.

        Args:
            new_transitions (list): Liste des nouvelles transitions.
            epochs (int): Nombre d'époques d'entraînement.
            batch_size (int): Taille des lots.
            learning_rate (float): Taux d'apprentissage.
        """
        # Ajouter les nouvelles transitions au buffer
        for transition in new_transitions:
            self.replay_buffer.add(transition)

        # Préparer les données pour l'entraînement
        sampled_transitions = self.replay_buffer.sample(batch_size * 10)
        states = np.array([np.concatenate((s.flatten(), a))
                          for s, a, _ in sampled_transitions], dtype=np.float32)
        next_states = np.array(
            [next_s.flatten() for _, _, next_s in sampled_transitions], dtype=np.float32)

        # Normaliser les données
        states, _ = self.normalize(states, scaler=self.state_action_scaler)
        next_states, _ = self.normalize(
            next_states, scaler=self.next_state_scaler)

        # Préparer le DataLoader
        dataset = TensorDataset(torch.tensor(
            states), torch.tensor(next_states))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Configuration de l'entraînement
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.mlp.parameters(), lr=learning_rate)

        # Boucle d'entraînement
        self.mlp.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.mlp(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss/len(loader):.4f}")

    def train_mlp(self, model, transitions, hidden_dim=128, epochs=50, batch_size=32, learning_rate=1e-3):
        """
        Entraîne un modèle MLP pour prédire l'état suivant à partir d'un état et d'une action.

        Args:
            model (nn.Module): Modèle MLP à entraîner.
            transitions (list): Liste de transitions (state, action, next_state).
            hidden_dim (int): Dimension des couches cachées.
            epochs (int): Nombre d'époques d'entraînement.
            batch_size (int): Taille des lots.
            learning_rate (float): Taux d'apprentissage.

        Returns:
            nn.Module: Modèle MLP entraîné.
        """
        # Préparer les données
        states = np.array([np.concatenate((s.flatten(), a))
                          for s, a, _ in transitions], dtype=np.float32)
        next_states = np.array([next_s.flatten()
                               for _, _, next_s in transitions], dtype=np.float32)

        if states.shape[1] != self.input_dim or next_states.shape[1] != self.output_dim:
            raise ValueError(f"Inconsistent data dimensions! Expected input: {self.input_dim}, output: {self.output_dim}, "
                             f"but got input: {states.shape[1]}, output: {next_states.shape[1]}")

        # Normaliser les données
        states, self.state_action_scaler = self.normalize(states)
        next_states, self.next_state_scaler = self.normalize(next_states)

        # Séparer les données en ensembles d'entraînement et de validation
        X_train, X_val, y_train, y_val = train_test_split(
            states, next_states, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(
            torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Configuration de l'entraînement
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Boucle d'entraînement
        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        print("Entraînement terminé.")
        return model

    def retrain_mlp_loop(self, interval=300, sample_size=100):
        """Boucle de ré-entraînement périodique avec replay buffer."""
        signal.signal(signal.SIGTERM, self.terminate_process)

        is_first_training = True  # Drapeau pour le premier entraînement

        while True:
            try:
                print(
                    "Récupération des nouvelles transitions depuis la base de données...")
                # Pas de limite pour le premier entraînement
                if is_first_training:
                    print("Récupération de l'ensemble de la base de données pour le premier entrainement")
                    new_transitions = self.fetch_transitions(limit=None)
                    is_first_training = False
                else:
                    # Par exemple, les sample_size dernières transitions
                    print(f"Récupération d'un ensemble des {sample_size} transitions")
                    new_transitions = self.fetch_transitions(limit=sample_size)

                print("Fine-tuning du MLP avec les nouvelles données...")
                self.incremental_train_with_buffer(new_transitions)
                self.save_mlp()

                print("Ré-entraînement terminé avec succès.")
                time.sleep(interval)

            except Exception as e:
                print(f"Erreur dans le ré-entraînement : {e}")
                time.sleep(interval)

    def start_retrain_loop(self, interval=3):
        """Démarre le processus de ré-entraînement périodique."""
        process = Process(target=self.retrain_mlp_loop, args=(interval,))
        process.start()
        print(
            f"Processus de ré-entraînement MLP démarré avec PID {process.pid}.")
        return process

    def terminate_process(self, signum, frame):
        """Gère l'arrêt du processus et sauvegarde le modèle."""
        print("Signal d'arrêt reçu. Sauvegarde du modèle en cours...")
        self.save_mlp()
        print("Processus terminé proprement.")
        os._exit(0)


def create_and_populate_database(db_name="transitions.db", num_transitions=10000, num_services=4, num_metrics=17):
    """
    Crée et remplit une base de données SQLite avec des transitions générées aléatoirement.

    Args:
        db_name (str): Nom du fichier de la base de données.
        num_transitions (int): Nombre de transitions à générer.
        num_services (int): Nombre de services dans chaque état.
        num_metrics (int): Nombre de métriques par service dans chaque état.
    """
    db_path = os.path.join("./", db_name)

    # Supprimer l'ancienne base de données si elle existe
    if os.path.exists(db_path):
        os.remove(db_path)

    # Connecter à la base de données
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Créer la table des transitions avec un index sur `state` et `action`
        cursor.execute("""
            CREATE TABLE transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state TEXT NOT NULL,
                action TEXT NOT NULL,
                next_state TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX idx_state_action ON transitions (state, action)")

        # Générer et insérer des transitions aléatoires
        for _ in range(num_transitions):
            state = np.random.rand(num_services, num_metrics).tolist()
            # Actions entre -5 et +5
            action = np.random.randint(-5, 6, size=(num_services,)).tolist()
            next_state = (np.array(state) +
                          np.random.rand(num_services, num_metrics)).tolist()

            # Convertir en JSON pour insérer dans la base de données
            state_json = json.dumps(state)
            action_json = json.dumps(action)
            next_state_json = json.dumps(next_state)

            # Insérer la transition
            cursor.execute("INSERT INTO transitions (state, action, next_state) VALUES (?, ?, ?)",
                           (state_json, action_json, next_state_json))

        conn.commit()

    print(f"Base de données créée et sauvegardée à : {db_path}")


if __name__ == '__main__':

    print("=" * 20)
    print("modeler_test.py")
    print("=" * 20)

    try:

        db_name = "transitions.db"

        modeler = Modeler(
            db_path=db_name, topology_path="../../utils/install_topology.json")

        if not os.path.isfile(db_name):
            create_and_populate_database(
                db_name=db_name, num_transitions=10000, num_services=modeler.num_services, num_metrics=modeler.num_metrics)

        time.sleep(5)

        process = modeler.start_retrain_loop(interval=300)

        print("Appuyez sur Ctrl+C pour arrêter le processus.")
        process.join()
    except KeyboardInterrupt:
        print("Arrêt manuel du processus...")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
