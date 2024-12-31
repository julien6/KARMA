import os
import pickle
from collections import deque
import numpy as np
from sklearn.neural_network import MLPRegressor

class Modeler:
    """
    Handles the collection of transitions, building and updating transition models.
    """
    def __init__(self, buffer_size=10000, model_save_path="model.pkl"):
        """
        Initialize the Modeler with a transition buffer and a model.

        Args:
            buffer_size (int): Maximum size of the transition buffer.
            model_save_path (str): Path to save the trained model.
        """
        self.transition_buffer = deque(maxlen=buffer_size)
        self.model_save_path = model_save_path
        self.model = None  # Placeholder for the MLP model

    def add_transition(self, state, action, next_state):
        """
        Add a transition to the buffer.

        Args:
            state (array-like): The current state.
            action (array-like): The action taken.
            next_state (array-like): The resulting next state.
        """
        self.transition_buffer.append((state, action, next_state))

    def save_buffer(self, path="buffer.pkl"):
        """
        Save the current transition buffer to disk.

        Args:
            path (str): Path to save the buffer.
        """
        with open(path, "wb") as f:
            pickle.dump(list(self.transition_buffer), f)
        print(f"Transition buffer saved to {path}")

    def load_buffer(self, path="buffer.pkl"):
        """
        Load a transition buffer from disk.

        Args:
            path (str): Path to load the buffer from.
        """
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.transition_buffer = deque(pickle.load(f), maxlen=len(self.transition_buffer))
            print(f"Transition buffer loaded from {path}")
        else:
            print(f"Buffer file {path} does not exist.")

    def train_model(self):
        """
        Train an MLP model on the transition data.
        """
        if len(self.transition_buffer) < 10:
            print("Not enough data in buffer to train the model.")
            return

        # Prepare training data
        states, actions, next_states = zip(*self.transition_buffer)
        X = np.hstack((np.array(states), np.array(actions)))
        y = np.array(next_states)

        # Train an MLPRegressor
        self.model = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, random_state=42)
        self.model.fit(X, y)
        print("Model trained successfully.")

        # Save the trained model
        self.save_model()

    def save_model(self):
        """
        Save the trained model to disk.
        """
        if self.model:
            with open(self.model_save_path, "wb") as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {self.model_save_path}")

    def load_model(self):
        """
        Load the trained model from disk.
        """
        if os.path.exists(self.model_save_path):
            with open(self.model_save_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_save_path}")
        else:
            print(f"Model file {self.model_save_path} does not exist.")

    def predict_next_state(self, state, action):
        """
        Predict the next state given a state and action using the trained model.

        Args:
            state (array-like): The current state.
            action (array-like): The action taken.

        Returns:
            array-like: The predicted next state.
        """
        if not self.model:
            print("Model is not loaded or trained.")
            return None

        X = np.hstack((state, action)).reshape(1, -1)
        return self.model.predict(X)[0]

# Example usage
if __name__ == "__main__":
    modeler = Modeler()

    # Example transitions
    for i in range(50):
        state = [i, i + 1]
        action = [i % 3]
        next_state = [i + 2, i + 3]
        modeler.add_transition(state, action, next_state)

    modeler.train_model()
    predicted = modeler.predict_next_state([10, 11], [1])
    print(f"Predicted next state: {predicted}")
