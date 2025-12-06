import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score


class LSTMModule:
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=1,
        lr=0.001,
        epochs=50,
        batch_size=32,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            list(self.lstm.parameters()) + list(self.fc.parameters()), lr=lr
        )

    def _forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)

        out, _ = self.lstm(X, (h0, c0))
        last = out[:, -1, :]
        logits = self.fc(last)
        return self.sigmoid(logits)

    def fit(self, X_train, y_train):
        X_train = self._to_tensor(X_train)
        y_train = self._to_tensor(y_train).view(-1, 1)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self._forward(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss = {loss.item():.4f}")

        return self

    def predict_proba(self, X):
        X = self._to_tensor(X)
        with torch.no_grad():
            return self._forward(X).numpy().reshape(-1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        return x
    
    def permutation_importance_lstm(self, X_val, y_val, feature_names):
        X_val = np.asarray(X_val)           # shape: (N, T, F)
        y_val = np.asarray(y_val)
        base_proba = self.predict_proba(X_val)
        base_auc = roc_auc_score(y_val, base_proba)
        importances = []

        for j, name in enumerate(feature_names):
            X_perm = X_val.copy()          # (N, T, F)
            # permute feature j across samples, for all timesteps
            perm_idx = np.random.permutation(X_perm.shape[0])
            X_perm[:, :, j] = X_perm[perm_idx, :, j]

            proba_perm = self.predict_proba(X_perm)
            auc_perm = roc_auc_score(y_val, proba_perm)
            importances.append(base_auc - auc_perm)

        importances = np.array(importances)
        if importances.sum() != 0:
            importances = importances / importances.sum()
        return importances

    def get_shap_predict_proba_wrapper(self):
        """
        Returns a callable function that accepts a 2D array from SHAP, 
        reshapes it to 3D, and calls self.predict_proba().
        """
        def shap_predict_proba_wrapper(X_2d_array):
            # X_2d_array shape: (N_samples, N_features)
            
            N_samples, N_features = X_2d_array.shape
            # Reshape to (N_samples, T=1, N_features)
            X_3d_array = X_2d_array.reshape(N_samples, 1, N_features)
            
            # Call the instance's predict_proba method
            return self.predict_proba(X_3d_array)
            
        return shap_predict_proba_wrapper
