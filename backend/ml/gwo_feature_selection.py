"""
gwo_feature_selection.py

Grey Wolf Optimizer (GWO) for feature selection on smart-grid datasets.
Models the social hierarchy (Alpha, Beta, Delta wolves) and hunting mechanism
to discover an optimal, parsimonious feature subset maximizing detection Macro-F1.
"""

import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class GWOFeatureSelector:
    def __init__(
        self,
        num_wolves=20,
        max_iterations=15,
        alpha_weight=0.90,
        random_state=42,
    ):
        self.num_wolves = num_wolves
        self.max_iterations = max_iterations
        self.alpha_weight = alpha_weight
        self.random_state = random_state

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    def _calculate_fitness(self, binary_mask, X_tr, y_tr, X_val, y_val, total_features):
        selected_idx = np.where(binary_mask == 1)[0]
        if len(selected_idx) == 0:
            return 1.0, 0.0

        clf = DecisionTreeClassifier(max_depth=8, random_state=self.random_state)
        clf.fit(X_tr[:, selected_idx], y_tr)
        preds = clf.predict(X_val[:, selected_idx])
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

        parsimony = len(selected_idx) / total_features
        fitness = self.alpha_weight * (1.0 - macro_f1) + (1.0 - self.alpha_weight) * parsimony
        return fitness, macro_f1

    def fit(self, X, y, feature_names, max_samples=8000):
        start_time = time.time()
        np.random.seed(self.random_state)

        # Stratified subsampling for responsive optimization
        if len(X) > max_samples:
            indices = np.arange(len(X))
            sub_idx, _ = train_test_split(
                indices,
                train_size=max_samples,
                stratify=y,
                random_state=self.random_state,
            )
            X_opt = X[sub_idx]
            y_opt = y[sub_idx]
        else:
            X_opt = X
            y_opt = y

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_opt, y_opt, test_size=0.30, stratify=y_opt, random_state=self.random_state
        )

        num_features = len(feature_names)

        # Continuous positions [-2, 2]
        positions = np.random.uniform(-2.0, 2.0, size=(self.num_wolves, num_features))

        alpha_pos = np.zeros(num_features)
        alpha_score = np.inf
        alpha_f1 = 0.0

        beta_pos = np.zeros(num_features)
        beta_score = np.inf
        beta_f1 = 0.0

        delta_pos = np.zeros(num_features)
        delta_score = np.inf
        delta_f1 = 0.0

        # Evaluate initial wolves
        for i in range(self.num_wolves):
            binary_mask = (self._sigmoid(positions[i]) >= 0.5).astype(int)
            if np.sum(binary_mask) == 0:
                binary_mask[np.random.randint(0, num_features)] = 1

            fitness, f1_val = self._calculate_fitness(
                binary_mask, X_tr, y_tr, X_val, y_val, num_features
            )

            if fitness < alpha_score:
                alpha_score = fitness
                alpha_f1 = f1_val
                alpha_pos = positions[i].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_f1 = f1_val
                beta_pos = positions[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_f1 = f1_val
                delta_pos = positions[i].copy()

        fitness_history = [float(alpha_score)]

        # GWO main hunting loop
        for t in range(self.max_iterations):
            a = 2.0 - t * (2.0 / self.max_iterations)  # a decreases linearly from 2 to 0

            for i in range(self.num_wolves):
                for j in range(num_features):
                    # Update against Alpha wolf
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2.0 * a * r1 - a
                    C1 = 2.0 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    # Update against Beta wolf
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2.0 * a * r1 - a
                    C2 = 2.0 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    # Update against Delta wolf
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2.0 * a * r1 - a
                    C3 = 2.0 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    positions[i, j] = (X1 + X2 + X3) / 3.0

                binary_mask = (self._sigmoid(positions[i]) >= 0.5).astype(int)
                if np.sum(binary_mask) == 0:
                    binary_mask[np.random.randint(0, num_features)] = 1

                fitness, f1_val = self._calculate_fitness(
                    binary_mask, X_tr, y_tr, X_val, y_val, num_features
                )

                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha_f1 = f1_val
                    alpha_pos = positions[i].copy()
                elif fitness < beta_score:
                    beta_score = fitness
                    beta_f1 = f1_val
                    beta_pos = positions[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_f1 = f1_val
                    delta_pos = positions[i].copy()

            fitness_history.append(float(alpha_score))

        final_mask = (self._sigmoid(alpha_pos) >= 0.5).astype(int)
        if np.sum(final_mask) == 0:
            final_mask[np.random.randint(0, num_features)] = 1

        selected_indices = np.where(final_mask == 1)[0].tolist()
        selected_features = [feature_names[idx] for idx in selected_indices]
        opt_time = round(time.time() - start_time, 2)

        return {
            "algorithm": "GWO",
            "original_feature_count": num_features,
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "selected_indices": selected_indices,
            "best_fitness": round(float(alpha_score), 4),
            "validation_macro_f1": round(float(alpha_f1), 4),
            "optimization_time_seconds": opt_time,
            "fitness_history": fitness_history,
        }
