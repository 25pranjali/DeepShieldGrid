"""
pso_feature_selection.py

Particle Swarm Optimization (PSO) for feature selection on smart-grid datasets.
Evaluates candidate feature subsets using validation fitness balancing
Macro-F1 score and feature parsimony (minimizing redundant features).
"""

import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class PSOFeatureSelector:
    def __init__(
        self,
        num_particles=20,
        num_iterations=15,
        w=0.7,
        c1=1.5,
        c2=1.5,
        alpha=0.90,
        random_state=42,
    ):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.random_state = random_state

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    def _calculate_fitness(self, mask, X_tr, y_tr, X_val, y_val, total_features):
        selected_idx = np.where(mask == 1)[0]
        if len(selected_idx) == 0:
            return 1.0, 0.0  # worst fitness

        clf = DecisionTreeClassifier(max_depth=8, random_state=self.random_state)
        clf.fit(X_tr[:, selected_idx], y_tr)
        preds = clf.predict(X_val[:, selected_idx])
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

        parsimony = len(selected_idx) / total_features
        fitness = self.alpha * (1.0 - macro_f1) + (1.0 - self.alpha) * parsimony
        return fitness, macro_f1

    def fit(self, X, y, feature_names, max_samples=8000):
        start_time = time.time()
        np.random.seed(self.random_state)

        # Stratified subsampling for fast, responsive optimization if dataset is large
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
        positions = np.random.randint(0, 2, size=(self.num_particles, num_features))
        # Ensure at least 1 feature is selected per particle
        for i in range(self.num_particles):
            if np.sum(positions[i]) == 0:
                positions[i, np.random.randint(0, num_features)] = 1

        velocities = np.random.uniform(-1.0, 1.0, size=(self.num_particles, num_features))
        pbest_positions = positions.copy()
        pbest_fitness = np.full(self.num_particles, np.inf)
        pbest_f1 = np.zeros(self.num_particles)

        gbest_position = positions[0].copy()
        gbest_fitness = np.inf
        gbest_f1 = 0.0

        # Evaluate initial particles
        for i in range(self.num_particles):
            fit_val, f1_val = self._calculate_fitness(
                positions[i], X_tr, y_tr, X_val, y_val, num_features
            )
            pbest_fitness[i] = fit_val
            pbest_f1[i] = f1_val
            if fit_val < gbest_fitness:
                gbest_fitness = fit_val
                gbest_f1 = f1_val
                gbest_position = positions[i].copy()

        fitness_history = [float(gbest_fitness)]

        # Swarm iterations
        for iteration in range(self.num_iterations):
            for i in range(self.num_particles):
                r1 = np.random.rand(num_features)
                r2 = np.random.rand(num_features)

                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest_position - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -4.0, 4.0)

                probs = self._sigmoid(velocities[i])
                positions[i] = (np.random.rand(num_features) < probs).astype(int)

                if np.sum(positions[i]) == 0:
                    positions[i, np.random.randint(0, num_features)] = 1

                fit_val, f1_val = self._calculate_fitness(
                    positions[i], X_tr, y_tr, X_val, y_val, num_features
                )

                if fit_val < pbest_fitness[i]:
                    pbest_fitness[i] = fit_val
                    pbest_f1[i] = f1_val
                    pbest_positions[i] = positions[i].copy()

                if fit_val < gbest_fitness:
                    gbest_fitness = fit_val
                    gbest_f1 = f1_val
                    gbest_position = positions[i].copy()

            fitness_history.append(float(gbest_fitness))

        selected_indices = np.where(gbest_position == 1)[0].tolist()
        selected_features = [feature_names[idx] for idx in selected_indices]
        opt_time = round(time.time() - start_time, 2)

        return {
            "algorithm": "PSO",
            "original_feature_count": num_features,
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "selected_indices": selected_indices,
            "best_fitness": round(float(gbest_fitness), 4),
            "validation_macro_f1": round(float(gbest_f1), 4),
            "optimization_time_seconds": opt_time,
            "fitness_history": fitness_history,
        }
