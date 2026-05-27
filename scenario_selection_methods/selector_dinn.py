import numpy as np
from sklearn.neural_network import MLPRegressor
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from sklearn.model_selection import LeaveOneOut       # <-- Wichtig für Jackknife
from mapie.regression import CrossConformalRegressor  # <-- MAPIE >= 1.0
from base import BaseSelector 

class NNBiasSelector(BaseSelector):
    """
    Active Learning Selector basierend auf einem Neuronalen Netz (MLP).
    Nutzt Jackknife+ Conformal Prediction via CrossConformalRegressor (MAPIE >= 1.0).
    """

    def __init__(self, space, n_select: int, n_initial: int = 10, batch_size: int = 1):
        super().__init__(space, n_select)
        self.n_initial = n_initial
        self.batch_size = batch_size
        
        # self.nn_model = MLPRegressor(
        #     hidden_layer_sizes=(32, 16),
        #     activation='relu',
        #     solver='lbfgs', 
        #     alpha=0.0001,
        #     max_iter=2000,
        #     random_state=42 
        # )

        self.nn_model = MLPRegressor(
            hidden_layer_sizes=(32, 32), # Massiv mehr "Gelenke" für Kanten
            activation='relu',                 
            solver='lbfgs', 
            alpha=0.0,                         # 0 Regularisierung = Erlaubt extrem steile Wände
            tol=1e-6,                          # Zwingt das Netz, die Punkte exakt zu treffen
            max_iter=5000,                     # Mehr Zeit für die komplexe Geometrie
            random_state=42 
        )

        self.sample_history = []
        

    def _get_diff_data(self, X):
        """Berechnet f(x) = Ground Truth (Realität) - Illusion (Simulation)."""
        y_true = np.array(self.space.get_values_for_points(X, save_points=False))
        y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true, X)])
        diff = y_true - y_illus
        return diff

    def select(self) -> list[tuple]:
        bounds = self.space.dimensions
        dim = len(bounds)
        
        sampler = qmc.LatinHypercube(d=dim, seed=22)
        X_train = sampler.random(n=self.n_initial)
        X_train = qmc.scale(X_train, [b[0] for b in bounds], [b[1] for b in bounds])
        
        current_diffs = self._get_diff_data(X_train)
        self.space.get_values_for_points(X_train, save_points=True)
        
        current_X = X_train
        counter = 0
        
        def normalize(arr):
            ptp = np.ptp(arr)
            return (arr - np.min(arr)) / ptp if ptp > 0 else np.zeros_like(arr)

        print("\n[NNBiasSelector] Starte Training & Sampling mit MAPIE (CrossConformalRegressor)...")
        
        while len(current_X) < self.n_select:
            # --- Basis-Modell trainieren ---
            self.nn_model.fit(current_X, current_diffs)
            self.sample_history.append(current_X.copy())
            
            X_cand = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (1000, dim))
            
            # =================================================================
            # MAPIE >= 1.0: JACKKNIFE+ CONFORMAL PREDICTION
            # =================================================================
            mapie = CrossConformalRegressor(
                estimator=self.nn_model, 
                cv=LeaveOneOut(), 
                method='plus',     
                random_state=42
            )
            
            mapie.fit_conformalize(current_X, current_diffs)
            
            try:
                pred_mean, pis = mapie.predict_interval(X_cand)
            except TypeError:
                pred_mean, pis = mapie.predict_interval(X_cand, alpha=0.1)
            
            lower_bound = pis[:, 0, 0]
            upper_bound = pis[:, 1, 0]
            
            # =================================================================
            # ACQUISITION FUNCTION
            # =================================================================
            
            # Kanidaten
            y_true_cand = np.array(self.space.get_values_for_points(X_cand, save_points=False))
            y_illus_cand = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true_cand, X_cand)])
            
            # 1. Prädizierte reale Kritikalität (C)
            y_true_cand = np.array(self.space.get_values_for_points(X_cand, save_points=False))
            y_illus_cand = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true_cand, X_cand)])
            pred_real_crit = y_illus_cand + pred_mean
            
            # 2. Unsicherheit (U) via MAPIE
            U = upper_bound - lower_bound
            
            # 3. Delta / Diskrepanz (E)
            delta = np.abs(pred_mean)
            
            # 4. Globale Diversität (D) 
            from scipy.spatial.distance import cdist # (Kann auch ganz oben in die Datei)
            dists_to_known = cdist(X_cand, current_X)
            # Für jeden Kandidaten nehmen wir die Distanz zu seinem nächsten Nachbarn.
            # Je größer diese Minimaldistanz, desto "einsamer" (besser für Exploration) ist der Punkt.
            min_dists = np.min(dists_to_known, axis=1)
            
            # Normierung auf [0, 1]
            norm_crit = normalize(pred_real_crit)
            norm_U = normalize(U)
            norm_delta = normalize(delta)
            norm_D = normalize(min_dists) # 1.0 = am weitesten weg von allen bekannten Punkten
            
            w_C = 0.25
            w_U = 0.25 
            w_E = 0.25  
            w_D = 0.25  
            
            scores = (w_C * norm_crit) + (w_U * norm_U) + (w_E * norm_delta) + (w_D * norm_D)
            
            # --- AUSWAHL DES NÄCHSTEN PUNKTES (Einfaches Sortieren reicht jetzt) ---
            points_needed = self.n_select - len(current_X)
            actual_n_per_iter = min(self.batch_size, points_needed)
            
            best_indices = np.argsort(scores)[-actual_n_per_iter:]
            next_X = X_cand[best_indices]
            
            new_diffs = self._get_diff_data(next_X)
            self.space.get_values_for_points(next_X, save_points=True)
            
            current_X = np.vstack([current_X, next_X])
            current_diffs = np.concatenate([current_diffs, new_diffs])
            
            counter += 1
            print(f" Iteration {counter} beendet. U-Score Max: {np.max(U):.4f}")
            
        self.nn_model.fit(current_X, current_diffs)
        self.sample_history.append(current_X.copy())
            
        return [tuple(p) for p in current_X]

    def get_model(self):
        return self.nn_model