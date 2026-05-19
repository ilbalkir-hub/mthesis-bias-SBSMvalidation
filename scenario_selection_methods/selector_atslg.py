import numpy as np
from base import BaseSelector
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc

import warnings
warnings.filterwarnings("ignore")

class ATSLGSelector(BaseSelector):
    """
    ATSLG Baseline nach Feng et al. (2020).
    Nutzt Gleichungen 28-32 für die Acquisition Function, um die 
    Performance-Dissimilarity (Bias) gezielt zu kompensieren.
    """

    def __init__(self, space, n_select: int, n_initial: int = 10, n_iterations: int = 10):
        
        super().__init__(space, n_select)
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_per_iter = max(1, (n_select - n_initial) // n_iterations)
        
        dim = len(space.dimensions)
        self.kernel = C(1.0) * RBF(length_scale=[1.0]*dim, length_scale_bounds=(1e-2, 1e3))
        
        self.gpc = GaussianProcessClassifier(kernel=self.kernel)
        self.gpr1 = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5) # Für Suboptimale Regionen
        self.gpr2 = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5) # Für Optimale Regionen

    def _get_diff_data(self, X):
        """Berechnet f(x) = Ground Truth - Illusion (Gleichung 12)."""
        y_true = np.array(self.space.get_values_for_points(X, save_points=False))
        y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true, X)])
        
        diff = y_true - y_illus
        
        # y(x) = +1 (Suboptimal) wenn f(x) != 0, sonst -1 (Optimal) nach Gl. 20
        # Toleranz von 1e-3 für Fließkomma-Vergleiche
        labels = (np.abs(diff) > 1e-3).astype(int) 
        return diff, labels, y_true

    def select(self) -> list[tuple]:
        bounds = self.space.dimensions
        dim = len(bounds)
        
        # 1. Initialisierung (Gleichung 17 / LHS)
        sampler = qmc.LatinHypercube(d=dim, seed=22)  #Powerpoint seed=42
        X_train = sampler.random(n=self.n_initial)
        X_train = qmc.scale(X_train, [b[0] for b in bounds], [b[1] for b in bounds])
        
        diffs, labels, _ = self._get_diff_data(X_train)
        self.space.get_values_for_points(X_train, save_points=True)
        
        current_X = X_train
        current_diffs = diffs
        current_labels = labels
        counter = 0 

        for i in range(self.n_iterations):
            X_cand = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (500, dim))
            unique_labels = np.unique(current_labels)
            
            # --- MODELLE TRAINIEREN ---
            if len(unique_labels) > 1:
                self.gpc.fit(current_X, current_labels)
                p1 = self.gpc.predict_proba(X_cand)[:, 1] # P_1,X_N(x) nach Gl. 21
            else:
                prob = 1.0 if unique_labels[0] == 1 else 0.0
                p1 = np.full(len(X_cand), prob)
            
            p2 = 1.0 - p1 # P_2,X_N(x) nach Gl. 22
            
            # GPR 1 (Suboptimal)
            if np.any(current_labels == 1):
                self.gpr1.fit(current_X[current_labels == 1], current_diffs[current_labels == 1])
                mean1, std1 = self.gpr1.predict(X_cand, return_std=True)
            else:
                mean1, std1 = np.zeros(len(X_cand)), np.ones(len(X_cand))
                
            # GPR 2 (Optimal)
            if np.any(current_labels == 0):
                self.gpr2.fit(current_X[current_labels == 0], current_diffs[current_labels == 0])
                mean2, std2 = self.gpr2.predict(X_cand, return_std=True)
            else:
                mean2, std2 = np.zeros(len(X_cand)), np.ones(len(X_cand))

            counter += 1
            if counter == 9:
                print("x")
            

            # --- ACQUISITION FUNCTION (Gleichungen 28 - 31) ---
            
            # Gleichung 30
            E1 = (mean1 ** 2) + (std1 ** 2)
            E2 = (mean2 ** 2) + (std2 ** 2)
            
            # Gleichung 29 (Ohne P(x)^2 / q(x) da unser Suchraum zunächst uniform ist)
            EI = (p1 * E1) + (p2 * E2)
            
            # Gleichung 31 (Normalization & Addition)
            # Klassifikations-Varianz: p * (1-p)
            class_var = p1 * (1 - p1)
            
            U_E = np.max(EI) if np.max(EI) > 0 else 1.0
            U_C = np.max(class_var) if np.max(class_var) > 0 else 1.0
            
            w = 0.5 # Gewichtungsfaktor nach Paper
            scores = w * (EI / U_E) + (1 - w) * (class_var / U_C) #her 
            var1 = (EI / U_E)
            var2 = (EI / U_E)
            # Wähle die Top-Kandidaten aus
            best_indices = np.argsort(scores)[-self.n_per_iter:]
            next_X = X_cand[best_indices]
            
            # Neue Punkte testen
            new_diffs, new_labels, _ = self._get_diff_data(next_X)
            self.space.get_values_for_points(next_X, save_points=True)
            
            current_X = np.vstack([current_X, next_X])
            current_diffs = np.concatenate([current_diffs, new_diffs])
            current_labels = np.concatenate([current_labels, new_labels])
            
        return [tuple(p) for p in current_X]
    
    
class StandardGPRSelector(BaseSelector):
    """
    Standard GPR Baseline ohne Klassifizierung.
    Nutzt nur ein einziges GPR-Modell, um die Fehlerlandkarte (Dissimilarity) 
    durchgehend zu schätzen.
    """

    def __init__(self, space, n_select: int, n_initial: int = 10, n_iterations: int = 10):
        super().__init__(space, n_select)
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_per_iter = max(1, (n_select - n_initial) // n_iterations)
        
        dim = len(space.dimensions)
        self.kernel = C(1.0) * RBF(length_scale=[1.0]*dim, length_scale_bounds=(1e-2, 1e3))
        
        # NUR NOCH EIN EINZIGES MODELL
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5)

    def _get_diff_data(self, X):
        """Berechnet f(x) = Ground Truth - Illusion (ohne Labels)."""
        y_true = np.array(self.space.get_values_for_points(X, save_points=False))
        y_illus = np.array([self.space.bias.apply(v, p) for v, p in zip(y_true, X)])
        diff = y_true - y_illus
        
        return diff, y_true

    def select(self) -> list[tuple]:
        bounds = self.space.dimensions
        dim = len(bounds)
        
        # 1. Initialisierung (Latin Hypercube)
        sampler = qmc.LatinHypercube(d=dim, seed=11)
        X_train = sampler.random(n=self.n_initial)
        X_train = qmc.scale(X_train, [b[0] for b in bounds], [b[1] for b in bounds])
        
        current_diffs, _ = self._get_diff_data(X_train)
        self.space.get_values_for_points(X_train, save_points=True)
        
        current_X = X_train

        for i in range(self.n_iterations):
            X_cand = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (500, dim))
            
            # --- MODEL TRAINIEREN ---
            # Das eine Modell lernt jetzt ALLE Punkte auf einmal, egal ob Fehler 0 oder >0
            self.gpr.fit(current_X, current_diffs)
            mean, std = self.gpr.predict(X_cand, return_std=True)

            # --- ACQUISITION FUNCTION ---
            # Da es keine Klassen mehr gibt, fällt P1, P2 und die Klassenvarianz weg.
            # Der Informationswert (Expected Improvement) ist rein der erwartete quadrierte Fehler.
            EI = (mean ** 2) + (std ** 2)
            
            # Keine Kombination mehr, EI ist direkt unser Score
            scores = EI 
            
            # Wähle die Top-Kandidaten aus
            best_indices = np.argsort(scores)[-self.n_per_iter:]
            next_X = X_cand[best_indices]
            
            # Neue Punkte testen
            new_diffs, _ = self._get_diff_data(next_X)
            self.space.get_values_for_points(next_X, save_points=True)
            
            current_X = np.vstack([current_X, next_X])
            current_diffs = np.concatenate([current_diffs, new_diffs])

        return [tuple(p) for p in current_X]