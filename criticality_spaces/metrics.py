"""
Project: Master Thesis - Metrics Framework
Version: 1.0.0

Description:
    Base framework for evaluating Search-Based Scenario Selection Methods (SBSSMs).
    This skeleton provides the necessary infrastructure (logging, grid initialization,
    and a dynamic dispatcher) to easily plug in custom metrics for the thesis.
"""

import numpy as np
import logging
from typing import TYPE_CHECKING, List, Union, Dict

if TYPE_CHECKING:
    from space import Space

logger = logging.getLogger(__name__)


class Metrics:
    """
    Evaluation metric suite for a criticality Space.
    Custom metrics for the master thesis will be implemented here.
    """

    def __init__(self, space: "Space", boundary_radius=0.2):
        """
        Parameters
        ----------
        space : Space
            The criticality space instance to evaluate.
        boundary_radius : float
            Default Euclidean distance threshold used by boundary metrics.
        """
        self.space = space
        self.boundary_radius = boundary_radius

    def run_metrics_suite(self, metric_names: Union[str, List[str]], surrogate_model=None) -> Dict[str, float]:
        """
        Run a list of specified metrics and return the results dynamically.

        Args:
            metric_names (str or List[str]): List of metric function names to execute.
            surrogate_model: Optional surrogate model if a metric requires it.

        Returns:
            dict: Dictionary of computed metric values.
        """
        logger.info("Running Custom Metrics Suite...")

        if isinstance(metric_names, str):
            metric_names = [metric_names]

        results = {}

        for metric_name in metric_names:
            # Holt dynamisch die Funktion aus dieser Klasse anhand des Strings
            metric_func = getattr(self, metric_name, None)

            if callable(metric_func):
                try:
                    logger.info(f"Calculating '{metric_name}'...")
                    
                    # Wenn die Funktion das surrogate_model als Parameter verlangt, 
                    # kann man das hier flexibel übergeben (mit kwargs erweiterbar)
                    import inspect
                    sig = inspect.signature(metric_func)
                    if 'surrogate_model' in sig.parameters:
                        results[metric_name] = metric_func(surrogate_model=surrogate_model)
                    else:
                        results[metric_name] = metric_func()
                        
                except Exception as e:
                    logger.error(f"Error calculating '{metric_name}': {e}")
            else:
                logger.warning(f"Metric '{metric_name}' is not implemented in the Metrics class.")

        logger.info("Metrics Suite Completed.")
        return results

    def _ensure_meshgrid_exists(self, default_n_points: int = 101):
        """
        Ensure that space.meshgrid is initialized. If not, initialize with default resolution.
        Useful for any grid-based metrics you might implement later.
        """
        if getattr(self.space, "n_points", None) is None or not hasattr(self.space, "meshgrid"):
            logger.warning(
                f"Meshgrid not defined. Initializing with {default_n_points} points per dimension."
            )
            self.space.n_points = default_n_points
            self.space.meshgrid = self.space.generate_meshgrid(default_n_points)
            self.space.values = self.space.calculate_space()

            grid_dict = self.space.create_point_value_dict()
            pvd = self.space.point_value_dict

            # Add only entries not already present in the cache
            for k, v in grid_dict.items():
                pvd.setdefault(k, v)
        return
    
    # =====================================================================
    # BASIS-METRIKEN (MAX, MIN, AVERAGE)
    # =====================================================================

    def max_criticality(self) -> float:
        """Maximaler wahrer Fehler auf dem gesamten Raster."""
        self._ensure_meshgrid_exists()
        val = np.max(self.space.values)
        logger.info(f"Max Criticality (Global): {val:.4f}")
        return float(val)

    def min_criticality(self) -> float:
        """Minimaler wahrer Fehler auf dem gesamten Raster."""
        self._ensure_meshgrid_exists()
        val = np.min(self.space.values)
        logger.info(f"Min Criticality (Global): {val:.4f}")
        return float(val)

    def average_criticality(self) -> float:
        """Durchschnittlicher wahrer Fehler auf dem gesamten Raster."""
        self._ensure_meshgrid_exists()
        val = np.mean(self.space.values)
        logger.info(f"Average Criticality (Global): {val:.4f}")
        return float(val)

    def max_criticality_selected(self) -> float:
        """Maximaler gefundener Fehler unter den gezogenen Samples."""
        pts = list(self.space.selected_points)
        if not pts: return 0.0
        vals = self.space.get_values_for_points(pts, save_points=False)
        val = np.max(vals)
        logger.info(f"Max Criticality (Selected): {val:.4f}")
        return float(val)

    def min_criticality_selected(self) -> float:
        """Minimaler gefundener Fehler unter den gezogenen Samples."""
        pts = list(self.space.selected_points)
        if not pts: return 0.0
        vals = self.space.get_values_for_points(pts, save_points=False)
        val = np.min(vals)
        logger.info(f"Min Criticality (Selected): {val:.4f}")
        return float(val)

    def average_criticality_selected(self) -> float:
        """Durchschnittlicher Fehler unter den gezogenen Samples."""
        pts = list(self.space.selected_points)
        if not pts: return 0.0
        vals = self.space.get_values_for_points(pts, save_points=False)
        val = np.mean(vals)
        logger.info(f"Average Criticality (Selected): {val:.4f}")
        return float(val)
    
    # =====================================================================
    # METRIKEN FÜR DIE MASTERARBEIT
    # =====================================================================

    def model_rmse(self, surrogate_model) -> float:
        """
        Modelgüte (RMSE): Misst, wie gut die Approximation (Ersatzmodell) 
        mit der echten Diskrepanz auf dem gesamten Raum übereinstimmt.
        """
        if surrogate_model is None:
            logger.warning("Kein Modell für RMSE übergeben.")
            return None

        self._ensure_meshgrid_exists()
        X = np.vstack([grid.ravel() for grid in self.space.meshgrid]).T
        
        # Ground Truth holen (Echte Diskrepanz)
        y_true = self.space.values.ravel()
        
        # Modell-Vorhersage holen
        y_pred = surrogate_model.predict(X).ravel()
        
        # RMSE berechnen
        mse = np.mean((y_true - y_pred) ** 2)
        rmse_val = np.sqrt(mse)
        
        logger.info(f"Modell RMSE: {rmse_val:.6f}")
        return float(rmse_val)


    def prediction_bias(self, surrogate_model) -> float:
        """
        Bias (Korrektheit): Differenz der Erwartungswerte.
        Zeigt, ob das Modell den Fehler systematisch über- oder unterschätzt.
        > 0 : Modell überschätzt die Gefahr (Pessimistisch)
        < 0 : Modell unterschätzt die Gefahr (Optimistisch)
        """
        if surrogate_model is None:
            logger.warning("Kein Modell für Bias übergeben.")
            return None

        self._ensure_meshgrid_exists()
        X = np.vstack([grid.ravel() for grid in self.space.meshgrid]).T
        
        y_true = self.space.values.ravel()
        y_pred = surrogate_model.predict(X).ravel()
        
        # Bias = E[Vorhersage] - E[Realität]
        bias_val = np.mean(y_pred) - np.mean(y_true)
        
        logger.info(f"Globaler Modell-Bias: {bias_val:.6f}")
        return float(bias_val)


    def sample_variance(self) -> float:
        """
        Varianz der ausgewählten Samples (1 Iteration):
        Misst die Streuung der Kritikalität (Diskrepanz) der aktiv gezogenen Punkte.
        Gibt Aufschluss über Exploration (hohe Varianz) vs. Exploitation (niedrige Varianz).
        """
        selected_points = list(self.space.selected_points)
        
        if len(selected_points) < 2:
            logger.warning("Nicht genug Samples für Varianz-Berechnung.")
            return 0.0
            
        # Die echten gemessenen Werte der Samples abrufen
        values = self.space.get_values_for_points(selected_points, save_points=False)
        
        # Empirische Varianz berechnen (ddof=1 für Stichprobenvarianz)
        variance = np.var(values, ddof=1)
        
        logger.info(f"Sample-Varianz: {variance:.6f} (N={len(values)})")
        return float(variance)


    def sampling_efficiency(self, weights: list = None) -> float:
        """
        Effizienz N_eff / N (Kish's Effective Sample Size).
        Wenn keine expliziten Gewichte übergeben werden, nutzen wir die 
        Diskrepanz-Werte der Samples als 'Wichtigkeit'.
        Nahe 1.0 = Sehr effizientes Sampling (Punkte sind gleichwertig/nützlich)
        Nahe 0.0 = Ineffizient (Algorithmus hängt auf wenigen extremen Punkten fest)
        """
        if weights is None:
            # Fallback: Wir nehmen die gemessenen Fehlerwerte als Wichtigkeit
            pts = list(self.space.selected_points)
            if not pts:
                return 0.0
            # Wir nehmen den Absolutwert, da Gewichte positiv sein müssen
            weights = np.abs(self.space.get_values_for_points(pts, save_points=False))
            
        w = np.array(weights)
        if len(w) == 0 or np.sum(w) == 0:
            return 0.0
            
        # N_eff = (sum(w)^2) / sum(w^2)
        n_eff = (np.sum(w) ** 2) / np.sum(w ** 2)
        
        # Normierung auf [0, 1] (N_eff / N)
        efficiency = n_eff / len(w)
        
        logger.info(f"Sampling Effizienz (N_eff/N): {efficiency:.4f}")
        return float(efficiency)
    
    def convergence_rate(self, threshold_ratio=0.95) -> int:
        """
        Konvergenzrate: Anzahl der Samples (Schritte), bis der Algorithmus 
        einen Punkt findet, der mindestens X% (z.B. 95%) des wahren 
        Maximalfehlers im Raum erreicht.
        """
        if not self.space.selected_points:
            logger.warning("Keine Samples für Konvergenzrate vorhanden.")
            return None
            
        # 1. Wahres globales Maximum im Raum berechnen
        self._ensure_meshgrid_exists()
        max_possible = np.max(self.space.values)
        target_value = max_possible * threshold_ratio
        
        # 2. Durch die Historie der gezogenen Punkte gehen
        # Wir gehen davon aus, dass selected_points chronologisch geordnet ist
        for i, pt in enumerate(self.space.selected_points):
            val = self.space.get_values_for_points([pt], save_points=False)[0]
            if val >= target_value:
                logger.info(f"Konvergenz (>{threshold_ratio*100}%) erreicht nach {i+1} Samples.")
                return i + 1  # +1, da der Index bei 0 beginnt
                
        logger.info(f"Konvergenz-Ziel im Budget nicht erreicht (Max gefordert: {target_value:.4f}).")
        # Wenn er es nicht schafft, geben wir das Gesamtbudget als "schlechtesten Fall" zurück
        return len(self.space.selected_points)
    
