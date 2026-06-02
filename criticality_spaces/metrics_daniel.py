"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Implements the full metric suite for evaluating Search-Based Scenario
    Selection Methods (SBSSMs) on a criticality Space. Metrics are organised
    into four categories (general, extremum_search, model_reconstruction,
    boundary_detection) and selected dynamically via metric_categories.py.

    - key role: Quantitative evaluation of sampling quality, extremum convergence,
                surrogate model fidelity, and boundary detection performance.
    - dependency: space.Space, metric_categories, scipy, sklearn
    - output: Dict of named metric values; optional boundary coverage plot.

Usage:
    # Metrics are instantiated automatically by Space; call via:
    space.metrics.run_metrics_suite(method_categories=["general", "boundary_detection"])
    # Or invoke individual metrics directly:
    cov = space.metrics.boundary_coverage(boundary_radius=0.5)
"""

import numpy as np
from metric_categories import ALL_METRICS_BY_CATEGORY
from scipy.stats import entropy, qmc
from sklearn.metrics import f1_score, mean_squared_error, r2_score
import logging
from typing import TYPE_CHECKING, List, Union, Dict
import itertools
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from space import Space

logger = logging.getLogger(__name__)


class Metrics:
    """
    Evaluation metric suite for a criticality Space.

    Metrics are grouped by evaluation objective and activated via category
    keys. All methods read from ``space.selected_points`` and
    ``space.point_value_dict``; grid-dependent metrics trigger lazy meshgrid
    initialisation via ``_ensure_meshgrid_exists``.
    """

    def __init__(self, space: "Space", boundary_radius=0.2):
        """
        Parameters
        ----------
        space : Space
            The criticality space instance to evaluate.
        boundary_radius : float
            Default Euclidean distance threshold used by boundary metrics
            (boundary_precision, boundary_coverage, boundary_effectiveness).
        """
        self.space = space
        self.boundary_radius = boundary_radius

    def run_metrics_suite(
        self, method_categories: Union[str, List[str]] = "general", surrogate_model=None
    ):
        """
        Run the full suite of metrics based on selected method categories and log the results.

        Args:
            method_categories (str or List[str]): One or more categories from:
                ["general", "extremum_search", "model_reconstruction", "boundary_detection"]

        Returns:
            dict: Dictionary of computed metric values.
        """
        logger.info("Running Metrics Suite...")

        if isinstance(method_categories, str):
            method_categories = [method_categories]

        # Combine all selected metrics from categories
        selected_metrics = set()
        for cat in method_categories:
            selected_metrics |= ALL_METRICS_BY_CATEGORY.get(cat, set())

        metrics = {}

        logger.info("Parameter Space Metrics")
        if "average_criticality" in selected_metrics:
            metrics["average_criticality"] = self.average_criticality()
        if "max_criticality" in selected_metrics:
            metrics["max_criticality"] = self.max_criticality()
        if "min_criticality" in selected_metrics:
            metrics["min_criticality"] = self.min_criticality()

        logger.info("Selection/Model Metrics")
        if "average_criticality_selected" in selected_metrics:
            metrics["average_criticality_selected"] = self.average_criticality(
                selected_only=True
            )
        if "max_criticality_selected" in selected_metrics:
            metrics["max_criticality_selected"] = self.max_criticality(
                selected_only=True
            )
        if "min_criticality_selected" in selected_metrics:
            metrics["min_criticality_selected"] = self.min_criticality(
                selected_only=True
            )

        if "spatial_entropy" in selected_metrics:
            metrics["spatial_entropy"] = self.spatial_entropy(selected_only=True)

        if "discrepancy" in selected_metrics:
            metrics["discrepancy"] = self.discrepancy(selected_only=True)

        if "convergence_rate" in selected_metrics:
            metrics["convergence_rate"] = self.convergence_rate(threshold=0.95)

        if "extremum_gap" in selected_metrics:
            if "max_criticality" not in metrics:
                metrics["max_criticality"] = self.max_criticality()
            known_opt = metrics["max_criticality"]
            metrics["extremum_gap"] = self.extremum_gap(known_opt)

        if "model_approximation_error" in selected_metrics:
            metrics["model_approximation_error"] = self.model_approximation_error(
                surrogate_model=surrogate_model
            )

        if "model_r2_score" in selected_metrics:
            metrics["model_r2_score"] = self.model_r2_score(surrogate_model)

        if "f1_coverage" in selected_metrics:
            f1_by_thresh = self.f1_coverage(surrogate_model, average=True)
            for t, f1 in f1_by_thresh.items():
                key = f"f1_coverage@{t}"
                metrics[key] = float(f1)

        if "boundary_precision" in selected_metrics:
            metrics["boundary_precision"] = self.boundary_precision()
            logger.info(f"Boundary Precision (M_prec): {metrics['boundary_precision']}")

        if "boundary_effectiveness" in selected_metrics:
            metrics["boundary_effectiveness"] = self.boundary_effectiveness()

        if "boundary_coverage" in selected_metrics:
            metrics["boundary_coverage"] = self.boundary_coverage()

        logger.info("Metrics Suite Completed.")
        return metrics

    def _ensure_meshgrid_exists(self, default_n_points: int = 101):
        """
        Ensure that space.meshgrid is initialized. If not, initialize with default resolution.
        """
        if getattr(self.space, "n_points", None) is None or not hasattr(
            self.space, "meshgrid"
        ):
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

    ## General Metrics
    def average_criticality(self, selected_only=False):
        """Calculate and return the average criticality from values."""
        if selected_only:
            if len(self.space.selected_points) == 0:
                logger.info(
                    "Average Criticality of selected scenarios: None (no points selected)"
                )
                return None
            avg = np.mean(self.space.get_values_for_points(self.space.selected_points, save_points=False))
            logger.info(f"Average Criticality of selected scenarios: {avg}")
        else:
            if self.space.n_points == None:
                logger.info(
                    "Average Criticality wasnt calculated because no grid was defined (n_points)"
                )
                return None
            avg = np.mean(self.space.values)
            logger.info(f"Average Criticality: {avg}")
        return avg

    def max_criticality(self, selected_only=False):
        """Calculate and return the maximum criticality from values."""
        if selected_only:
            if len(self.space.selected_points) == 0:
                logger.info(
                    "Maximum Criticality of selected scenarios: 0 (no points selected)"
                )
                return 0
            max_val = np.max(
                self.space.get_values_for_points(self.space.selected_points, save_points=False)
            )
            logger.info(f"Maximum Criticality of selected scenarios: {max_val}")
        else:
            if not hasattr(self.space, "values"):
                return None
            max_val = np.max(self.space.values)
            logger.info(f"Maximum Criticality: {max_val}")
        return max_val

    def min_criticality(self, selected_only=False):
        """Calculate and return the minimum criticality from values."""
        if selected_only:
            if len(self.space.selected_points) == 0:
                logger.info(
                    "Minimum Criticality of selected scenarios: 0 (no points selected)"
                )
                return 0
            min_val = np.min(
                self.space.get_values_for_points(self.space.selected_points, save_points=False)
            )
            logger.info(f"Minimum Criticality of selected scenarios: {min_val}")
        else:
            if not hasattr(self.space, "values"):
                return None
            min_val = np.min(self.space.values)
            logger.info(f"Minimum Criticality: {min_val}")
        return min_val

    def count_scenarios(self):
        """Count the number of scenarios."""
        count = len(self.space.selected_points)
        logger.info(f"Number of selected scenarios: {count}")
        return count
    
    def suggested_bins_per_dim(self, N, d, target_occupancy=50, min_bins=2, max_bins=8):
        """
        Heuristic: choose histogram bins per dimension so that each bin
        contains roughly ``target_occupancy`` points on average.
        Result is clamped to [min_bins, max_bins].
        """
        b = int((N / float(target_occupancy)) ** (1.0 / d))
        return max(min_bins, min(b, max_bins))

    def spatial_entropy(self, selected_only=True, bins=None):
        """
        Shannon entropy of the spatial distribution of selected points,
        estimated via a d-dimensional histogram. Higher entropy indicates
        more uniform coverage of the input space.

        Parameters
        ----------
        selected_only : bool
            If True (default), compute over selected_points only.
            Computing on the full uniform meshgrid is uninformative and skipped.
        bins : list of int, optional
            Number of histogram bins per dimension. If None, determined
            automatically by ``suggested_bins_per_dim``.

        Returns
        -------
        float or None
            Shannon entropy in nats, or None if no points are available.
        """
        if bins == None:
            bins_per_dim = self.suggested_bins_per_dim(self.count_scenarios(), len(self.space.dimensions))
            bins = [bins_per_dim]*len(self.space.dimensions)
        if selected_only and not self.space.selected_points:
            logger.warning("No selected_points")
            return None

        if selected_only:
            points = np.array(list(self.space.selected_points))
        else:
            logger.warning(
                "Spatial entropy calculation on the full meshgrid is skipped because the grid is uniformly distributed and does not provide informative entropy."
            )
        if len(points) == 0:
            return None

        hist, _ = np.histogramdd(points, bins=bins, range=self.space.dimensions)
        probs = hist.flatten() / np.sum(hist)
        probs = probs[probs > 0]
        return entropy(probs)

    def discrepancy(self, selected_only=True):
        """
        Centered discrepancy (CD) of the selected points, normalised to the
        unit hypercube defined by ``space.dimensions``. Lower values indicate
        a more uniform distribution; computing on a uniform meshgrid is
        trivially zero and therefore skipped.

        Returns
        -------
        float or None
            CD discrepancy score, or None if fewer than one point is selected.
        """
        if selected_only:
            points = np.array(list(self.space.selected_points))
        else:
            logger.warning(
                "Discrepancy is not meaningful for uniform meshgrids. Skipping metric."
            )
            return None

        if points.shape[0] == 0:
            return None

        # Extract bounds from space.dimensions
        bounds = np.array(self.space.dimensions)
        mins = bounds[:, 0]
        maxs = bounds[:, 1]
        ranges = maxs - mins

        # Normalize to [0, 1]^d
        points_normalized = (points - mins) / ranges

        if not ((0 <= points_normalized).all() and (points_normalized <= 1).all()):
            raise ValueError("Normalized points fall outside unit hypercube.")

        return qmc.discrepancy(points_normalized, method="CD")

    ## Extremum Search Metrics
    def convergence_rate(self, threshold: float = 0.9) -> Union[int, None]:
        """
        Return the index (1-based) of the first selected point that reaches the criticality threshold.

        Args:
            threshold (float): Criticality threshold defining a successful convergence.

        Returns:
            int or None: The index (1-based) of first point meeting threshold, or None if not reached.
        """
        if not self.space.selected_points:
            return None  # No selected points
        for i, pt in enumerate(self.space.selected_points):
            value = self.space.point_value_dict.get(pt)
            if value is not None and value >= threshold:
                return i + 1  # 1-based index for interpretability

        return None  # Threshold never reached

    def extremum_gap(self, known_optimum):
        """
        Absolute gap between the global maximum criticality of the space and
        the highest criticality value found among selected points.

        Parameters
        ----------
        known_optimum : float
            Ground-truth maximum criticality (typically from ``max_criticality()``).

        Returns
        -------
        float or None
            Gap ≥ 0; 0 means the global maximum was found. None if no valid
            selected points exist.
        """
        if not self.space.selected_points:
            logger.warning("No selected points available to compute extremum gap.")
            return None
        
        if known_optimum == None:
            return None

        # Fetch precomputed values from point_value_dict
        selected_values = [
            self.space.point_value_dict.get(pt)
            for pt in self.space.selected_points
            if pt in self.space.point_value_dict
        ]

        if not selected_values:
            logger.warning(
                "No valid values found in point_value_dict for selected points."
            )
            return None

        max_selected = max(selected_values)
        
        gap = known_optimum - max_selected
        logger.info(f"Extremum Gap: {gap:.6f}")
        return gap

    ## Boundary Region Metrics
    def boundary_coverage(self, boundary_radius: float = None, vis=False) -> float:
        """
        Computes the boundary coverage as defined in Mullins et al. (2018):
        The fraction of all grid boundary points that are within `boundary_radius` of any selected scenario.

        Args:
            boundary_radius (float): Euclidean distance threshold (default Object setting).
        Returns:
            float: Coverage [0, 1].
        """
        if boundary_radius == None:
            boundary_radius = self.boundary_radius
        # Ensure meshgrid and values are available
        self._ensure_meshgrid_exists()
        space = self.space

        # Discretize values and compute boundary mask
        boundary_mask = space.create_boundary_mask(
            hop_distance=1, thresholds=space.criticality_thresholds
        )

        # Get grid coordinates and select only boundary points
        grid_points = np.vstack([g.ravel() for g in space.meshgrid]).T
        boundary_points = grid_points[boundary_mask.ravel()]

        selected_points = np.array(list(space.selected_points))
        if selected_points.ndim == 1 and selected_points.size > 0:
            selected_points = selected_points.reshape(1, -1)

        if selected_points.shape[0] == 0 or boundary_points.shape[0] == 0:
            logger.info("No selected or boundary points. Coverage is zero.")
            return 0.0

        # Build KDTree for selected points, check distance for each boundary point

        tree = cKDTree(selected_points)
        dists, _ = tree.query(boundary_points, k=1)
        covered = dists < boundary_radius
        coverage = np.sum(covered) / len(boundary_points)
        logger.info(
            f"Boundary Coverage (M_cov): {coverage:.4f} (radius={boundary_radius})"
        )

        # --- Plotting ---
        if vis:
            covered_points = boundary_points[covered]
            uncovered_points = boundary_points[~covered]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(
                grid_points[:, 0],
                grid_points[:, 1],
                s=5,
                color="lightblue",
                alpha=0.2,
                label="Grid Points",
            )
            ax.scatter(
                uncovered_points[:, 0],
                uncovered_points[:, 1],
                color="red",
                s=12,
                label="Uncovered Boundary",
            )
            ax.scatter(
                covered_points[:, 0],
                covered_points[:, 1],
                color="green",
                s=12,
                label="Covered Boundary",
            )
            ax.scatter(
                selected_points[:, 0],
                selected_points[:, 1],
                color="black",
                marker="x",
                s=20,
                label="Selected Points",
            )

            ax.set_title(f"Boundary Coverage Visualization (radius={boundary_radius})")
            ax.set_xlabel("D1")
            ax.set_ylabel("D2")
            ax.legend()
            ax.set_aspect("equal")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return coverage

    def boundary_precision(self, boundary_radius: float = None) -> float:
        """
        Compute the precision metric as defined as the proportion of
        generated samples (selected_points) that lie within a given
        radius of the true boundary region (boundary_points).

        Args:
            selected_points (np.ndarray): Sampled or predicted boundary points (N x d).
            boundary_points (np.ndarray): Ground-truth boundary points (M x d).
            radius (float): Euclidean distance threshold to consider a point "correct".

        Returns:
            float: Precision score in [0, 1]
        """
        if boundary_radius == None:
            boundary_radius = self.boundary_radius
        # Ensure meshgrid and values are available
        self._ensure_meshgrid_exists()
        space = self.space

        # Discretize values and compute boundary mask
        boundary_mask = space.create_boundary_mask(
            hop_distance=1, thresholds=space.criticality_thresholds
        )

        # Get grid coordinates and select only boundary points
        grid_points = np.vstack([g.ravel() for g in space.meshgrid]).T
        boundary_points = grid_points[boundary_mask.ravel()]

        selected_points = np.array(list(space.selected_points))
        if selected_points.ndim == 1 and selected_points.size > 0:
            selected_points = selected_points.reshape(1, -1)

        if selected_points.shape[0] == 0 or boundary_points.shape[0] == 0:
            logger.info("No selected or boundary points. Coverage is zero.")
            return 0.0

        tree = cKDTree(boundary_points)
        distances, _ = tree.query(selected_points, k=1)
        true_positive_mask = distances < boundary_radius
        n_true_positives = np.sum(true_positive_mask)
        n_predicted_positives = selected_points.shape[0]

        precision = (
            n_true_positives / n_predicted_positives
            if n_predicted_positives > 0
            else 0.0
        )
        logger.info(
            f"Boundary Precision (M_prec): {precision:.4f}"
        )
        return precision

    def boundary_effectiveness(self, boundary_radius: float = None) -> float:
        """
        Effectiveness M_eff: Ratio of selected points that are within a boundary region
        to the total number of queries (approx. all selected points, incl. eliminated ones).

        M_eff = N_p / N_query

        Returns:
            float: Effectiveness score in [0, 1]
        """
        if boundary_radius == None:
            boundary_radius = self.boundary_radius
        self._ensure_meshgrid_exists()
        space = self.space

        # All boundary points from grid
        boundary_mask = space.create_boundary_mask(
            hop_distance=1, thresholds=space.criticality_thresholds
        )
        grid_points = np.vstack([g.ravel() for g in space.meshgrid]).T
        boundary_points = grid_points[boundary_mask.ravel()]

        selected_points = np.array(list(space.selected_points))
        if selected_points.ndim == 1 and selected_points.size > 0:
            selected_points = selected_points.reshape(1, -1)

        if selected_points.shape[0] == 0 or boundary_points.shape[0] == 0:
            logger.info("No selected or boundary points. M_eff is zero.")
            return 0.0

        # Total SUT query count; fall back to selected_points length if not tracked
        if space.n_query_calls == 0:
            N_query = len(space.selected_points)
        else:
            N_query = space.n_query_calls

        # Count selected points within radius of the true boundary
        tree = cKDTree(boundary_points)
        distances, _ = tree.query(selected_points, k=1)
        N_p = np.sum(distances < boundary_radius)

        M_eff = N_p / N_query if N_query > 0 else 0.0
        logger.info(
            f"Boundary Effectiveness (M_eff): {M_eff:.4f} (N_p={N_p}, N_query={N_query})"
        )
        return M_eff

    ## Landscape reconstruction/ Model Metrics
    def model_approximation_error(self, surrogate_model) -> float:
        """
        Compute the MSE between surrogate predictions and ground-truth criticality.

        Args:
            surrogate_model: Fitted model with .predict(X) method.

        Returns:
            float: Mean Squared Error, or None if model or space is invalid.
        """

        if surrogate_model is None:
            logger.warning("No surrogate model provided.")
            return None

        if self.space.n_points is None:
            logger.warning("No grid defined. Cannot compute model approximation error.")
            return None

        self._ensure_meshgrid_exists()
        X = np.vstack([grid.ravel() for grid in self.space.meshgrid]).T

        # Retrieve or compute ground-truth values
        y_true = []
        for point in X:
            rounded = tuple(round(x, self.space.decimal_precision) for x in point)
            if rounded in self.space.point_value_dict:
                y_true.append(self.space.point_value_dict[rounded])
            else:
                val = self.space.get_values_for_points([point], save_points=False)[0]
                y_true.append(val)
        y_true = np.array(y_true)

        # Get surrogate predictions
        y_pred = surrogate_model.predict(X)

        mse = mean_squared_error(y_true, y_pred)
        logger.info(f"Model Approximation Error (MSE): {mse:.6f}")
        return mse

    def model_r2_score(self, surrogate_model) -> float:
        """
        Compute the R² score between surrogate model predictions and ground-truth values.

        Args:
            surrogate_model: Fitted model with .predict(X) method.

        Returns:
            float: R² score, or None if model or space is invalid.
        """
        if surrogate_model is None:
            logger.warning("No surrogate model provided.")
            return None

        if self.space.n_points is None:
            logger.warning("No grid defined. Cannot compute R².")
            return None

        # Reconstruct meshgrid points (shape: N x d)
        self._ensure_meshgrid_exists()
        X = np.vstack([grid.ravel() for grid in self.space.meshgrid]).T

        y_true = []
        for point in X:
            rounded = tuple(round(x, self.space.decimal_precision) for x in point)
            if rounded in self.space.point_value_dict:
                y_true.append(self.space.point_value_dict[rounded])
            else:
                val = self.space.get_values_for_points([point], save_points=False)[0]
                y_true.append(val)
        y_true = np.array(y_true)

        y_pred = surrogate_model.predict(X)
        r2 = r2_score(y_true, y_pred)
        logger.info(f"Model R² Score: {r2:.6f}")
        return r2
    
    def f1_coverage(
        self, surrogate_model, thresholds: Union[float, List[float]] = None, average: bool = False
    ) -> Union[float, Dict[float, float]]:
        """
        Compute F1 scores between selected and true critical regions over one or more thresholds.

        Args:
            thresholds (float or List[float], optional): Criticality threshold(s). If None, uses self.space.criticality_thresholds.
            average (bool): If True, returns mean F1 score over thresholds.

        Returns:
            Dict[float, float] or float: F1 score(s) per threshold, or average if specified.
        """
        self._ensure_meshgrid_exists()

        if thresholds is None:
            thresholds = self.space.criticality_thresholds
        elif isinstance(thresholds, float):
            thresholds = [thresholds]

        # Flatten the full value grid (ground truth)
        flat_values = self.space.values.ravel()

        # Build full grid of points and get surrogate mean predictions on the grid
        flat_points = np.vstack([axis.ravel() for axis in self.space.meshgrid]).T
        pred_mean = surrogate_model.predict(flat_points)
        pred_mean = np.asarray(pred_mean).ravel()

        if pred_mean.shape[0] != flat_values.shape[0]:
            raise ValueError(
                f"Prediction length {pred_mean.shape[0]} does not match grid size {flat_values.shape[0]}"
            )

        # Compute F1 score for each threshold
        f1_scores = {}

        for thresh in thresholds:
            y_true = (flat_values >= thresh).astype(int)
            y_pred = (pred_mean >= thresh).astype(int)

            if np.sum(y_true) == 0:
                f1 = 0.0  # No ground truth positive region
            else:
                f1 = f1_score(y_true, y_pred, zero_division=0)

            f1_scores[thresh] = f1
            logger.info(f"F1 Coverage (threshold={thresh}): {f1:.4f}")

        if average:
            mean_f1 = np.mean(list(f1_scores.values()))
            f1_scores['mean'] = mean_f1
            logger.info(f"Mean F1 Coverage over thresholds {thresholds}: {mean_f1:.4f}")

        return f1_scores
