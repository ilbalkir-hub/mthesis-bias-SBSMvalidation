"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Defines the central Space class, which represents a d-dimensional scenario
    parameter space with an associated criticality surface. The surface is
    composed by summing primitive function instances (from functions.py) over a
    regular meshgrid. Space provides unified evaluation, caching, discretisation,
    boundary detection, and lazy access to Metrics and SpaceVisualizer.

    - key role: Core data structure of the framework; all sampling, metric, and
                visualisation operations are performed through a Space instance.
    - dependency: functions.py, distributions.py, metrics.py, visualizations.py
    - output: Criticality values (continuous and discretised); point-value cache;
              boundary mask; metric results via space.metrics.

Usage:
    from space import Space
    from functionloader import load_functions_from_json
    fns = load_functions_from_json("Spaces/examples/example2dwnoise.json")
    space = Space(dimensions=2, functions=fns, n_points=100,
                  criticality_thresholds=[0.3, 0.9])
    space.get_values_for_points(samples)
    space.metrics.run_metrics_suite(method_categories=["general"])
"""

import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Callable, Union, Dict, Set

from distributions import DistributionHandler
from metrics import Metrics
from visualizations import SpaceVisualizer


class Space:
    """
    d-dimensional criticality space composed of summed analytic function primitives.

    On construction the full meshgrid is evaluated (if n_points is given) and
    cached in ``point_value_dict`` for O(1) lookup during subsequent sampling.
    Metrics and visualisation are accessed via lazy properties to avoid circular
    imports and unnecessary initialisation.
    """

    def __init__(
        self,
        dimensions: List[Tuple],
        functions: List[Callable],
        n_points: int = None,
        decimal_precision: int = 6,
        criticality_thresholds: List[float] = [0.6],
        distribution_specs: List[Dict] = None,
        show_progressbar=True,
        active_point_value_dict=True,
    ):
        """
        Parameters
        ----------
        dimensions : int or list of (float, float)
            If int, creates that many dimensions with default bounds (0, 10).
            If list, each entry is a (min, max) bound for one dimension.
        functions : list of callable
            Ordered list of criticality function instances (from functions.py).
            Their outputs are summed to form the composite criticality surface.
        n_points : int, optional
            Grid resolution per dimension. If provided, the full meshgrid is
            computed at construction; if None, evaluation is on-demand only.
        decimal_precision : int
            Rounding precision for point coordinates used as dict keys.
        criticality_thresholds : list of float
            Thresholds that partition the output into criticality classes.
            E.g. [0.3, 0.9] → non-critical / critical / highly-critical.
        distribution_specs : list of dict, optional
            Per-axis distribution specifications passed to DistributionHandler.
            If None, density queries are disabled.
        show_progressbar : bool
            Display tqdm progress bar during meshgrid evaluation.
        active_point_value_dict : bool
            If True, cache newly evaluated points in point_value_dict for reuse.
        """
        if isinstance(dimensions, int):
            self.dimensions = [(0, 10)] * dimensions
        else:
            self.dimensions = dimensions
        self.active_point_value_dict = active_point_value_dict
        self.n_query_calls = 0
        self.show_progressbar = show_progressbar
        self.criticality_thresholds = criticality_thresholds
        self.selected_points: List[Tuple[float, ...]] = []
        self.functions = functions
        self.decimal_precision = decimal_precision
        self.n_points = n_points
        if n_points is not None:
            self.meshgrid = self.generate_meshgrid(n_points)
            self.values = self.calculate_space()
            self.point_value_dict = self.create_point_value_dict()
        else:
            self.point_value_dict = {}
        self.distribution_handler = (
            DistributionHandler(distribution_specs)
            if distribution_specs is not None
            else None
        )
        self.boundary_mask = None

    @property
    def metrics(self):
        if not hasattr(self, "_metrics"):
            self._metrics = Metrics(self)
        return self._metrics

    @property
    def visualizer(self):
        if not hasattr(self, "_visualizer"):
            self._visualizer = SpaceVisualizer(self)
        return self._visualizer

    def discretize_values(
        self, values: np.ndarray, thresholds: List[float] = None
    ) -> np.ndarray:
        """Map continuous criticality values to integer class labels via thresholds.
        Label 0 = non-critical, 1 = critical, 2 = highly-critical (for two thresholds)."""
        thresholds = np.asarray(thresholds)
        return np.searchsorted(thresholds, values, side="right")

    def reset_selected_points(self):
        """Clear selected_points and reset the SUT query counter."""
        self.selected_points.clear()
        self.n_query_calls = 0
        return

    def reset(self):
        """Clear selected_points and flush the point-value cache."""
        self.selected_points.clear()
        self.point_value_dict = {}

    def generate_meshgrid(self, n_points):
        """Return a list of d coordinate arrays on a uniform n_points-per-dim grid."""
        ranges = [np.linspace(start, stop, n_points) for start, stop in self.dimensions]
        return np.meshgrid(*ranges, indexing="ij")

    def calculate_space(self, meshgrid=None):
        """
        Evaluate the composite criticality surface over the meshgrid.
        Each function's contribution is added only within its active sub-region
        (via the boolean mask returned by ``calculate_relevant_values``).

        Returns
        -------
        np.ndarray
            Array of criticality values with the same shape as the meshgrid.
        """
        if meshgrid is None:
            meshgrid = self.meshgrid
        summed_values = np.zeros_like(meshgrid[0], dtype=np.float64)

        for function in tqdm(
            self.functions,
            desc="Processing functions",
            disable=not self.show_progressbar,
        ):
            subgrid_mask, values = function.calculate_relevant_values(meshgrid)
            summed_values[subgrid_mask] += values

        return summed_values

    def create_point_value_dict(self):
        """Build a dict mapping rounded grid-point tuples to their criticality values.
        Used for O(1) cache lookup during ``get_values_for_points``."""
        # Flatten the meshgrid and values
        meshgrid_flat = np.vstack([grid.ravel() for grid in self.meshgrid]).T
        values_flat = self.values.ravel()

        # Round the coordinates using vectorized operations
        rounded_meshgrid_flat = np.round(meshgrid_flat, self.decimal_precision)

        # Convert the rounded coordinates to tuples
        point_tuples = [tuple(point) for point in rounded_meshgrid_flat]

        # Use dictionary comprehension to create the point-value dictionary
        point_value_dict = {
            point: value for point, value in zip(point_tuples, values_flat)
        }
        return point_value_dict

    def activate_bias(self, bias_object):
        """Aktiviert einen Bias durch ein übergebenes Bias-Objekt."""
        self.bias = bias_object
        print(f"\n ACHTUNG: Bias '{self.bias.name}' wurde aktiviert!")

    def deactivate_bias(self):
        self.bias = None

    def get_values_for_points(self, points: Union[np.ndarray, list], save_points=True):
        """
        Evaluate the criticality surface at arbitrary points, using the cache where possible.

        Parameters
        ----------
        points : array-like of shape (N, d)
            The scenario parameter points to evaluate.
        save_points : bool
            If True, append all points to ``selected_points`` (used by metrics).
            Set to False for internal reference evaluations that should not be
            counted as method queries.

        Returns
        -------
        np.ndarray of shape (N,)
            Criticality values at each input point.
        """
        if any(len(point) != len(self.dimensions) for point in points):
            raise ValueError(
                "The dimensions of the point do not match the dimensions of the benchmark."
            )
        if isinstance(points, (set, tuple)):
            points = list(points)
        points = np.array(points)
        rounded_points = [
            tuple(round(coord, self.decimal_precision) for coord in point)
            for point in points
        ]
        values = np.zeros(points.shape[0])

        points_to_calculate = []
        for idx, rp in enumerate(rounded_points):
            if rp in self.point_value_dict:
                values[idx] = self.point_value_dict[rp]
            else:
                points_to_calculate.append((idx, rp))
        if save_points:
            for rp in rounded_points:
                self.selected_points.append(rp)

        if points_to_calculate:
            indices, points_needing_calculation = zip(*points_to_calculate)
            points_needing_calculation = np.array(points_needing_calculation)
            calculated_values = np.zeros(points_needing_calculation.shape[0])

            for function in self.functions:
                function_values = function.get_values_for_points(
                    points_needing_calculation
                )
                calculated_values += function_values

            if self.active_point_value_dict:
                for idx, value in zip(indices, calculated_values):
                    values[idx] = value
                    self.point_value_dict[rounded_points[idx]] = value

        # ==========================================
        # NEU: BIAS INJEKTION FÜR DAS EXPERIMENT
        # ==========================================
        # Wir belügen den Algorithmus nur, wenn "save_points=True" ist. 
        # Das bedeutet, der Selektor fragt gerade aktiv Punkte an.
        # Nun in OOP-Version

        force_illusion = getattr(self, "force_evaluation_bias", False)
        
        if save_points and getattr(self, "bias", None) is not None:
            biased_values = np.copy(values)
            for i, point in enumerate(points):
                # Egal welcher Bias aktiv ist, wir rufen einfach .apply() auf!
                biased_values[i] = self.bias.apply(biased_values[i], point)
            return biased_values
        # ==========================================

        # Wenn kein Bias aktiv ist oder Metriken berechnet werden, sag die Wahrheit
        return values

    def get_discrete_values_for_points(
        self,
        points: Union[np.ndarray, list],
        thresholds: List[float] = None,
        save_points: bool = True,
    ) -> np.ndarray:
        """Evaluate points and return integer class labels according to thresholds.
        Convenience wrapper around ``get_values_for_points`` + ``discretize_values``."""
        continuous_values = self.get_values_for_points(points, save_points=save_points)

        discrete_labels = self.discretize_values(continuous_values, thresholds)

        return discrete_labels

    def get_density_for_point(self, point: Union[List[float], Tuple[float]]):
        """
        Computes the marginal (axis-specific) and joint probability density at a given point.

        This method assumes that the axis distributions are independent and returns both the
        individual densities per axis and the joint density as their product.

        Note:
            These are probability densities, not probabilities. To estimate a probability,
            the joint density must be multiplied by the volume element (i.e., resolution)
            of the discretization in each dimension.

        Args:
            point (Union[List[float], Tuple[float]]): A point in the input space.

        Returns:
            Tuple[float, List[float]]:
                A tuple containing:
                - joint_density: The joint probability density at the point.
                - axis_densities: A list of the marginal densities for each axis.
        """
        if self.distribution_handler is None:
            raise ValueError(
                "No distribution handler defined. Please provide distribution specs."
            )

        return self.distribution_handler.get_joint_density(point)

    def selected_indices(self):
        """
        Return indices of selected points in the meshgrid.
        Useful for mask-based evaluations.
        """
        index_map = {
            tuple(np.round(pt, self.decimal_precision)): idx
            for idx, pt in enumerate(zip(*[grid.ravel() for grid in self.meshgrid]))
        }
        return [
            index_map.get(tuple(np.round(pt, self.decimal_precision)), None)
            for pt in self.selected_points
            if tuple(np.round(pt, self.decimal_precision)) in index_map
        ]

    ## Helper functions
    def create_boundary_mask(
        self, hop_distance: int = 1, thresholds: List[float] = None
    ) -> np.ndarray:
        """
        Create a binary mask over the full grid that marks boundary regions based on label changes.

        Args:
            hop_distance (int): Neighborhood radius (in hops) to scan for label change.
            thresholds (List[float]): List of criticality thresholds to discretize values.

        Returns:
            boundary_mask (np.ndarray): Boolean mask of same shape as self.values, True at boundary points.
        """
        if self.n_points is None:
            raise ValueError(
                "No meshgrid defined. Run with n_points to create grid first."
            )
        
        if self.boundary_mask is not None:
            return self.boundary_mask

        # Step 1: Discretize the full space
        discrete_labels = self.discretize_values(
            self.values, thresholds or self.criticality_thresholds
        )

        # Step 2: Create padded array to support border neighborhoods
        pad_width = [(hop_distance, hop_distance) for _ in discrete_labels.shape]
        padded_labels = np.pad(discrete_labels, pad_width, mode="edge")

        # Step 3: Scan neighborhood
        boundary_mask = np.zeros_like(discrete_labels, dtype=bool)

        # Create all offsets in the neighborhood except the zero vector
        import itertools

        offsets = list(
            itertools.product(
                *[range(-hop_distance, hop_distance + 1) for _ in self.meshgrid]
            )
        )
        offsets.remove((0,) * len(self.meshgrid))  # Remove self

        it = np.nditer(discrete_labels, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            center_label = discrete_labels[idx]

            is_boundary = False
            for offset in offsets:
                neighbor_idx = tuple(i + o + hop_distance for i, o in zip(idx, offset))
                neighbor_label = padded_labels[neighbor_idx]
                if neighbor_label != center_label:
                    is_boundary = True
                    break
            boundary_mask[idx] = is_boundary
            it.iternext()

        # Cache result for downstream use
        self.boundary_mask = boundary_mask
        return boundary_mask
