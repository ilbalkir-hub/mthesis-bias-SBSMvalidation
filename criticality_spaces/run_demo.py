"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    End-to-end demo script for the criticality_spaces framework. Constructs a
    criticality Space from a JSON-defined scenario function, applies a sampling
    strategy, evaluates criticality metrics, and visualizes the result.

    - key role: Minimal working example; entry point for new users.
    - dependency: space.Space, functionloader, samplingstrategies
    - output: 3-D criticality surface plot

Usage:
    python run_demo.py

Configuration (edit the variables in main()):
    - space_file_path   : JSON file defining the criticality function(s)
                          (see Spaces/examples/ for templates)
    - distribution_file_path : JSON file specifying input distributions
                          (see Distributions/ for templates)
    - n_points          : number of evaluation points per sample call
    - dimensions        : list of (min, max) bounds per input dimension
    - criticality_thresholds : [low, high] thresholds that partition the space
                          into non-critical / critical / highly-critical regions
    - sampling strategy : replace ss.latin_hypercube_sampling(...) with any
                          function from samplingstrategies (e.g. random_sampling)
"""

from pathlib import Path
import json
from space import Space

from functionloader import load_functions_from_json
import samplingstrategies as ss


def main():
    # --- Search space configuration ---
    n_dim = 2                          # number of scenario input dimensions
    dimensions = [(0, 10)] * n_dim     # (min, max) bounds for each dimension
    n_points = 5                  # grid resolution for space evaluation

    base_path = Path(__file__).parent

    # --- Load criticality function(s) from JSON ---
    # JSON schema: see Spaces/examples/example2dwnoise.json
    # Switch to a different file to test other scenario functions.
    # space_file_path = base_path / "Spaces" / "generated_spaces" / "2D" / "6_2D.json"
    space_file_path = base_path / 'Spaces' / 'test_cases' / '2D' / 'with_noise' / 'test_3.json'
    function_instances = load_functions_from_json(space_file_path)

    # --- Load input distribution specification (optional) ---
    # Set distribution_specs=None to use uniform sampling instead.
    distribution_file_path = base_path / "Distributions" / "dist3d.json"
    with open(distribution_file_path, "r") as file:
        json_data = json.load(file)
    distribution_specs = json_data.get("distributions", None)

    # --- Construct criticality space ---
    # criticality_thresholds: [low, high] — partition the output into
    #   non-critical (< low), critical (low–high), highly-critical (> high).
    space = Space(
        dimensions=n_dim,
        functions=function_instances,
        n_points=n_points,
        distribution_specs=distribution_specs,
        criticality_thresholds=[0.3, 0.9],
    )

    # --- Apply sampling strategy ---
    # Replace with ss.random_sampling(dimensions, 500, n_points) or any other
    # strategy from samplingstrategies to benchmark a different method.
    samples = ss.latin_hypercube_sampling(dimensions, n_samples=50)

    # --- Evaluate and analyse ---
    # # space.get_values_for_points(samples)
    # Optionally query density at a specific point:
    # joint_prob, axis_probs = space.get_density_for_point((5.0, 5.0, 5.0))
    # # space.metrics.run_metrics_suite(method_categories=["general"])

    # --- Evaluate and analyse ---
    space.get_values_for_points(samples)
    
    # 1. Speichere das Ergebnis der Metriken in einer Variablen
    results = space.metrics.run_metrics_suite(method_categories=["general"])

    # 2. Gib die Ergebnisse explizit im Terminal aus
    print("\n" + "="*30)
    print("   BENCHMARK ERGEBNISSE")
    print("="*30)
    for metric_name, value in results.items():
        print(f"{metric_name:25}: {value}")
    print("="*30 + "\n")

    # --- Visualise ---
    # fixed_values: fix all but two dimensions for the 3-D surface plot;
    # set an entry to None to treat that axis as a free variable.
    space.visualizer.plot_3d_two_varied(
        show_critplane=True, fixed_values=[None, None, 5]
    )
    return


if __name__ == "__main__":
    main()
