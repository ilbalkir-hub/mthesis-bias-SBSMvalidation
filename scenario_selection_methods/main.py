"""
Project: Toward Standardized Benchmarking of Search-Based Scenario Selection Methods in Autonomous System Validation
Version: 1.0.0

Description:
    Benchmark entry point: iterates over pre-generated test-case spaces,
    instantiates each configured selector via the factory, runs it, computes
    the metric suite, and writes results incrementally to a timestamped XLSX
    file with both raw results and per-run summary statistics.

    - key role: Orchestrates the full evaluation loop across spaces, methods,
                dimensionalities, and repeated runs.
    - dependency: criticality_spaces (Space, functionloader), factory,
                  pandas, openpyxl, concurrent.futures
    - output: ``evaluations/evaluation_results_<timestamp>.xlsx`` with sheets
              ``raw_results`` and ``summary_stats``;
              ``evaluations/evaluation.log`` with per-run progress.

Usage:
    # Configure methods, dimensionalities, and budget in main(), then run:
    python main.py
"""


from pathlib import Path
import sys
import logging, re
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

base_path = Path(__file__).parent
cs_path = (base_path.parent / "criticality_spaces").resolve()
if cs_path not in sys.path:
    sys.path.insert(0, str(cs_path))

from functionloader import load_functions_from_json
from space import Space
from factory import get_selector
from bias import GlobalBias, RegionBias, LocalStructuralBias, TopologyBoundaryBias, CalibrationBias, RegionDependentBias, LocalizationBias, BiasChain

logging.basicConfig(
    filename="evaluations/evaluation.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

def to_display_path(p: Path, anchor: str = "test_cases") -> str:
    """
    Return a short, portable path starting at the given anchor directory.
    Example: .../Spaces/test_cases/6D/without_noise/test_1.json
             -> test_cases/6D/without_noise/test_1.json
    Falls back to file name if the anchor is not found.
    """
    parts = p.parts
    try:
        i = parts.index(anchor)
        return Path(*parts[i:]).as_posix()  # forward slashes for consistency
    except ValueError:
        # If the anchor isn't present, try a best-effort relative path
        return p.name


def evaluate_single_config(
    space: Space,
    config_file: Path,
    n_dim: int,
    method_name: str,
    n_select: int,
    run_id: int,
    eval_metrics: list
):
    """
    Run a single selector on one space and return a flat result dict.

    Instantiates the selector via ``get_selector``, calls ``select()`` or
    ``get_model()`` depending on the selector type, measures wall-clock
    execution time, and evaluates the requested metric categories.

    Parameters
    ----------
    space : Space
        Pre-constructed criticality space (already has a meshgrid).
    config_file : Path
        Path to the JSON spec that produced the space (used for logging and
        the result record).
    n_dim : int
        Number of dimensions (stored in the result record).
    method_name : str
        Selector key understood by ``get_selector`` (e.g. ``"gpr"``).
    n_select : int
        Sample budget passed to the selector.
    run_id : int
        1-based repetition index (stored in the result record).
    eval_metrics : list of str
        Metric category keys forwarded to ``space.metrics.run_metrics_suite``.

    Returns
    -------
    dict
        Flat dictionary with keys: file, dimension, noise, method, run,
        execution_time_sec, and all metric values returned by the suite.

    Raises
    ------
    NotImplementedError
        If the selector implements neither ``select()`` nor ``get_model()``.
    """
    logging.info(f"Evaluating {config_file} with method '{method_name}' [Run {run_id}]")

    
    selector = get_selector(method_name, space, n_select)

    if hasattr(selector, "select"):
        start_time = time.perf_counter()
        selector.select()
        duration = time.perf_counter() - start_time
        metrics = space.metrics.run_metrics_suite(
            method_categories=eval_metrics
        )
    elif hasattr(selector, "get_model"):
        start_time = time.perf_counter()
        model = selector.get_model()
        duration = time.perf_counter() - start_time
        metrics = space.metrics.run_metrics_suite(
            method_categories=eval_metrics, surrogate_model=model
        )
    else:
        raise NotImplementedError(
            f"The selector class '{type(selector).__name__}' must implement either "
            f"'select()' or 'get_model()'."
        )

   # ==========================================
    # 1. METRIKEN DER REALITÄT (Ground Truth)
    # ==========================================
    if hasattr(selector, "get_model"):
        metrics_real = space.metrics.run_metrics_suite(method_categories=eval_metrics, surrogate_model=model)
    else:
        metrics_real = space.metrics.run_metrics_suite(method_categories=eval_metrics)
    
    # ==========================================
    # 2. METRIKEN DER ILLUSION (Sicher & Manuell berechnet)
    # ==========================================
    # Wir kopieren die echten Metriken als Basis
    metrics_illusion = metrics_real.copy()
    
    # Jetzt überschreiben wir die Werte, die der Selektor gesehen hat
    if getattr(space, "bias", None) is not None and len(space.selected_points) > 0:
        # Hole die wahren Werte (Ground Truth)
        raw_values = space.get_values_for_points(space.selected_points, save_points=False)
        
        # Wende den Bias an -> Das ist die Illusion!
        perceived_values = [
            space.bias.apply(val, pt) for val, pt in zip(raw_values, space.selected_points)
        ]
        
        # Überschreibe die relevanten Metriken in der Illusion
        metrics_illusion["average_criticality_selected"] = sum(perceived_values) / len(perceived_values)
        metrics_illusion["max_criticality_selected"] = max(perceived_values)
        metrics_illusion["min_criticality_selected"] = min(perceived_values)
    # ==========================================

    # Die Basis-Informationen
    base_info = {
        "file": to_display_path(config_file),
        "dimension": n_dim,
        "noise": "with_noise" if "with_noise" in str(config_file) else "without_noise",
        "method": method_name,
        "run": run_id,
        "bias": space.bias.name if getattr(space, "bias", None) is not None else "None",
        "execution_time_sec": duration
    }

    row_real = {**base_info, "world": "Reality", **metrics_real}
    row_illusion = {**base_info, "world": "Illusion", **metrics_illusion}

    logging.info(f"Completed {config_file} [Run {run_id}]\n")
    return [row_real, row_illusion]


def next_power_of_two(x):
    """Return the smallest power of two that is >= x."""
    return 1 << (x - 1).bit_length()


def extract_test_number(path):
    """Extract the integer test index from a filename like ``test_3.json``."""
    match = re.search(r"test_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else float("inf")

def append_to_excel(output_file: Path, df_new: pd.DataFrame, sheet_name="raw_results"):
    """
    Append ``df_new`` to the given sheet of an XLSX file, creating it if absent.

    Parameters
    ----------
    output_file : Path
        Target XLSX file path.
    df_new : pd.DataFrame
        Rows to append.
    sheet_name : str, optional
        Sheet to read from and write to (default ``"raw_results"``).
    """
    if output_file.exists():
        # Read existing data and append new rows
        df_existing = pd.read_excel(output_file, sheet_name=sheet_name)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # First write — nothing to merge
        df_all = df_new

    # Rewrite the complete table
    with pd.ExcelWriter(output_file, mode="w", engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name=sheet_name, index=False)

def summary_to_excel(
    all_results: list,
    output_file: Path,
    group_keys = ["file", "method", "dimension", "noise"],
    summary_sheet: str = "summary_stats",
    results_sheet: str = "raw_results"
):
    """
    Write raw results and per-group mean/std summary to two XLSX sheets.

    Parameters
    ----------
    all_results : list of dict
        Flat result dicts as returned by ``evaluate_single_config``.
    output_file : Path
        Target XLSX file (overwritten in full).
    group_keys : list of str, optional
        Columns used for grouping when computing mean and std.
    summary_sheet : str, optional
        Sheet name for the aggregated statistics.
    results_sheet : str, optional
        Sheet name for the raw per-run results.

    Raises
    ------
    ValueError
        If ``all_results`` is empty.
    """
    if not all_results:
        raise ValueError("No results to summarize.")

    df_all = pd.DataFrame(all_results)
    if "run" in df_all.columns:
        df_all = df_all.drop(columns=["run"])

    # Compute mean and std
    df_mean = df_all.groupby(group_keys).mean(numeric_only=True).reset_index()
    df_std = df_all.groupby(group_keys).std(numeric_only=True).reset_index()

    # Rename columns
    df_mean = df_mean.rename(
        columns={col: f"{col}_mean" for col in df_mean.columns if col not in group_keys}
    )
    df_std = df_std.rename(
        columns={col: f"{col}_std" for col in df_std.columns if col not in group_keys}
    )

    df_summary = pd.merge(df_mean, df_std, on=group_keys)

    # Write both sheets in a single pass (overwrite mode)
    with pd.ExcelWriter(output_file, mode="w", engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name=results_sheet, index=False)
        df_summary.to_excel(writer, sheet_name=summary_sheet, index=False)

    return


def main():
    """
    Configure and run the full benchmark evaluation loop.

    Edit ``method_to_evaluations``, ``n_runs``, ``n_points``, ``n_select``,
    and the dimension/noise loops below to reproduce specific paper experiments
    or define new ones. Results are written incrementally so a partial run is
    never lost.
    """
    test_root = (
        Path(__file__).resolve().parent.parent
        / "criticality_spaces"
        / "Spaces"
        / "test_cases"
    )

    method_to_evaluations = {
        # "random": ["general"],
        # "lhs": ["general"],
        # "sobol": ["general", "boundary_detection"],
        # "nndv": ["general", "boundary_detection"],
        # "gdnnas": ["general", "boundary_detection"],
        # "idw": ["general", "model_reconstruction"],
        # "ann": ["general", "model_reconstruction"],
        "gpr": ["general", "extremum_search"],
        # "ipso": ["general", "extremum_search"]
    }

    n_runs = 1
    n_points = 101
    n_select = 100 # n_select = next_power_of_two(9000)
    
    all_results = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"evaluation_results_{timestamp}.xlsx"

    for dimension_str in ["2D"]:  # Optionally ["2D", "3D", "4D"]
        n_dim = int(dimension_str.replace("D", ""))
        for noise_type in ["without_noise"]: # ["with_noise", "without_noise"]
            config_dir = test_root / dimension_str / noise_type
            if not config_dir.exists():
                logging.exception(f"Could not find path: {config_dir}")
                continue

            config_files = sorted(
                config_dir.glob("*.json"), key=extract_test_number
            )
            for config_file in config_files:
                functions = load_functions_from_json(config_file)
                space = Space(dimensions=[(0, 10)] * n_dim, functions=functions, n_points=n_points, criticality_thresholds=None)

                # Bias per OOP
                
                # 1. Die einzelnen Störfaktoren definieren
                Bias1 = LocalizationBias(space=space, x_shift=1.5, y_shift=-0.5)
                Bias2 = TopologyBoundaryBias(boundary_margin=1.5, penalty=0.5)
                Bias3 = RegionDependentBias(x_range=(4.0, 7.0), y_range=(4.0, 7.0), drop_factor=0.2)
                Bias4 = LocalStructuralBias(amplitude=0.07, frequency=12.0)
                Bias5 = CalibrationBias(scale=1.2, offset=0.0)
                # 2. Die Pipeline zusammenbauen (Reihenfolge beachten, Verschiebung in Inputdimension muss als erste gemacht werden)
                mein_experiment_bias = BiasChain([Bias1, Bias2, Bias3,Bias4,Bias5])
                space.activate_bias(mein_experiment_bias)

                # space.deactivate_bias()
                # ==========================================

                for method, eval_metrics in method_to_evaluations.items():
                    for run in range(1, n_runs + 1):
                        logging.info(f" Running {dimension_str} {config_file} - {method} - run: {run}")
                        print(f" Running {dimension_str} {config_file} - {method} - run: {run}")
                        space.reset_selected_points()
                        result_list = evaluate_single_config(
                            space, config_file, n_dim, method, n_select, run, eval_metrics)
                        if result_list:
                            # result_list enthält jetzt unsere 2 Zeilen!
                            for res in result_list:
                                all_results.append(res)

                            # Beide Zeilen direkt in die Excel schreiben
                            df_current = pd.DataFrame(result_list)
                            append_to_excel(output_file, df_current, sheet_name="raw_results")

                        space.visualizer.plot_3d_two_varied(simulate_bias=False)
                        space.visualizer.plot_3d_two_varied(simulate_bias=True)
                break # Only run 1 Space

    # Post-process results
    if all_results:
        summary_to_excel( all_results, output_file)
        logging.info(f"Saved evaluation results and summary to {output_file}")
    else:
        logging.warning("No results to save.")


if __name__ == "__main__":
    main()
