from pathlib import Path
import sys
import time

# Pfade setzen
base_path = Path(__file__).parent
cs_path = (base_path.parent / "criticality_spaces").resolve()
if cs_path not in sys.path:
    sys.path.insert(0, str(cs_path))

from functionloader import load_functions_from_json
from space import Space
from factory import get_selector
from bias import LocalizationBias, TopologyBoundaryBias, RegionDependentBias, LocalStructuralBias, CalibrationBias, BiasChain

def evaluate_single_config(space, method_name, n_select, eval_metrics):
    """Führt einen Selektor aus und vergleicht Illusion mit Realität."""
    print(f"\n Starte Selektor: {method_name}")
    selector = get_selector(method_name, space, n_select)

    start_time = time.perf_counter()
    if hasattr(selector, "select"):
        selector.select()
        metrics_real = space.metrics.run_metrics_suite(method_categories=eval_metrics)
    elif hasattr(selector, "get_model"):
        model = selector.get_model()
        metrics_real = space.metrics.run_metrics_suite(method_categories=eval_metrics, surrogate_model=model)
    else:
        raise NotImplementedError("Selektor muss 'select()' oder 'get_model()' haben.")
    
    duration = time.perf_counter() - start_time

    # Metriken der Illusion berechnen
    metrics_illusion = metrics_real.copy()
    if getattr(space, "bias", None) is not None and len(space.selected_points) > 0:
        raw_values = space.get_values_for_points(space.selected_points, save_points=False)
        perceived_values = [space.bias.apply(val, pt) for val, pt in zip(raw_values, space.selected_points)]
        
        metrics_illusion["average_criticality_selected"] = sum(perceived_values) / len(perceived_values)
        metrics_illusion["max_criticality_selected"] = max(perceived_values)
        metrics_illusion["min_criticality_selected"] = min(perceived_values)

    return metrics_real, metrics_illusion, duration


def main():

    # Testraum
    test_file_path = base_path.parent / "criticality_spaces" / "Spaces" / "test_cases" / "2D" / "without_noise" / "test_4.json"
    print(f"[1] Lade Raum: {test_file_path.name}")
    
    functions = load_functions_from_json(test_file_path)
    space = Space(dimensions=[(0, 10), (0, 10)], functions=functions, n_points=101, criticality_thresholds=None)

    # 2. BIAS AKTIVIEREN
    print("Biases aktivieren...")
    bias1 = LocalizationBias(space=space, x_shift=-2, y_shift=-1)
    #bias2 = TopologyBoundaryBias(boundary_margin=1.5, penalty=-0.7)
    #bias3 = RegionDependentBias(x_range=(4.0, 9.0), y_range=(4.0, 9.0), drop_factor=0.1)
    #bias4 = LocalStructuralBias(amplitude=0.02, frequency=10.0)
    #bias5 = CalibrationBias(scale=0.8, offset=0.0)
    
    ultimate_pipeline = BiasChain([bias1])
    space.activate_bias(ultimate_pipeline)

    # 3. SELEKTOR STARTEN
    method = "sobol"
    n_select = 100
    metrics_to_calc = ["general", "extremum_search" ]
    
    space.reset_selected_points()
    metrics_real, metrics_illu, duration = evaluate_single_config(space, method, n_select, metrics_to_calc)

    # 4. ERGEBNISSE AUSGEBEN
    print("\n" + "-"*50)
    print(f"ERGEBNISSE ({method}) - Dauer: {duration:.2f}s")
    print("-" * 50)
    print(f"Maximal gefundene Gefahr (Illusion): {metrics_illu['max_criticality_selected']:.4f}")
    print(f"Maximal gefundene Gefahr (Realität): {metrics_real['max_criticality_selected']:.4f}")
    print(f"Sim-to-Reality GAP: {metrics_illu['max_criticality_selected'] - metrics_real['max_criticality_selected']:+.4f}")

    # 5. VISUALISIERUNG
    space.visualizer.plot_3d_two_varied(simulate_bias=False)
    space.visualizer.plot_3d_two_varied(simulate_bias=True)

if __name__ == "__main__":
    main()