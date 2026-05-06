from pathlib import Path
import sys
import time
import pandas as pd
from datetime import datetime

# Pfade setzen
base_path = Path(__file__).parent
cs_path = (base_path.parent / "criticality_spaces").resolve()
if cs_path not in sys.path:
    sys.path.insert(0, str(cs_path))

from functionloader import load_functions_from_json
from space import Space
from factory import get_selector
from bias import  RegionDependentBias, CalibrationBias, BiasChain, AxisGradientBias

def evaluate_single_config(space, method_name, n_select, eval_metrics):
    """Führt einen Selektor aus und vergleicht Illusion mit Realität."""
    print(f"\n🚀 Starte Selektor: {method_name}")
    selector = get_selector(method_name, space, n_select)

    # NEU: Der intelligente Metrik-Filter (verhindert Abstürze)
    safe_metrics = eval_metrics.copy() 
    if not hasattr(selector, "get_model") and "model_reconstruction" in safe_metrics:
        safe_metrics.remove("model_reconstruction")
        print(f"   ⚠️ Warnung: '{method_name}' hat kein Modell. 'model_reconstruction' übersprungen!")

    start_time = time.perf_counter()
    
    # Nutzt jetzt "safe_metrics" statt "eval_metrics"
    if hasattr(selector, "select"):
        selector.select()
        metrics_real = space.metrics.run_metrics_suite(method_categories=safe_metrics)
    elif hasattr(selector, "get_model"):
        model = selector.get_model()
        metrics_real = space.metrics.run_metrics_suite(method_categories=safe_metrics, surrogate_model=model)
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
    print("="*50)
    print("START: ALGORITHMUS ENTWICKLUNGS-UMGEBUNG")
    print("="*50)

    # Testraum
    test_file_path = base_path.parent / "criticality_spaces" / "Spaces" / "test_cases" / "2D" / "without_noise" / "test_5.json"
    print(f"[1] Lade Raum: {test_file_path.name}")
    
    functions = load_functions_from_json(test_file_path)
    space = Space(dimensions=[(0, 10), (0, 10)], functions=functions, n_points=101, criticality_thresholds=None)

    # 2. BIAS AKTIVIEREN
    print("[2] Biases aktivieren...")
    #bias1 = LocalizationBias(space=space, x_shift=-2, y_shift=-1)
    #bias2 = TopologyBoundaryBias(boundary_margin=1.5, penalty=-0.7)
    #bias3 = RegionDependentBias(x_range=(4.0, 9.0), y_range=(4.0, 9.0), drop_factor=0.1)
    #bias4 = LocalStructuralBias(amplitude=0.02, frequency=10.0)
    #bias5 = CalibrationBias(scale=0.8, offset=0.0)
    bias6  = AxisGradientBias(axis_index=0, start_coord=0, drop_per_unit=-0.02)
    
    ultimate_pipeline = BiasChain([bias6])
    space.activate_bias(ultimate_pipeline)

    # 3. SELEKTOR STARTEN
    method = "sobol"
    n_select = 100
    metrics_to_calc = ["general", "extremum_search"]
    
    space.reset_selected_points()
    metrics_real, metrics_illu, duration = evaluate_single_config(space, method, n_select, metrics_to_calc)

    # 4. EXCEL EXPORT (NEU)
    print("\n[4] Speichere Ergebnisse in Excel...")
    base_info = {
        "Testdatei": test_file_path.name,
        "Methode": method,
        "Budget (n_select)": n_select,
        "Dauer (Sekunden)": round(duration, 2)
    }

    row_real = {**base_info, "Welt": "Realität (Ground Truth)", **metrics_real}
    row_illusion = {**base_info, "Welt": "Illusion (Sensordaten)", **metrics_illu}

    output_dir = base_path / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"dev_ergebnisse_{method}_{timestamp}.xlsx"

    df = pd.DataFrame([row_illusion, row_real])
    df.to_excel(output_file, index=False, sheet_name="Metriken")

    # 5. ERGEBNISSE AUSGEBEN
    print("\n" + "-"*50)
    print(f"📊 ERGEBNISSE ({method}) - Dauer: {duration:.2f}s")
    print(f"📂 Datei: {output_file}")
    print("-" * 50)
    
    # .get() schützt vor Abstürzen, falls eine Metrik mal fehlen sollte
    print(f"Maximal gefundene Gefahr (Illusion): {metrics_illu.get('max_criticality_selected', 0):.4f}")
    print(f"Maximal gefundene Gefahr (Realität): {metrics_real.get('max_criticality_selected', 0):.4f}")
    print(f"Sim-to-Reality GAP: {metrics_illu.get('max_criticality_selected', 0) - metrics_real.get('max_criticality_selected', 0):+.4f}")

    # 6. VISUALISIERUNG
    print("\n[6] Öffne 3D-Plots...")
    space.visualizer.plot_3d_two_varied(simulate_bias=False)
    space.visualizer.plot_3d_two_varied(simulate_bias=True)

if __name__ == "__main__":
    main()