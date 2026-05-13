from pathlib import Path
import sys
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Pfade setzen
base_path = Path(__file__).parent
cs_path = (base_path.parent / "criticality_spaces").resolve()
if cs_path not in sys.path:
    sys.path.insert(0, str(cs_path))

from functionloader import load_functions_from_json
from space import Space
from factory import get_selector
from bias import RegionDependentBias, CalibrationBias, BiasChain, AxisGradientBias

def evaluate_single_config(space, method_name, n_select, eval_metrics):
    """Führt einen Selektor aus und vergleicht Illusion mit Realität."""
    print(f" Starte Selektor: {method_name}")
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

    return metrics_real, metrics_illusion, duration, selector


# --- NEUE FUNKTION ZUM PLOTTEN DER FEHLERLANDSCHAFTEN ---
def plot_standard_gpr_landscape(selector, space, resolution=50):
    """
    Erstellt einen 2-Panel 3D-Plot für den Standard-GPR ohne Klassifizierung.
    Zeigt: Echter Fehler vs. Gelerntes Einzel-Modell.
    """
    print("\n[Visualisierung] Generiere Fehler-Landschaft für Standard-GPR (2 Panels)...")
    
    bounds = space.dimensions
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    # 1. Echten Fehler berechnen
    y_true = np.array(space.get_values_for_points(grid_points, save_points=False))
    y_illus = np.array([space.bias.apply(v, p) for v, p in zip(y_true, grid_points)])
    Z_true = (y_true - y_illus).reshape(X.shape)
    
    # 2. Gelernten Fehler berechnen (Nur ein GPR!)
    if hasattr(selector.gpr, "X_train_"):
        mean, _ = selector.gpr.predict(grid_points, return_std=True)
    else:
        mean = np.zeros(len(grid_points))
    Z_pred = mean.reshape(X.shape)
    
    # 3. Plotten im 1x2 Grid
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Ablationsstudie: Standard GPR-Verhalten ohne Klassifikator", fontsize=14, fontweight='bold')
    
    # Linker Plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_true, cmap='Reds', alpha=0.8, edgecolor='none')
    ax1.set_title("Echte Realität (Wahrer Fehler)")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    
    # Rechter Plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_pred, cmap='Oranges', alpha=0.8, edgecolor='none')
    ax2.set_title("Gelerntes Modell (Einziger GPR)")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

    # Trainingspunkte einzeichnen
    if len(selector.space.selected_points) > 0:
        selected_X = np.array(selector.space.selected_points)
        pt_y_true = np.array(space.get_values_for_points(selected_X, save_points=False))
        pt_y_illus = np.array([space.bias.apply(v, p) for v, p in zip(pt_y_true, selected_X)])
        ax2.scatter(selected_X[:, 0], selected_X[:, 1], (pt_y_true - pt_y_illus), color='black', s=20, label="Trainingsdaten")
        ax2.legend()
        
    plt.tight_layout()
    plt.show()

def plot_atslg_landscape(selector, space, resolution=50):
    """
    Erstellt ein 2x3 Gitter (6 Panels). 
    Zeigt Realität, Finale, GPC Unsicherheit, GPR1 Mean, GPR1 Varianz und GPR2 Mean.
    """
    print("\n[Visualisierung] Generiere vollständige Analyse-Plots (6 Panels)...")
    
    bounds = space.dimensions
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    # --- 1. Daten berechnen ---
    y_true = np.array(space.get_values_for_points(grid_points, save_points=False))
    y_illus = np.array([space.bias.apply(v, p) for v, p in zip(y_true, grid_points)])
    Z_true = (y_true - y_illus).reshape(X.shape)
    
    try:
        p1 = selector.gpc.predict_proba(grid_points)[:, 1]
        gpc_uncertainty = p1 * (1 - p1) * 4 
    except:
        p1 = np.zeros(len(grid_points))
        gpc_uncertainty = np.zeros(len(grid_points))
        
    mean1, std1 = selector.gpr1.predict(grid_points, return_std=True) if hasattr(selector.gpr1, "X_train_") else (np.zeros(len(grid_points)), np.ones(len(grid_points)))
    mean2, _ = selector.gpr2.predict(grid_points, return_std=True) if hasattr(selector.gpr2, "X_train_") else (np.zeros(len(grid_points)), np.zeros(len(grid_points)))

    pred_diff = (p1 * mean1) + ((1 - p1) * mean2)
    
    # --- 2. Plotten im 2x3 Grid ---
    fig = plt.figure(figsize=(20, 12)) 
    fig.suptitle("Vollständige ATSLG-Diagnose: Realität, Regressoren & Klassifikator", fontsize=16, fontweight='bold')
    
    # [1] Realität
    ax1 = fig.add_subplot(231, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_true, cmap='Reds', alpha=0.8, edgecolor='none')
    ax1.set_title("1. Echte Realität")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    
    # [2] Finales Modell
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(X, Y, pred_diff.reshape(X.shape), cmap='Blues', alpha=0.8, edgecolor='none')
    ax2.set_title("2. Finales ATSLG-Modell")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
    
    # [3] GPC Unsicherheit
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(X, Y, gpc_uncertainty.reshape(X.shape), cmap='Wistia', alpha=0.8, edgecolor='none')
    ax3.set_title("6. GPC Unsicherheit (Grenzbereich)")
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
    
    # [4] GPR 1 Mean
    ax4 = fig.add_subplot(234, projection='3d')
    surf4 = ax4.plot_surface(X, Y, mean1.reshape(X.shape), cmap='Oranges', alpha=0.8, edgecolor='none')
    ax4.set_title("3. GPR 1 Mean (Fehler-Form)")
    fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10, pad=0.1)
    
    # [5] GPR 1 Varianz
    ax5 = fig.add_subplot(235, projection='3d')
    surf5 = ax5.plot_surface(X, Y, (std1**2).reshape(X.shape), cmap='magma', alpha=0.8, edgecolor='none')
    ax5.set_title("4. GPR 1 Varianz (Unsicherheit)")
    fig.colorbar(surf5, ax=ax5, shrink=0.5, aspect=10, pad=0.1)

    # [6] GPR 2 Mean (Wieder da!)
    ax6 = fig.add_subplot(236, projection='3d')
    surf6 = ax6.plot_surface(X, Y, mean2.reshape(X.shape), cmap='Greens', alpha=0.8, edgecolor='none')
    ax6.set_title("5. GPR 2 Mean (Gesund-Ebene)")
    fig.colorbar(surf6, ax=ax6, shrink=0.5, aspect=10, pad=0.1)

    # Messpunkte einzeichnen
    if len(selector.space.selected_points) > 0:
        selected_X = np.array(selector.space.selected_points)
        pt_y_true = np.array(space.get_values_for_points(selected_X, save_points=False))
        pt_diff = pt_y_true - np.array([space.bias.apply(v, p) for v, p in zip(pt_y_true, selected_X)])
        
        n_init = getattr(selector, 'n_initial', 10)
        ax2.scatter(selected_X[:n_init, 0], selected_X[:n_init, 1], pt_diff[:n_init], color='blue', marker='^', s=45, label="Initial", zorder=10)
        ax2.scatter(selected_X[n_init:, 0], selected_X[n_init:, 1], pt_diff[n_init:], color='black', marker='o', s=20, label="Aktiv", zorder=10)
        ax2.legend()
        
        try:
            labels = selector.gpc.predict(selected_X) 
            ax4.scatter(selected_X[labels == 1, 0], selected_X[labels == 1, 1], pt_diff[labels == 1], color='black', s=15, zorder=10)
            ax6.scatter(selected_X[labels == 0, 0], selected_X[labels == 0, 1], pt_diff[labels == 0], color='black', s=15, zorder=10)
        except:
            pass
            
    plt.tight_layout()
    plt.show()

    # Messpunkte einzeichnen
    if len(selector.space.selected_points) > 0:
        selected_X = np.array(selector.space.selected_points)
        pt_y_true = np.array(space.get_values_for_points(selected_X, save_points=False))
        pt_y_illus = np.array([space.bias.apply(v, p) for v, p in zip(pt_y_true, selected_X)])
        pt_diff = pt_y_true - pt_y_illus
        
        n_init = getattr(selector, 'n_initial', 10)
        X_init = selected_X[:n_init]
        diff_init = pt_diff[:n_init]
        X_active = selected_X[n_init:]
        diff_active = pt_diff[n_init:]

        if len(X_init) > 0:
            ax2.scatter(X_init[:, 0], X_init[:, 1], diff_init, color='blue', marker='^', s=45, label="Initiale Samples", zorder=6)
        if len(X_active) > 0:
            ax2.scatter(X_active[:, 0], X_active[:, 1], diff_active, color='black', marker='o', s=20, label="Aktive Suche", zorder=5)
            
        ax2.legend()
        
        try:
            labels = selector.gpc.predict(selected_X) 
            X_err = selected_X[labels == 1]
            diff_err = pt_diff[labels == 1]
            X_ok = selected_X[labels == 0]
            diff_ok = pt_diff[labels == 0]
            
            if len(X_err) > 0:
                ax3.scatter(X_err[:, 0], X_err[:, 1], diff_err, color='black', s=20, label="Daten GPR 1", zorder=5)
            if len(X_ok) > 0:
                ax4.scatter(X_ok[:, 0], X_ok[:, 1], diff_ok, color='black', s=20, label="Daten GPR 2", zorder=5)
        except:
            pass 
            
    plt.tight_layout()
    plt.show()

def main():
    print("="*50)
    print("START: ALGORITHMUS ENTWICKLUNGS-UMGEBUNG")
    print("="*50)

    # Testraum
    test_file_path = base_path.parent / "criticality_spaces" / "Spaces" / "test_cases" / "2D" / "with_noise" / "test_4.json"
    print(f"[1] Lade Raum: {test_file_path.name}")
    
    functions = load_functions_from_json(test_file_path)
    space = Space(dimensions=[(0, 10), (0, 10)], functions=functions, n_points=101, criticality_thresholds=None)

    # 2. BIAS AKTIVIEREN
    print("[2] Biases aktivieren...")
    bias3 = RegionDependentBias(x_range=(4.0, 9.0), y_range=(4.0, 9.0), drop_factor=0.1)
    
    ultimate_pipeline = BiasChain([bias3])
    space.activate_bias(ultimate_pipeline)

    # 3. SELEKTOR STARTEN
    method = "atslg"
    n_select = 100
    metrics_to_calc = ["general", "extremum_search"]
    
    space.reset_selected_points()
    
    metrics_real, metrics_illu, duration, active_selector = evaluate_single_config(space, method, n_select, metrics_to_calc)

    # 4. EXCEL EXPORT
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
    
    print(f"Maximal gefundene Gefahr (Illusion): {metrics_illu.get('max_criticality_selected', 0):.4f}")
    print(f"Maximal gefundene Gefahr (Realität): {metrics_real.get('max_criticality_selected', 0):.4f}")
    print(f"Sim-to-Reality GAP: {metrics_illu.get('max_criticality_selected', 0) - metrics_real.get('max_criticality_selected', 0):+.4f}")

    # 6. VISUALISIERUNG
    print("\n[6] Öffne 3D-Plots...")
    space.visualizer.plot_3d_two_varied(simulate_bias=False)
    space.visualizer.plot_3d_two_varied(simulate_bias=True)
    
    if method == "atslg":
        plot_atslg_landscape(active_selector, space)
    elif method == "atslgnc":
        plot_standard_gpr_landscape(active_selector, space)
    elif method == "standard_gpr":
        print("Der StandardGPR hat keinen Klassifikator. Zeige nur die Standard-Plots.")

    print("Skript ist fertig. Plot sollte offen sein.")
    input("Drücke ENTER im Terminal, um das Skript und die Plots zu schließen...")

if __name__ == "__main__":
    main()