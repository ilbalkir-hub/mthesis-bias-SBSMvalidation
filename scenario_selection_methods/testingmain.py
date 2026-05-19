from pathlib import Path
import sys
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

# Pfade setzen
base_path = Path(__file__).parent
cs_path = (base_path.parent / "criticality_spaces").resolve()
if cs_path not in sys.path:
    sys.path.insert(0, str(cs_path))

from functionloader import load_functions_from_json
from space import Space
from factory import get_selector
from bias import RegionDependentBias, CalibrationBias, BiasChain, AxisGradientBias,FlatRegionBias

def evaluate_single_config(space, method_name, n_select, eval_metrics):
    """Führt einen Selektor aus und vergleicht Illusion mit Realität."""
    print(f" Starte Selektor: {method_name}")
    selector = get_selector(method_name, space, n_select)

    safe_metrics = eval_metrics.copy() 
    if not hasattr(selector, "get_model") and "model_reconstruction" in safe_metrics:
        safe_metrics.remove("model_reconstruction")
        print(f"   ⚠️ Warnung: '{method_name}' hat kein Modell. 'model_reconstruction' übersprungen!")

    start_time = time.perf_counter()
    
    if hasattr(selector, "select"):
        selector.select()
        metrics_real = space.metrics.run_metrics_suite(method_categories=safe_metrics)
    elif hasattr(selector, "get_model"):
        model = selector.get_model()
        metrics_real = space.metrics.run_metrics_suite(method_categories=safe_metrics, surrogate_model=model)
    else:
        raise NotImplementedError("Selektor muss 'select()' oder 'get_model()' haben.")
    
    duration = time.perf_counter() - start_time

    metrics_illusion = metrics_real.copy()
    if getattr(space, "bias", None) is not None and len(space.selected_points) > 0:
        raw_values = space.get_values_for_points(space.selected_points, save_points=False)
        perceived_values = [space.bias.apply(val, pt) for val, pt in zip(raw_values, space.selected_points)]
        
        metrics_illusion["average_criticality_selected"] = sum(perceived_values) / len(perceived_values)
        metrics_illusion["max_criticality_selected"] = max(perceived_values)
        metrics_illusion["min_criticality_selected"] = min(perceived_values)

    return metrics_real, metrics_illusion, duration, selector

def plot_standard_gpr_landscape(selector, space, resolution=50):
    """Ablationsstudie: 2-Panel Plot."""
    print("\n[Visualisierung] Generiere Fehler-Landschaft für Standard-GPR (2 Panels)...")
    
    bounds = space.dimensions
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    y_true = np.array(space.get_values_for_points(grid_points, save_points=False))
    y_illus = np.array([space.bias.apply(v, p) for v, p in zip(y_true, grid_points)])
    Z_true = (y_true - y_illus).reshape(X.shape)
    
    if hasattr(selector.gpr, "X_train_"):
        mean, _ = selector.gpr.predict(grid_points, return_std=True)
    else:
        mean = np.zeros(len(grid_points))
    Z_pred = mean.reshape(X.shape)
    
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Ablationsstudie: Standard GPR-Verhalten ohne Klassifikator", fontsize=14, fontweight='bold')
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_true, cmap='Reds', alpha=0.8, edgecolor='none')
    ax1.set_title("Echte Realität (Wahrer Fehler)")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_pred, cmap='Oranges', alpha=0.8, edgecolor='none')
    ax2.set_title("Gelerntes Modell (Einziger GPR)")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

    if len(selector.space.selected_points) > 0:
        selected_X = np.array(selector.space.selected_points)
        pt_y_true = np.array(space.get_values_for_points(selected_X, save_points=False))
        pt_y_illus = np.array([space.bias.apply(v, p) for v, p in zip(pt_y_true, selected_X)])
        ax2.scatter(selected_X[:, 0], selected_X[:, 1], (pt_y_true - pt_y_illus), color='black', s=20, label="Trainingsdaten")
        ax2.legend()
        
    plt.tight_layout()
    plt.show()

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

def plot_atslg_landscape(selector, space, resolution=50):
    """
    Erstellt ein interaktives 2x5 Gitter (10 Panels) im Browser via Plotly.
    Inklusive aller Samples (Initial & Aktiv).
    """
    print("\n[Visualisierung] Generiere interaktive HTML-Datei inkl. Samples (10 Panels)...")
    
    bounds = space.dimensions
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    # --- 1. Echte Daten berechnen ---
    y_true = np.array(space.get_values_for_points(grid_points, save_points=False))
    y_illus = np.array([space.bias.apply(v, p) for v, p in zip(y_true, grid_points)])
    Z_true = (y_true - y_illus).reshape(X.shape)
    
    # --- 2. Modelle abfragen ---
    try:
        p1 = selector.gpc.predict_proba(grid_points)[:, 1]
        p2 = 1.0 - p1
        gpc_uncertainty_raw = p1 * p2
        gpc_uncertainty_plot = gpc_uncertainty_raw * 4 
    except:
        p1 = np.zeros(len(grid_points))
        p2 = np.ones(len(grid_points))
        gpc_uncertainty_raw = np.zeros(len(grid_points))
        gpc_uncertainty_plot = np.zeros(len(grid_points))
        
    if hasattr(selector.gpr1, "X_train_"):
        mean1, std1 = selector.gpr1.predict(grid_points, return_std=True)
    else:
        mean1, std1 = np.zeros(len(grid_points)), np.ones(len(grid_points))
        
    if hasattr(selector.gpr2, "X_train_"):
        mean2, std2 = selector.gpr2.predict(grid_points, return_std=True)
    else:
        mean2, std2 = np.zeros(len(grid_points)), np.ones(len(grid_points))

    # --- 3. Mathematik der Acquisition Function ---
    pred_diff = (p1 * mean1) + (p2 * mean2) 
    
    E1 = (mean1 ** 2) + (std1 ** 2)
    E2 = (mean2 ** 2) + (std2 ** 2)
    EI = (p1 * E1) + (p2 * E2)
    
    U_E = np.max(EI) if np.max(EI) > 0 else 1.0
    U_C = np.max(gpc_uncertainty_raw) if np.max(gpc_uncertainty_raw) > 0 else 1.0
    
    w = 0.5 # Echte 50/50 Balance
    af_scores = w * (EI / U_E) + (1 - w) * (gpc_uncertainty_raw / U_C)

    # --- 4. Plotly Interaktive Figure erstellen ---
    fig = make_subplots(
        rows=2, cols=5,
        specs=[[{'is_3d': True}] * 5, [{'is_3d': True}] * 5],
        subplot_titles=(
            "1. Realität", "2. Finales Modell", "3. P1 (Fehler)", "4. P2 (Gesund)", "5. GPC Unsicherheit",
            "6. Acquisition Function", "7. GPR 1 Mean", "8. GPR 1 Var", "9. GPR 2 Mean", "10. GPR 2 Var"
        ),
        horizontal_spacing=0.02,
        vertical_spacing=0.05
    )

    # Hilfsfunktion zum Hinzufügen der Oberflächen
    def add_surface(z_data, colorscale, row, col):
        fig.add_trace(go.Surface(x=x, y=y, z=z_data.reshape(X.shape), colorscale=colorscale, showscale=False), row=row, col=col)

    # Flächen zeichnen
    add_surface(Z_true, 'reds', 1, 1)
    add_surface(pred_diff, 'blues', 1, 2)
    add_surface(p1, 'purples', 1, 3)
    add_surface(p2, 'ylgn', 1, 4)
    add_surface(gpc_uncertainty_plot, 'solar', 1, 5) 

    add_surface(af_scores, 'plasma', 2, 1)
    add_surface(mean1, 'oranges', 2, 2)
    add_surface(std1**2, 'inferno', 2, 3)
    add_surface(mean2, 'greens', 2, 4)
    add_surface(std2**2, 'viridis', 2, 5)

    # --- 5. SAMPLES (PUNKTE) HINZUFÜGEN ---
    if len(selector.space.selected_points) > 0:
        selected_X = np.array(selector.space.selected_points)
        pt_y_true = np.array(space.get_values_for_points(selected_X, save_points=False))
        pt_diff = pt_y_true - np.array([space.bias.apply(v, p) for v, p in zip(pt_y_true, selected_X)])
        
        n_init = getattr(selector, 'n_initial', 10)
        
        # Panel 2: Finales Modell (Initial vs. Aktiv)
        # Initiale Punkte (Blau, Diamanten)
        fig.add_trace(go.Scatter3d(
            x=selected_X[:n_init, 0], y=selected_X[:n_init, 1], z=pt_diff[:n_init],
            mode='markers', marker=dict(size=6, color='blue', symbol='diamond'),
            name="Initiale Samples"
        ), row=1, col=2)
        
        # Aktive Punkte (Schwarz, Kreise)
        fig.add_trace(go.Scatter3d(
            x=selected_X[n_init:, 0], y=selected_X[n_init:, 1], z=pt_diff[n_init:],
            mode='markers', marker=dict(size=4, color='black', symbol='circle'),
            name="Aktive Samples"
        ), row=1, col=2)

        # Panel 6: Acquisition Function (Alle aktiven Punkte oben auf die Decke projiziert)
        max_af = np.max(af_scores) if np.max(af_scores) > 0 else 1.0
        fig.add_trace(go.Scatter3d(
            x=selected_X[n_init:, 0], y=selected_X[n_init:, 1], z=np.full(len(selected_X)-n_init, max_af),
            mode='markers', marker=dict(size=4, color='black', symbol='circle'),
            name="Gezogene Punkte (AF)"
        ), row=2, col=1)
        
        # Panel 7 & 9: GPR 1 und GPR 2 Split
        try:
            labels = selector.gpc.predict(selected_X) 
            # GPR 1 (Fehlerbereich)
            mask_1 = labels == 1
            if np.any(mask_1):
                fig.add_trace(go.Scatter3d(
                    x=selected_X[mask_1, 0], y=selected_X[mask_1, 1], z=pt_diff[mask_1],
                    mode='markers', marker=dict(size=3, color='black'), showlegend=False
                ), row=2, col=2)
                
            # GPR 2 (Gesunder Bereich)
            mask_0 = labels == 0
            if np.any(mask_0):
                fig.add_trace(go.Scatter3d(
                    x=selected_X[mask_0, 0], y=selected_X[mask_0, 1], z=pt_diff[mask_0],
                    mode='markers', marker=dict(size=3, color='black'), showlegend=False
                ), row=2, col=4)
        except:
            pass

    # --- 6. Layout & Export ---
    fig.update_layout(
        title_text="Vollständige ATSLG-Architektur (Interaktiv)",
        title_x=0.5, height=900, width=2200, margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False # Legende ausblenden für mehr Platz
    )

    # Kamera-Winkel
    camera = dict(eye=dict(x=1.5, y=1.5, z=0.5))
    for i in range(1, 11):
        fig.update_layout(**{f'scene{i}_camera': camera})

    html_file = "atslg_interaktiv.html"
    fig.write_html(html_file)
    print(f"[Visualisierung] Fertig! Öffne {html_file} im Browser...")
    
    webbrowser.open('file://' + os.path.realpath(html_file))

def main():
    print("="*50)
    print("START: ALGORITHMUS ENTWICKLUNGS-UMGEBUNG")
    print("="*50)

    # Testraum
    test_file_path = base_path.parent / "criticality_spaces" / "Spaces" / "test_cases" / "2D" / "without_noise" / "test_1.json"
    print(f"[1] Lade Raum: {test_file_path.name}")
    
    functions = load_functions_from_json(test_file_path)
    space = Space(dimensions=[(0, 10), (0, 10)], functions=functions, n_points=101, criticality_thresholds=None)

    # 2. BIAS AKTIVIEREN
    print("[2] Biases aktivieren...")
    bias3 = FlatRegionBias(x_range=(4.0, 9.0), y_range=(4.0, 9.0), offset=0.2)
    ultimate_pipeline = BiasChain([bias3])
    space.activate_bias(ultimate_pipeline)

    # 3. SELEKTOR STARTEN
    method = "atslg"
    n_select = 15
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

    print("Skript ist fertig. Plot sollte offen sein.")
    input("Drücke ENTER im Terminal, um das Skript und die Plots zu schließen...")

if __name__ == "__main__":
    main()