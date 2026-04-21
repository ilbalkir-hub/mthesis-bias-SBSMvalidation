import sys
from pathlib import Path

# --- INTELLIGENTE PFAD-FINDUNG ---
current_path = Path(__file__).resolve().parent

# Wir suchen den Ordner, der 'criticality_spaces' enthält
# Wir prüfen den aktuellen Ordner und den darüber
if (current_path / "criticality_spaces").exists():
    project_root = current_path
elif (current_path.parent / "criticality_spaces").exists():
    project_root = current_path.parent
else:
    print("❌ Kritischer Fehler: Projektordner 'criticality_spaces' nicht gefunden!")
    sys.exit(1)

# Jetzt die Pfade korrekt hinzufügen
sys.path.insert(0, str(project_root / "criticality_spaces"))
sys.path.insert(0, str(project_root / "scenario_selection_methods"))

print(f"✅ Suche Module in: {project_root}")

try:
    from space import Space
    from functionloader import load_functions_from_json
    from factory import get_selector
    print("✅ Erfolg: Alle Module geladen!")
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    # Zeige uns, was Python aktuell sieht, falls es immer noch klemmt:
    print(f"Aktueller Suchpfad (sys.path): {sys.path[:2]}")
    sys.exit(1)

def main():
    # --- KONFIGURATION ---
    # Pfad zur JSON-Datei (Relativ zum Hauptordner)
    json_path = project_root / "criticality_spaces" / "Spaces" / "generated_spaces" / "2D" / "6_2D.json"
    
    if not json_path.exists():
        print(f"❌ Fehler: Die Datei {json_path} wurde nicht gefunden!")
        return

    # --- INITIALISIERUNG ---
    print(f"Lade Test-Raum: {json_path.name}...")
    fns = load_functions_from_json(json_path)
    space = Space(
        dimensions=2, 
        functions=fns, 
        n_points=200, 
        criticality_thresholds=[0.3, 0.9]
    )

    # --- SELEKTOR AUSFÜHREN ---
    # Wir nehmen GPR mit 50 Samples als Baseline
    n_samples = 50
    print(f"Starte Selektor 'gpr' mit {n_samples} Samples...")
    
    selector = get_selector("gpr", space=space, n_select=n_samples)
    selector.select()

    # --- AUSWERTUNG ---
    print("\n" + "="*40)
    print("      ERGEBNISSE DER SELEKTION")
    print("="*40)
    
    results = space.metrics.run_metrics_suite(
        method_categories=["general", "boundary_detection"]
    )
    
    for metric, value in results.items():
        print(f"{metric:30}: {value}")
    print("="*40)

if __name__ == "__main__":
    main()