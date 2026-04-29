"""
Implements an Importance Sampling (IS) Selector as a statistical baseline.
Samples points proportional to their 'Illusion' (biased sensor readings) 
and evaluates them against the Ground Truth.
"""

import numpy as np
from base import BaseSelector

class ImportanceSamplingSelector(BaseSelector):
    def __init__(self, space, n_select: int, pool_size: int = 10000):
        """
        Parameters
        ----------
        space : Space
            Der Suchraum.
        n_select : int
            Das Budget (Wie viele echte Tauchgänge darf das UUV machen?).
        pool_size : int, optional
            
        """
        super().__init__(space, n_select)
        self.pool_size = pool_size

    def select(self) -> list[tuple]:
        bounds = self.space.dimensions
        d = len(bounds)

        # ==========================================
        # SCHRITT 1: Fiktives Raster auswerfen
        # ==========================================
        # Wir werfen 10.000 zufällige Punkte aus
        candidates = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.pool_size, d)
        )

        # ==========================================
        # SCHRITT 2: Die Illusion befragen
        # ==========================================
        # Zuerst holen wir heimlich die echten Werte, OHNE sie als "gesucht" zu protokollieren
        raw_values = self.space.get_values_for_points(candidates, save_points=False)

        # Jetzt jagen wir sie durch die Bias-Pipeline. Das ist das, was das UUV "sieht".
        if getattr(self.space, "bias", None) is not None:
            illusion_values = np.array([
                self.space.bias.apply(val, pt) for val, pt in zip(raw_values, candidates)
            ])
        else:
            illusion_values = np.array(raw_values)

        # ==========================================
        # SCHRITT 3: Wahrscheinlichkeiten berechnen
        # ==========================================
        # Wir wollen dort am häufigsten suchen, wo die Illusion am gefährlichsten aussieht.
        # Wir addieren ein winziges Epsilon (1e-8), damit wir nicht durch Null teilen, 
        # falls die Illusion überall 0 anzeigt.
        probs = illusion_values + 1e-8
        probs = probs / np.sum(probs) # Normalisieren, damit alle Werte zusammen 1.0 ergeben

        # ==========================================
        # SCHRITT 4: Das "Sampling" (Das Ziehen aus der Urne)
        # ==========================================
        # Wir ziehen nun n_select Punkte aus den 10.000 Kandidaten.
        # Ein Punkt mit 0.9 Gefahr hat eine viel höhere Chance gezogen zu werden als einer mit 0.1.
        selected_indices = np.random.choice(
            self.pool_size,
            size=self.n_select,
            replace=False, # replace=False bedeutet: Keinen Punkt zweimal absuchen!
            p=probs
        )
        selected_candidates = candidates[selected_indices]

        # ==========================================
        # SCHRITT 5: Den echten Tauchgang durchführen
        # ==========================================
        # Jetzt sagen wir dem Schiedsrichter offiziell: "Wir tauchen hier!"
        # save_points=True trägt die Punkte in die Metriken ein.
        self.space.get_values_for_points(selected_candidates, save_points=True)

        # Formatieren für das Rückgabe-Format des Frameworks
        selected_points = [
            tuple(np.round(p, self.space.decimal_precision))
            for p in selected_candidates
        ]

        return selected_points