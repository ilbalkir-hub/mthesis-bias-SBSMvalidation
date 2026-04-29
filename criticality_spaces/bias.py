import math

class GlobalBias:
    def __init__(self, offset=0.2):
        self.offset = offset
        self.name = "GlobalBias"

    def apply(self, value, point):
        """Hebt oder senkt die Kritikalität überall gleichmäßig an. Standart Anhebung um 0,2"""
        return max(0.0, min(1.0, value + self.offset))

class RegionBias:
    def __init__(self, threshold_x=5.0, multiplier=0.5):
        self.threshold_x = threshold_x
        self.multiplier = multiplier
        self.name = "RegionBias"

    def apply(self, value, point):
        """Verändert die Kritikalität nur in einer bestimmten Region (z.B. X > 5.0)."""
        if point[0] > self.threshold_x:
            return value * self.multiplier
        return value

class CalibrationBias:
    def __init__(self, offset=0.0, scale=1):
        self.offset = offset
        self.scale = scale
        self.name = "CalibrationBias"

    def apply(self, value, point):
        """Staucht oder streckt die Werte und verschiebt sie."""
        # Beispiel: value * 0.8 dämpft alle Signale ab (Sensorschwäche)
        new_value = (value * self.scale) + self.offset
        return max(0.0, min(1.0, new_value))

class LocalStructuralBias:
    def __init__(self, amplitude=0.2, frequency=5.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.name = "LocalStructuralBias"

    def apply(self, value, point):
        """Legt ein künstliches, wellenartiges Rauschen über den Raum."""
        # Erzeugt ein künstliches "Wabenmuster" aus Störsignalen
        noise = math.sin(point[0] * self.frequency) * math.cos(point[1] * self.frequency) * self.amplitude
        return max(0.0, min(1.0, value + noise))
    
class TopologyBoundaryBias:
    def __init__(self, boundary_margin=2.25, penalty=0.4):
        self.margin = boundary_margin
        self.penalty = penalty
        self.name = "TopologyBoundaryBias"

    def apply(self, value, point):
        """Zieht an den Rändern des Raumes (0 bis 10) drastisch Kritikalität ab."""
        # Wenn wir näher als 1.5 Einheiten am Rand sind (X oder Y Achse)
        if point[0] < self.margin or point[0] > (10.0 - self.margin) or \
           point[1] < self.margin or point[1] > (10.0 - self.margin):
            return max(0.0, min(1.0, value - self.penalty))
        return value

class RegionDependentBias:
    def __init__(self, x_range=(4.0, 7.0), y_range=(2.0, 5.0), drop_factor=0.2):
        self.x_range = x_range
        self.y_range = y_range
        self.drop_factor = drop_factor
        self.name = "RegionDependentBias"

    def apply(self, value, point):
        """Wenn der Punkt in der definierten Box liegt, wird das Signal gedämpft."""
        if self.x_range[0] <= point[0] <= self.x_range[1] and \
           self.y_range[0] <= point[1] <= self.y_range[1]:
            # Die Gefahr wird künstlich klein gerechnet (Blindflug)
            return value * self.drop_factor
        return value
    
class LocalizationBias:
    def __init__(self, space, x_shift=0.5, y_shift=0.0):
        self.space = space
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.name = "LocalizationBias"

    def apply(self, value, point):
        """
        Das UUV denkt, es ist bei 'point', aber durch die Strömung 
        ist es in Wirklichkeit bei 'shifted_point'.
        """
        # Echte Position
        shifted_x = point[0] + self.x_shift
        shifted_y = point[1] + self.y_shift
        
        # Sicherstellen, das wir im Raum bleiben
        shifted_x = max(0.0, min(10.0, shifted_x))
        shifted_y = max(0.0, min(10.0, shifted_y))
        
        # 2. Wir fragen den Raum heimlich: "Was ist wirklich an dieser verschobenen Stelle?"
        # WICHTIG: save_points=False, damit wir diesen heimlichen Blick nicht im Algorithmus protokollieren!
        real_value_at_shifted_pos = self.space.get_values_for_points([(shifted_x, shifted_y)], save_points=False)[0]
        
        # 3. Wir geben diesen falschen Wert an den ahnungslosen Selektor zurück
        return real_value_at_shifted_pos

class BiasChain:
    def __init__(self, biases):
        """
        Nimmt eine Liste von Bias-Objekten entgegen und wendet sie nacheinander an.
        """
        self.biases = biases
        
        # Baut einen dynamischen Namen aus allen übergebenen Biases
        bias_names = [b.name for b in biases]
        self.name = "Pipeline(" + " -> ".join(bias_names) + ")"

    def apply(self, value, point):
        """
        Reicht den Wert wie bei einer stillen Post von einem Bias zum nächsten.
        """
        current_value = value
        
        for bias in self.biases:
            current_value = bias.apply(current_value, point)
            
        return current_value