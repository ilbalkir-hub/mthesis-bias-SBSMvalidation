
#
# imports
#

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