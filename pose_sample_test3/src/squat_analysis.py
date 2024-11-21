import numpy as np
from dataclasses import dataclass

@dataclass
class SquatStandards:
    STANDING = {
        'hip_angle': 172,
        'knee_angle': 175,
        'ankle_angle': 80,
        'tolerance': 5
    }
    
    PARALLEL = {
        'hip_angle': 95,
        'knee_angle': 90,
        'ankle_angle': 70,
        'tolerance': 8
    }

class SquatAnalysisEngine:
    def __init__(self):
        self.standards = SquatStandards()
        self.rep_count = 0
        self.current_phase = "STANDING"
    
    def analyze_frame(self, landmarks):
        # 구현 예정
        pass
