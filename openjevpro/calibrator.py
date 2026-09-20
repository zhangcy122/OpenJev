import math
from typing import Dict

class TemperatureCalibrator:
    """Scales logits by temperature T to fix LLM overconfidence and calibrate probabilities."""
    
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature

    def calibrate(self, logits: Dict[str, float]) -> Dict[str, float]:
        """Applies softmax with temperature scaling."""
        if not logits:
            return {}
        
        # Numerical stability: subtract max
        max_logit = max(logits.values())
        scaled_exp = {k: math.exp((v - max_logit) / self.temperature) for k, v in logits.items()}
        total = sum(scaled_exp.values())
        
        if total == 0:
            uniform = 1.0 / len(logits)
            return {k: uniform for k in logits}
            
        return {k: v / total for k, v in scaled_exp.items()}
