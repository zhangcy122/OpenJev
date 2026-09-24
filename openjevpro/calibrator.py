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

    def fit(self, logits_list: list[Dict[str, float]], targets: list[str], min_t: float = 0.1, max_t: float = 10.0, tol: float = 1e-5) -> float:
        """Fits temperature T by minimizing negative log-likelihood (NLL) over validation logits and labels.

        Uses bounded 1D golden-section search to find optimal T in [min_t, max_t].
        """
        if not logits_list or len(logits_list) < 2 or len(logits_list) != len(targets):
            raise ValueError("Insufficient validation samples to fit temperature (at least 2 matching pairs required).")
        
        def nll_obj(T_val: float) -> float:
            total_nll = 0.0
            valid_count = 0
            for l_dict, tgt in zip(logits_list, targets):
                if not l_dict or tgt not in l_dict:
                    continue
                max_l = max(l_dict.values())
                exps = {k: math.exp((v - max_l) / T_val) for k, v in l_dict.items()}
                s = sum(exps.values())
                if s <= 0:
                    continue
                p_tgt = exps[tgt] / s
                total_nll -= math.log(max(p_tgt, 1e-12))
                valid_count += 1
            if valid_count == 0:
                return float("inf")
            return total_nll / valid_count

        # Golden-section search on [min_t, max_t]
        a, b = min_t, max_t
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        while abs(b - a) > tol:
            if nll_obj(c) < nll_obj(d):
                b = d
            else:
                a = c
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
        optimal_t = round((b + a) / 2, 3)
        self.temperature = optimal_t
        return self.temperature

