# Design: TypeSafe Jev Runtime Calibration & Adaptive Safety Guard Harness

## Context

OpenJevPro uses `DecisionEngine` for constrained classification and routing, backed by `TemperatureCalibrator` for logit temperature scaling. When operating with neural engines (e.g. TypeSafe Jev / Gemma 4), probability outputs can be overconfident or near-boundary. A formal safety harness is needed to wrap any underlying predictor or probability dict and enforce conservative decision boundaries.

## Architecture & Math

### 1. Pseudo-Logit Inversion
Given normalized probabilities $\mathbf{p} = [p_1, \dots, p_K]$ where $\sum p_i = 1$:
$$z_i = \ln(\max(p_i, \epsilon))$$
with numerical stabilizer $\epsilon = 10^{-6}$.

### 2. Temperature Calibration
Apply `TemperatureCalibrator.calibrate(z)`:
$$p_i^{\text{cal}} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
where default $T = 1.25$ or fitted via NLL minimization.

### 3. Adaptive Dual-Threshold Abstention
Define threshold $\tau$:
$$\tau = \max\left(\tau_{\min}, \frac{\alpha}{K}\right)$$
- $\tau_{\min}$: Absolute floor confidence (default 0.72).
- $\alpha$: Uniform prior factor (default 1.25).
- $K$: Number of classes ($K \ge 1$).

If $\max_i(p_i^{\text{cal}}) < \tau$:
- `choice = "UNKNOWN"`
- `is_abstained = True`
- `tentative_choice = \arg\max_i(p_i^{\text{cal}})`
- `confidence = \max_i(p_i^{\text{cal}})`

Otherwise:
- `choice = \arg\max_i(p_i^{\text{cal}})`
- `is_abstained = False`
- `tentative_choice = choice`
- `confidence = \max_i(p_i^{\text{cal}})`

## Data Contract

```python
@dataclass
class GuardDecision:
    choice: str
    confidence: float
    is_abstained: bool
    tentative_choice: str
    threshold_used: float
    calibrated_probs: Dict[str, float]
    raw_probs: Dict[str, float]
```

## Resilience & Edge Cases
1. Empty classes ($K = 0$): returns `UNKNOWN` immediately with confidence 0.0.
2. Single class ($K = 1$): threshold evaluates to $\max(\tau_{\min}, \alpha)$; calibrated prob 1.0 passes if $1.0 \ge \tau$.
3. Degenerate / zero probabilities: clipped by $\epsilon = 10^{-6}$ avoiding $-\infty$ or NaN.
4. Direct probability map input or delegate engine input support.
