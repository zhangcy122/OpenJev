from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class ChoiceDecision(BaseModel):
    """Result of a categorical decision over a discrete set of options.
    
    When `abstained` is True, `value` is normalized to 'UNKNOWN' to prevent callers
    from executing uncalibrated actions, while the raw argmax option is preserved in
    `tentative_value`.
    """
    value: str = Field(description="Selected choice value, or 'UNKNOWN' if abstained")
    probabilities: Dict[str, float] = Field(description="Calibrated probability distribution across options")
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated posterior confidence")
    abstained: bool = Field(default=False, description="Whether the model abstained due to uncertainty")
    tentative_value: Optional[str] = Field(default=None, description="Tentative winning option prior to abstention")
    raw_logits: Optional[Dict[str, float]] = None
    escalated: bool = Field(default=False, description="Whether System 2 deliberative reasoning was triggered")
    crystallization_receipt: Optional[Dict[str, Any]] = Field(default=None, description="Receipt detailing exploration and criteria crystallization")

class NoulDecision(BaseModel):
    """Result of a binary truth/assertion judgment."""
    value: bool = Field(description="True or False decision")
    probability_true: float = Field(ge=0.0, le=1.0, description="Calibrated probability of True")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    abstained: bool = Field(default=False, description="Whether the judgment was abstained")

class ScoreDecision(BaseModel):
    """Result of an ordinal rating/score evaluation."""
    expected_score: float = Field(description="Expected score based on probability distribution")
    level_probabilities: Dict[str, float] = Field(description="Probability assigned to each score tier")
    confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence")
    abstained: bool = Field(default=False)
