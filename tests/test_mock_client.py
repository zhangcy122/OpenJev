import itertools
import pytest
from openjevpro import OpenJevProClient, MockClient
from openjevpro.schemas import ChoiceDecision, NoulDecision, ScoreDecision

def test_mock_client_initialization():
    client1 = MockClient()
    assert client1.model == "mock-simulator-v1"

    client2 = OpenJevProClient(mock=True)
    assert client2.mock is True
    assert client2._mock_client is not None

    client3 = OpenJevProClient(base_url="mock://localhost")
    assert client3.mock is True

def test_mock_choice_decision_schema():
    client = MockClient()
    state = {"ticket": "Detection of unauthorized high-volume login attempt from unknown IP."}
    candidates = ["billing", "tech_support", "security_fraud", "human_review"]

    decision = client.decide_choice(state=state, candidates=candidates, order_invariant=True)

    assert isinstance(decision, ChoiceDecision)
    assert decision.value == "security_fraud"
    assert decision.confidence > 0.80
    assert not decision.abstained
    assert sum(decision.probabilities.values()) == pytest.approx(1.0, rel=1e-3)
    assert "security_fraud" in decision.probabilities

def test_mock_client_strict_permutation_invariance():
    client = MockClient()
    state = {"ticket": "Database connection timeout and memory leak in worker pool."}
    candidates = ["billing", "tech_support", "security_fraud", "human_review"]

    # Test across all 24 permutations of the 4 candidates
    winners = []
    prob_records = []

    for perm in itertools.permutations(candidates):
        res = client.decide_choice(state=state, candidates=list(perm), order_invariant=True)
        winners.append(res.value)
        prob_records.append(res.probabilities[res.value])

    # 1. Winning candidate MUST be 100% identical across all 24 permutations
    assert len(set(winners)) == 1
    assert winners[0] == "tech_support"

    # 2. Maximum probability variance across permutations MUST be < 1e-4
    max_p = max(prob_records)
    min_p = min(prob_records)
    assert (max_p - min_p) < 1e-4

def test_mock_client_abstention():
    # Abstain threshold set high
    client = MockClient(abstain_threshold=0.99)
    state = {"ticket": "Random gibberish xyz abc 123"}
    candidates = ["billing", "tech_support", "security_fraud"]

    decision = client.decide_choice(state=state, candidates=candidates, allow_abstain=True)
    assert decision.abstained is True
    assert decision.value == "UNKNOWN"

def test_mock_noul_decision():
    client = OpenJevProClient(mock=True)
    state = {"ticket": "Server crashed with out-of-memory error"}
    noul = client.decide_noul(state=state, assertion="This is an urgent technical outage.")

    assert isinstance(noul, NoulDecision)
    assert isinstance(noul.value, bool)
    assert 0.0 <= noul.confidence <= 1.0

def test_mock_score_decision():
    client = OpenJevProClient(mock=True)
    state = {"ticket": "Routine invoice query regarding payment schedule."}
    score = client.decide_score(state=state, criteria="Assess severity level.")

    assert isinstance(score, ScoreDecision)
    assert 0.0 <= score.expected_score <= 1.0
    assert "LOW" in score.level_probabilities
