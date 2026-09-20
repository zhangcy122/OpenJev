from openjevpro.schemas import ChoiceDecision
from openjevpro.client import OpenJevProClient

class IntentRoute(StrEnum):
    SEARCH = "web_search"
    EXECUTE_CODE = "code_execution"
    ANSWER_DIRECTLY = "answer_directly"
    REQUEST_CLARIFICATION = "request_clarification"

def main():
    print("OpenJevPro Demo: Routing Workflow")
    client = OpenJevProClient(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-4B-Instruct",
        temperature_scaling=1.35,
        abstain_threshold=0.40
    )

    state = {
        "user_query": "What was Apple's total revenue reported in their latest 10-K filing?",
        "session_history": [],
        "available_tools": ["search", "calculator", "python_interpreter"]
    }

    print("State:", state)
    print("Evaluating decision across candidate routes...")
    # In live environments with running vLLM, client.decide_choice returns calibrated decision.

if __name__ == "__main__":
    main()
