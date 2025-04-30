from services.azure_auth import get_chat_response
from services.state import AgentState

def orchestrator(state: AgentState) -> dict:
    print("Starting Orchestrator")
    print(f"State is {state}")

    user_request = state.request
    if not user_request:
        raise ValueError("State must include a 'request' key with user input.")

    routing_prompt = f"""
        You are a smart orchestrator.
        The Web Search agent must always run.

        Decide only if the Salesforce agent is needed.
        Use the Salesforce agent if the request involves:
        - CRM data
        - Customer accounts
        - Sales records
        - Company profiles from internal systems

        User request: "{state.request}"

        Respond ONLY with:
        - "Use Salesforce" if Salesforce agent is needed
        - "Skip Salesforce" if not needed"
    """
    result = get_chat_response("system", routing_prompt)
    action = result.strip()
    if "Use Salesforce" in result:
        action = "sf_agent"

    print(f"Orchestrator Decision: {action}")

    return {"next": action}