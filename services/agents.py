import os
from services.azure_auth import get_chat_response
from services.state import AgentState
from simple_salesforce import Salesforce

# Initialize the client
os.environ["SF_USERNAME"] = "USERNAME"
os.environ["SF_PASSWORD"] = "PASSWORD"
os.environ["TOKEN"] = "TOKEN"

def generate_queries(state_request: str) -> list:
    sf_prompt = f"""
        You are a Salesforce SOQL assistant.
        Convert a user request into a valid SOQL query. Query only by Company or Person.
        Show the number of records in Accounts, Contacts, and Opportunities found.
        From Accounts query Name and Industry fields,
        From Contacts query First Name, Last Name, and Email of the first contact associated with the account,
        From Opportunities query only recently created records with the following fields: Name, StageName, CloseDate, Amount, and AccountId.

        If the request is about a person, query the Contact object.
        Use the following format:
        SELECT <fields> FROM <object> WHERE <conditions>
        Ensure the queries are valid and does not include any unnecessary fields or objects.
        Response must be in SELECT <fields> FROM <object> WHERE <conditions> format. Do not include any other text or characters.

        User request: "{state_request}"
    """
    queries = get_chat_response("user", sf_prompt).split("\n")
    queries = [query.strip() for query in queries if query.startswith("SELECT")]

    return queries


def get_sf_data(queries: list):
    print("Getting Salesforce query results")

    sf = Salesforce(
        username=os.getenv("SF_USERNAME"),
        password=os.getenv("SF_PASSWORD"),
        security_token=os.getenv("TOKEN"),
        domain="login"
    )

    results = []
    for query in queries:
        try:
            result = sf.query_all(query)["records"]
            results.append(result)
            print(f"Result: {results}")
        except Exception as e:
            print(f"Error executing query '{query}': {e}")
            results.append({"error": str(e)})

    return results

def sf_agent(state: AgentState) -> dict:
    print("SF Agent is running")

    soql_queries = generate_queries(state.request)
    print(f"Generated SOQL queries: {soql_queries}")

    result_text = get_sf_data(soql_queries)
    return {"sf_response": result_text}

def web_agent(state: AgentState) -> dict:
    print("Web Agent is running")
    print(f"State request: {state.request}")

    web_prompt = f"""
                You are a web agent.
                Find the latest information about the person or company mentioned in the user request.
                If the request is about a person, find their net worth.
                If the request is about a company, find their quarterly financial results in the last 2 years.
                If the request is about both, find both pieces of information.

                User request: "{state.request}"
            """

    result = get_chat_response("user", web_prompt)
    if not result:
        result = "Could not retrieve a response"

    return {"web_response": result}

def aggregator(state: AgentState) -> dict:
    print("Aggregator is running")
    print(f"State: {state}")

    aggregator_prompt = f"""
            You are an aggregator agent.
            Combine the responses from the Salesforce and Web agents.
            Suggest if the person can afford IT consulting services based on Salesforce Opportunities amount.
            If the Salesforce agent did not run, only return the web response.

            Salesforce response: "{state.sf_response if state.sf_response else 'No response from SF Agent'}"
            Web response: "{state.web_response if state.web_response else 'No response from Web Agent'}"

            Tell me your opinion if he can afford my IT consulting services.
    """

    result = get_chat_response("user", aggregator_prompt)
    print(f"Aggregator result: {result}")

    return {"final_output": result}
