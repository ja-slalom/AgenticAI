from IPython.core.display import Image
from IPython.core.display_functions import display
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from services.agents import sf_agent, web_agent, aggregator
from services.orchestrator import orchestrator
from services.state import AgentState


def build_graph():
    graph_builder = StateGraph(AgentState)

    # Graph Node
    graph_builder.add_node("orchestrator", orchestrator)
    graph_builder.add_node("sf_agent", sf_agent)
    graph_builder.add_node("web_agent", web_agent)
    graph_builder.add_node("aggregator", aggregator)

    # Graph Edges
    graph_builder.add_edge(START, "orchestrator")
    graph_builder.add_conditional_edges("orchestrator", orchestrator, {"sf_agent", END})
    # graph_builder.add_edge("orchestrator", "sf_agent")
    graph_builder.add_edge("orchestrator", "web_agent")
    graph_builder.add_edge("sf_agent", "aggregator")
    graph_builder.add_edge("web_agent", "aggregator")
    graph_builder.add_edge("aggregator", END)

    # Compile Graph
    graph = graph_builder.compile()


    print("\n=== Workflow Graph Structure ===")
    graph.get_graph().print_ascii()


    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"Could not display image: {e}")

    return graph

if __name__ == "__main__":
    graph = build_graph()

    # input_prompt = {"request": "Find the customer account details and recent market news for Tesla."}
    # input_prompt = {"request": "Tell me more about Elon Musk and his net worth."}
    input_prompt = {"request": "What is the USA?"}

    result = graph.invoke(input_prompt)
    print("\n=== FINAL OUTPUT ===")
    print(result.get("final_output", "No output produced."))