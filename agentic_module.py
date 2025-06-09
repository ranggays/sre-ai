from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
import requests

@tool
def add_numbers(numbers: str) -> str:
    """Add a list of numbers."""
    try:
        nums = [float(n.strip()) for n in numbers.split(",")]
        return str(sum(nums))
    except Exception as e:
        return (f"Error adding numbers: {e}")
    
@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a summary of the query."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("extract", "No summary found.")
        return "Error fetching Wikipedia summary."
    except Exception as e:
        return f"Error during Wikipedia search: {e}"
    
tools = [add_numbers, wiki_search]
prompt_template = PromptTemplate.from_template(
    """You are a helpful assistant. Use the tools provided to asnwer the user's question.\n\n {input} 
    
    {agent_scratchpad}
    """
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
def run_agentic_task(task: str) -> str:
    result = agent_executor.invoke({"input": task})
    return {"result": result["output"]}

