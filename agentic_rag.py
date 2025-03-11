from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.wikipedia import WikipediaTools
from phi.tools.googlesearch import GoogleSearch

import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

web_search_agent= Agent(
    name = "Healthcare web search agent",
    description = "Provides healthcare information from web sources",
    role = "Search the web for symptoms,conditions and treatments",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = [
        "Search for reliable healthcare information",
        "Focus on symptoms,causes and treatments"
    ],
    show_tool_calls = True,
    markdown = True,
)

wikipedia_agent = Agent(
    name = "Healthcare Wikipedia Agent",
    description = "Fetches healthcare information from wikipedia",
    model = Groq(id="llama-3.3-70b-versatile"),
    tools = [WikipediaTools()],
    instructions = [
        "Provide accurate healthcare information from wikipedia",
        "Focus on medical conditions and treatments"
    ],
    show_tool_calls = True,
    markdown = True,
)

google_search_agent = Agent(
name="Healthcare Google Search Agent",
    description="Searches Google for healthcare-related information.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[GoogleSearch()],
    instructions=[
        "Retrieve healthcare information from trusted sources.",
        "Focus on medical conditions, symptoms, and treatments."
    ],
    show_tool_calls=True,
    debug_mode=True,
)
multi_ai_agent = Agent(
    name = "Healthcare Chatbot Agent",
    description = "Answers healthcare questions and provides reliable information",
    team = [web_search_agent,wikipedia_agent,google_search_agent],
    model = Groq(id = "llama-3.3-70b-versatile"),
    instructions = [
        "Provide concise and accurate healthcare information",
        "use reliable sources only"
    ],
    show_tool_calls = True,
    markdown = True
)
try:
    multi_ai_agent.print_response(
        "How to treat influenza at home?",
        stream = True
    )

except Exception as e:
    print(f"an error occured : {e}")



