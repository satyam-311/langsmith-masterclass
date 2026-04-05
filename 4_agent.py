from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["LANGCHAIN_PROJECT"]="React Agent"

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  weatherstack_api_key = os.getenv("WEATHERSTACK_API_KEY")
  if not weatherstack_api_key:
    return "WEATHERSTACK_API_KEY is not set."

  url = f'https://api.weatherstack.com/current?access_key={weatherstack_api_key}&query={city}'

  response = requests.get(url)

  return str(response.json())

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# Step 2: Create the agent
agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
    system_prompt=(
        "You are a helpful assistant that can use search and weather tools."
    )
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Identify the birthplace city of Kalpana Chawla (search) and give its current temperature."}]}
)
print(response)

print(response["messages"][-1].content)
