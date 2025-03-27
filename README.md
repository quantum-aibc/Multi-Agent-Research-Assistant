# Multi-Agent-Research-Assistant
import os
import pinecone
import requests
import openai
from crewai import Agent, Task, Crew
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# Set API Keys
OPENAI_API_KEY = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENV = "your-pinecone-environment"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "ai-research"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")
vectorstore = LC_Pinecone(index_name=index_name, embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY))

# Research Agent: Fetches AI papers from arXiv
def fetch_arxiv_papers(query, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}"
    response = requests.get(url)
    return response.text  # (For simplicity, actual parsing needed)

research_agent = Agent(
    name="ResearchAgent",
    role="Research Specialist",
    goal="Find latest AI research papers",
    tools=[fetch_arxiv_papers]
)

# Summarization Agent
def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Summarize this: {text}"}]
    )
    return response["choices"][0]["message"]["content"]

summarization_agent = Agent(
    name="SummarizerAgent",
    role="AI Summarization Expert",
    goal="Summarize research papers",
    tools=[summarize_text]
)

# Analysis Agent: Performs trend analysis
def analyze_trends(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Analyze trends in this research: {text}"}]
    )
    return response["choices"][0]["message"]["content"]

analysis_agent = Agent(
    name="AnalysisAgent",
    role="AI Research Analyst",
    goal="Analyze research trends",
    tools=[analyze_trends]
)

# Define Tasks
task1 = Task("Fetch research papers on latest AI topics", agent=research_agent)
task2 = Task("Summarize key findings from research papers", agent=summarization_agent)
task3 = Task("Analyze AI research trends and insights", agent=analysis_agent)

# Crew Coordination
crew = Crew(agents=[research_agent, summarization_agent, analysis_agent], tasks=[task1, task2, task3])
crew.kickoff()

print("AI Research Assistant is running...")
