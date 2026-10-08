import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel
from langchain_core.tracers.context import collect_runs

load_dotenv()
api_key=str(os.getenv("API_KEY"))
base_url=str(os.getenv("BASE_URL"))

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    base_url=base_url,
    temperature=0.7,
)

prompt = PromptTemplate(
    template="Tell me a joke about {topic}"
)
parser = StrOutputParser()

response = parser.invoke(
    llm.invoke(
        prompt.invoke(
            {"topic": "python"}
        )
    )
)

print(response)

runnables = [prompt, llm, parser]