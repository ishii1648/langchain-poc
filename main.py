import os
import asyncio

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Optional
from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.tools.base import BaseTool
from langchain.tools import Tool, tool
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from bs4 import BeautifulSoup
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()

question = "Cloud Run JobsをTerraformで作成するコードを教えてください"


class GoogleSerperRun(BaseTool):
    """Tool that queries the Serper.dev Google search API."""

    name: str = "google_serper"
    description: str = "A low-cost Google Search API." "Input should be a search query."
    api_wrapper: SerpAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        # self.api_wrapper.params = {"as_sitesearch": "registry.terraform.io", "hl": "ja"}
        self.api_wrapper.params = {"gl": "jp", "hl": "ja"}
        return str(self.api_wrapper.run(query))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("google_serper does not support async")


async def ascrape_playwright(url: str) -> str:
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)
            await page.keyboard.press("PageDown")
            await asyncio.sleep(3)
            results = await page.content()  # Simply get the HTML content
        except Exception as e:
            print(f"Error: {e}")
        await browser.close()
    return results


def lookup(url: str):
    html_content = asyncio.run(
        ascrape_playwright(
            "https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/cloud_run_v2_job"
        )
    )
    bs = BeautifulSoup(html_content, "html.parser")

    html: str = ""
    for article in bs.select("article#provider-docs-content"):
        html = article.decode_contents()

    file_name = "terraform_docs.html"
    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(html)

    loader = UnstructuredHTMLLoader("./terraform_docs.html")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    db = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="./db")

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=500),
        chain_type="stuff",
        retriever=retriever,
    )

    result = qa.run(question)
    return result


llm = OpenAI(temperature=0)
tools = [
    GoogleSerperRun(api_wrapper=SerpAPIWrapper()),
    Tool.from_function(
        func=lookup,
        name="lookup",
        description=(
            "useful for when you need to answer questions about current events"
            "Input should be a URL."
        ),
    ),
]

suffix = """
Follow the conditions:
- Please respond in Japanese.
- The domain of the page URL to be obtained using `google_serper` MUST be "registry.terraform.io". If the domain of the acquired URL is different, please retry search using `google_serper`.

Let's get started!
Question: {input}
{agent_scratchpad}"""

format_instructions = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    suffix=suffix,
    format_instructions=format_instructions,
    input_variables=["input", "agent_scratchpad"],
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]

agent = AgentExecutor.from_agent_and_tools(
    agent=ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names),
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    # return_intermediate_steps=True,
)

agent.run(question)
