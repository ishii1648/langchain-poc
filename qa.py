import os
import asyncio

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from bs4 import BeautifulSoup
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


load_dotenv()


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


html_content = asyncio.run(
    ascrape_playwright(
        "https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/cloud_run_v2_job"
    )
)

from bs4 import BeautifulSoup
import tiktoken
from tiktoken.core import Encoding

bs = BeautifulSoup(html_content, "html.parser")

html: str = ""
for article in bs.select("article#provider-docs-content"):
    html = article.decode_contents()

encoding: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens = encoding.encode(text=html)
tokens_count = len(tokens)

# print(html)
# print(f"{tokens_count=}")

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

llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=500)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
print(qa.run("Cloud Run Jobsを作成するTerraformのコードを教えてください"))
