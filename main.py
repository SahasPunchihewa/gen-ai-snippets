import os

import bs4
from fastapi import FastAPI
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from pydantic import BaseModel
from typing_extensions import List, TypedDict

app = FastAPI()

# os.environ['LANGSMITH_TRACING'] = 'true'
# os.environ['LANGSMITH_API_KEY'] = getpass.getpass()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini')

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=('post-content', 'post-title', 'post-header')
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull('rlm/rag-prompt')


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state['question'])
    return {'context': retrieved_docs}


def generate(state: State):
    docs_content = '\n\n'.join(doc.page_content for doc in state['context'])
    messages = prompt.invoke({'question': state['question'], 'context': docs_content})
    response = llm.invoke(messages)
    return {'answer': response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph = graph_builder.compile()


class Item(BaseModel):
    question: str


@app.post('/test', status_code=200)
async def test(item: Item):
    response = graph.invoke({'question': item.question})
    answer = response['answer']
    print(answer)
    return answer
