from fastapi import FastAPI
from typing_extensions import List

from in_memory_db import players, landmarks, trains
from rag import Item, graph

app = FastAPI()


# Sample RAG API
@app.post('/test', status_code=200)
async def test(item: Item):
    response = graph.invoke({'question': item.question})
    answer = response['answer']
    print(answer)
    return answer


# Mock APIs for tools
@app.get("/players", response_model=List[dict])
async def get_all_players():
    """Retrieve all players."""

    print('player request received')
    return players


@app.get("/landmarks", response_model=List[dict])
async def get_all_landmarks():
    """Retrieve all landmarks."""

    print('landmark request received')
    return landmarks


@app.get("/trains", response_model=List[dict])
async def get_all_trains():
    """Retrieve all landmarks."""

    print('train request received')
    return trains
