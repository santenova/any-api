from typing import Annotated, Literal


from pydantic import BaseModel, Field

import os, sys, json
import httpx, time, requests
from fastapi.responses import  StreamingResponse, JSONResponse, Response
from typing import Optional, List, Dict, Any

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, BackgroundTasks, Query, Body
from fastapi import FastAPI, Request, HTTPException, APIRouter

from .agents.agent_manager import AgentManager
from dotenv import load_dotenv


load_dotenv()



class FilterParams(BaseModel):
    model_config = {"extra": "forbid"}

    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: list[str] = []








class Git:

    def __init__(self, name: str):
        self.name = name
        self.router = APIRouter()
        self.router.get("/hello")(self.hello) # use decorator

    def hello(self):
        return {"Hello": self.name}


class Agents:

    def __init__(self, name: str):
        self.name = name
        self.router = APIRouter()
        self.router.get("/agents")(self.agents) # use decorator

    def agents(self):
        return {"agents": self.name}

class Products:

    def __init__(self, name: str):
        self.name = name
        self.router = APIRouter()
        self.router.get("/products")(self.products) # use decorator

    def products(self):
        return {"products": self.name}

from app.db import User, create_db_and_tables
from app.schemas import UserCreate, UserRead, UserUpdate
from app.users import (
    SECRET,
    auth_backend,
    current_active_user,
    fastapi_users,
    google_oauth_client,
)

from app.agents.agent_base import AgentBase
from app.utils.logger import logger

OLLAMA_BASE = os.getenv('OLLAMA_BASE', 'http://ollama:11434')
DEFAULT_MODEL = os.getenv('OLLAMA_MODEL',"qwen3:0.6b")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Not needed if you setup a migration system like Alembic
    await create_db_and_tables()
    yield

print(OLLAMA_BASE)
app = FastAPI(lifespan=lifespan)


git = Git("World")
app.include_router(git.router,prefix="/git",tags=["git"])


products = Products("World")
app.include_router(products.router,prefix="/products",tags=["products"])


agents = Agents("Base")
app.include_router(agents.router,prefix="/agents",tags=["agents"])


app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)
app.include_router(
    fastapi_users.get_oauth_router(google_oauth_client, auth_backend, SECRET),
    prefix="/auth/google",
    tags=["auth"],
)


@app.get("/authenticated-route")
async def authenticated_route(user: User = Depends(current_active_user)):
    return {"message": f"Hello {user.email}!"}




@app.get("/items/")
async def read_items(filter_query: Annotated[FilterParams, Query()]):
    return filter_query



class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


class User(BaseModel):
    username: str
    full_name: str | None = None


@app.put("/items/{item_id}")
async def update_item(
    *,
    item_id: int,
    item: Item,
    user: User,
    importance: Annotated[int, Body(gt=0)],
    q: str | None = None,
):
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    if q:
        results.update({"q": q})
    return results



@app.post("/agent/concepts")
def list_concepts(in_model: str,prompt: str, amount: int):
    try:

      if in_model is None:
        model = DEFAULT_MODEL
      else:
        model = in_model

      message = """
                 write a company presentation book about only-agent.ai a startup,\n
                 with a saas platform and AI agents to buy,\n
                 create a list of 100 creative points to help the startup to succseed\n
                 the startup is not fully funded yet\n
                 there looking for suitable products to add to there shop\n
                 the purpose of the book is to attract seed investors to the startup\n\n
                """

      if prompt is None:
        prompt = message

      concept = prompt
      # Full path to current concept, including the concept itself
      full_path = [concept]
      amount = 20

      # Prompt
      prompt = f"""
Starting with the concept: "{concept}", generate {amount} to 50, of the most close related concepts to our Starting concept.

Context: We're building a concept web and have followed this path to get here:
{' → '.join(full_path)}

Guidelines:
1. Seek maximum intellectual diversity - span across domains like science, art, philosophy, technology, culture, etc.
2. Each concept should be expressed in 1-5 words (shorter is better)
3. Avoid obvious associations - prefer surprising or thought-provoking connections
4. Consider how your suggested concepts relate to BOTH:
 - The immediate parent concept "{concept}"
 - The overall path context: {' → '.join(full_path)}
5. Consider these different types of relationships:
 - Metaphorical parallels
 - Contrasting opposites
 - Historical connections
 - Philosophical implications
 - Cross-disciplinary applications

Avoid any concepts already in the path. Be creative but maintain meaningful connections.

Return ONLY a JSON array of strings, with no explanation or additional text.
Example: ["Related concept 1", "Related concept 2", "Related concept 3", "Related concept 4","Related concept 5", "Related concept 6", "Related concept 7", "Related concept 8"]
      """


      res = requests.post(f"{OLLAMA_BASE}/api/generate", json={
        "prompt": prompt,
        "stream" : False,
        "model" : model
      })

      return Response(content=res.text, media_type="application/json")
      #return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str
    amount: int


@app.post("/custom")
def custom_lake(prompt: Prompt):
  try:

    # Set the root concept and model
    if prompt.prompt is None:
      prompt.prompt = "How to make most easy revenue with AI Agents as a startup, convince seed investors"  # Default root concept

    if prompt.amount is None:
      prompt.amount = 10

    request_data = {"prompt":prompt.prompt, "amount":prompt.amount}

    agent_manager = AgentManager(max_retries=2, verbose=True)
    agent = agent_manager.get_agent("hub_products")


    out = {} #agent.execute(request_data,agent_manager.model,prompt,1,amount)

    return JSONResponse(
            content=[request_data,out]
        )

  except Exception as e:
      logger.error(f"Error listing models: {str(e)}")
      raise HTTPException(status_code=500, detail=str(e))




@app.post("/agent/chat")
async def list_settings(request: Request):
    try:

      request_data = await request.json()
      if request_data is  None:
        request_data = "{}"




      model = request_data.get('model', DEFAULT_MODEL)


      message = """
               write a company presentation book about only-agent.ai a startup,\n
               with a saas platform and AI agents to buy,\n
               create a list of 100 creative points to help the startup to succseed\n
               the startup is not fully funded yet\n
               there looking for suitable products to add to there shop\n
               the purpose of the book is to attract seed investors to the startup\n\n
               """

      prompt = request_data.get('prompt', message)


      res = requests.post(f"{OLLAMA_BASE}/api/generate", json={
        "prompt": prompt,
        "stream" : False,
        "model" : model
      })

      return Response(content=res.text, media_type="application/json")
      #return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/create_book")
async def create_book():
    try:
      return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/create_curse")
async def create_curse():
    try:
      return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/create_tutorial")
async def create_tutorial():
    try:
      return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/agent/test_models")
async def test_models():
    try:
      return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#@app.get("/agent/benchmark_models")
def benchmark_models():
    try:
      return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/models")
async def models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE}/api/tags")
            models = response.json()
            return {
                "object": "list",
                "data": [{"id": model["name"],
                         "object": "model",
                         "created": int(time.time()),
                         "owned_by": "organization-owner"}
                        for model in models.get("models", [])]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




def write_log(message: str):
    print(message)
    with open("log.txt", mode="a") as log:
        log.write(message)


def get_query(background_tasks: BackgroundTasks, q: str | None = None):
    if q:
        message = f"found query: {q}\n"
        background_tasks.add_task(write_log, message)
    return q



@app.get("/health")
async def models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE}/api/tags")
            models = response.json()
            if len(models.get("models", [])):
              return True


    except Exception as e:
        return False

@app.post("/task/send-notification/{email}")
async def send_notification(
    email: str, background_tasks: BackgroundTasks, q: str = Depends(get_query)
):
    message = f"message to {email}\n"
    background_tasks.add_task(write_log, message)
    return {"message": "Message sent"}



