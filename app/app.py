import asyncio, random
from sqlalchemy.orm import Session


from datetime import datetime

import os, sys, json
import httpx, time, requests
from fastapi.responses import  StreamingResponse, JSONResponse, Response, FileResponse


from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, BackgroundTasks, Query, Body
from fastapi import FastAPI, Request, HTTPException, APIRouter

from .agents.agent_manager import AgentManager
from dotenv import load_dotenv

load_dotenv()

# Import our custom modules
from database import (
    create_tables, get_db, create_chat_session, get_chat_session,
    get_all_chat_sessions, add_message, get_session_messages,
    delete_chat_session, update_session_title, ChatSession, Message
)
from ollama_service import (
    get_ollama_service, close_ollama_service, OllamaService,
    OllamaConnectionError, OllamaModelError
)



from app.db import User, create_db_and_tables
from app.schemas import UserCreate, UserRead, UserUpdate
from app.users import (
    SECRET,
    auth_backend,
    current_active_user,
    fastapi_users,
    google_oauth_client,
)

from app.classes import (Products, Agents, Git, Base44, Rooms, Item, User, Annotated, FilterParams, HealthResponse)

from app.agents.agent_base import AgentBase
from app.utils.logger import logger

OLLAMA_BASE = os.getenv('OLLAMA_BASE', 'http://ollama:11434')
DEFAULT_MODEL = os.getenv('OLLAMA_MODEL',"qwen3:0.6b")
FAVICON_PATH = 'favicon.ico'

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
Application lifespan manager.
Handles startup and shutdown tasks like database initialization.
    """
    # Startup
    if OLLAMA_BASE:
      print("🚀 Ollama base url: "+OLLAMA_BASE)

    # Create database tables
    try:
        await health_check()
        create_tables()
        await create_db_and_tables()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise

    # Test Ollama connection
    try:
        ollama = await get_ollama_service()
        is_healthy = await ollama.health_check()
        if is_healthy:
            print("✅ Ollama connection verified")
        else:
            print("⚠️ Warning: Ollama service not responding")
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to Ollama: {e}")

    yield

    # Shutdown
    print("🛑 Shutting down Local LLM Network Gateway...")
    await close_ollama_service()



app = FastAPI(lifespan=lifespan)

products = Products("World")
agents   = Agents("Base")
git      = Git("World")
data     = Base44("data World")
rooms    = Rooms("any")


app.include_router(rooms.router,prefix="/rooms",tags=["rooms"])
app.include_router(products.router,prefix="/products",tags=["products"])
app.include_router(agents.router,  prefix="/agents",tags=["agents"])
app.include_router(git.router,     prefix="/git",tags=["git"])
app.include_router(data.router,    prefix="/data",tags=["data"])


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


#@app.get("/agent/benchmark_models")
def has_data(data):
    try:
      if len(data) > 0:
        return True
      else:
        return False
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


@app.post("/send-notification/{email}")
async def send_notification(self,
        email: str, background_tasks: BackgroundTasks, q: str = Depends(get_query)
    ):
        message = f"message to {email}\n"
        background_tasks.add_task(write_log, message)
        return {"message": "Message sent"}


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(FAVICON_PATH)


@app.get("", response_model=HealthResponse)
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
Health check endpoint to verify service status.
Tests both database and Ollama connectivity.
    """
    ollama_connected = False
    database_connected = False

    # Test Ollama connection

    activity = Base44("data").activity()
    products = Base44("data").products()
    subscription = Base44("data").subscription()
    ollama = await get_ollama_service()
    ollama_connected = await ollama.health_check()

    models = await Rooms('any-api').get_available_models()
    random.shuffle(models['models'])

    model =models['models']
    messages = []
    has_activity = has_data(activity)
    has_products = has_data(products)
    has_subscription = has_data(subscription)
    #title = await ollama.generate_title(model,"hello i am the user")

    # Test database connection
    try:
        # Simple database query to test connectivity
        db = next(get_db())
        database_connected = True
        db.close()
    except Exception as e:

        logger.info(f"[{e}][] ")
        pass

    status = "healthy" if (ollama_connected and database_connected) else "degraded"

    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow().isoformat(),
        ollama_connected=ollama_connected,
        database_connected=database_connected,
        data_activity=has_activity,
        data_subscription=has_subscription,
        data_products=has_products
    )
