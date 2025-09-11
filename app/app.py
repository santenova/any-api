import asyncio, random
from typing import Annotated, Literal
from sqlalchemy.orm import Session


from pydantic import BaseModel, Field
from datetime import datetime

import os, sys, json
import httpx, time, requests
from fastapi.responses import  StreamingResponse, JSONResponse, Response, FileResponse
from typing import Optional, List, Dict, Any

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


class FilterParams(BaseModel):
    model_config = {"extra": "forbid"}

    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: list[str] = []


class ChatMessage(BaseModel):
    """Model for individual chat messages."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ChatSessionCreate(BaseModel):
    """Model for creating new chat sessions."""
    title: Optional[str] = Field("New Chat", description="Session title")
    model_name: str = Field(..., description="LLM model to use")


class ChatSessionResponse(BaseModel):
    """Model for chat session data returned to frontend."""
    id: int
    title: str
    model_name: str
    created_at: str
    updated_at: str
    message_count: int


class SendMessageRequest(BaseModel):
    """Model for sending messages to chat sessions."""
    content: str = Field(..., description="User message content")
    stream: bool = Field(True, description="Whether to stream the response")


class UpdateSessionTitleRequest(BaseModel):
    """Model for updating session titles."""
    title: str = Field(..., min_length=1, max_length=255, description="New session title")


class HealthResponse(BaseModel):
    """Model for health check responses."""
    status: str
    timestamp: str
    ollama_connected: bool
    database_connected: bool
    data_activity: bool
    data_subscription: bool
    data_products: bool



class Prompt(BaseModel):
    model: str
    prompt: str
    amount: int



message = """
write a company presentation book about only-agent.ai a startup
with a saas platform and AI agents to buy
create a list of 100 creative points to help the startup to succseed
the startup is not fully funded yet
there looking for suitable products to add to there shop
the purpose of the book is to attract seed investors to the startup

          """

class Rooms:

    def __init__(self, name: str):
        self.name = name
        self.router = APIRouter()
        self.router.get("/hello")(self.hello) # use decorator
        self.router.get("/get_available_models")(self.get_available_models) # use decorator
        self.router.get("/sessions")(self.get_chat_sessions) # use decorator
        self.router.post("/new_session")(self.create_new_chat_session) # use decorator
        self.router.post("/sessions/{session_id}")(self.get_chat_session_details) # use decorator
        self.router.post("/sessions/{session_id}/messages")(self.send_message) # use decorator
        self.router.put("/sessions/{session_id}/title")(self.update_chat_session_title) # use decorator
        self.router.delete("/sessions/{session_id}")(self.delete_chat_session_endpoint) # use decorator
        self.router.get("/sessions/{session_id}/export")(self.export_chat_session) # use decorator


    def hello(self):
        return {"Hello": self.name}

    # self.router.get("/api/models")
    async def get_available_models(self):
        """
    Get list of available LLM models from Ollama.
    Returns model information including name, size, and last modified date.
        """
        try:
            ollama = await get_ollama_service()
            models = await ollama.get_available_models()
            return {"models": models}
        except OllamaConnectionError as e:
            raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {e}")
        except OllamaModelError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")





    # self.router.get("/api", response_model=List[ChatSessionResponse])
    async def get_chat_sessions(self,db: Session = Depends(get_db)):
        """
    Get all chat sessions with basic information.
    Returns sessions ordered by most recent activity.
        """
        try:
            sessions = get_all_chat_sessions(db)
            response = []

            for session in sessions:
                response.append(ChatSessionResponse(
                                    id=session.id,
                                    title=session.title,
                                    model_name=session.model_name,
                                    created_at=session.created_at.isoformat(),
                                    updated_at=session.updated_at.isoformat(),
                                    message_count=len(session.messages)
                                ))

            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve sessions: {e}")


    #@app.post("/api/sessions", response_model=ChatSessionResponse)
    async def create_new_chat_session(self,
            session_data: ChatSessionCreate,
            db: Session = Depends(get_db)
    ):
        """
    Create a new chat session with specified model.
    Validates that the model is available in Ollama.
        """
        try:
            # Validate model exists
            ollama = await get_ollama_service()
            if not await ollama.validate_model(session_data.model_name):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{session_data.model_name}' is not available"
                )

            # Create session in database
            session = create_chat_session(db, session_data.title, session_data.model_name)

            return ChatSessionResponse(
                id=session.id,
                title=session.title,
                model_name=session.model_name,
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
                message_count=0
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")


    # self.router.get("/api/sessions/{session_id}")
    async def get_chat_session_details(self,session_id: int, db: Session = Depends(get_db)):
        """
    Get detailed information about a specific chat session including messages.
        """
        try:
            session = get_chat_session(db, session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")

            messages = get_session_messages(db, session_id)
            message_list = []

            for msg in messages:
                message_list.append({
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat()
                })

            return {
                "id": session.id,
                "title": session.title,
                "model_name": session.model_name,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "messages": message_list
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {e}")


    #@app.post("/api/sessions/{session_id}/messages")
    async def send_message(self,
            session_id: int,
            message_data: SendMessageRequest,
            background_tasks: BackgroundTasks,
            db: Session = Depends(get_db)
    ):
        """
    Send a message to a chat session and get LLM response.
    Supports both streaming and non-streaming responses.
        """
        try:
            # Verify session exists
            session = get_chat_session(db, session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")

            # Store user message
            user_message = add_message(db, session_id, "user", message_data.content)

            # Get conversation history for context
            messages = get_session_messages(db, session_id, limit=20)  # Last 20 messages
            conversation_history = []

            for msg in messages:
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content
                })

            # Get Ollama service and send request
            ollama = await get_ollama_service()

            if message_data.stream:
                # Return streaming response
                return StreamingResponse(
                    stream_chat_response(ollama, session, conversation_history, db, session_id),
                    media_type="text/plain"
                )
            else:
                # Return complete response
                response_content = ""
                async for chunk in ollama.chat_completion(
                    session.model_name,
                    conversation_history,
                    stream=False
                ):
                    response_content = chunk.get("message", {}).get("content", "")

                # Store assistant response
                assistant_message = add_message(db, session_id, "assistant", response_content)

                # Update session title if this is the first exchange
                if len(messages) <= 2:  # User message + assistant response
                    background_tasks.add_task(
                        update_session_title_async,
                        db, session_id, ollama, session.model_name, message_data.content
                    )

                return {
                    "user_message": {
                        "id": user_message.id,
                        "content": user_message.content,
                        "timestamp": user_message.created_at.isoformat()
                    },
                    "assistant_message": {
                        "id": assistant_message.id,
                        "content": assistant_message.content,
                        "timestamp": assistant_message.created_at.isoformat()
                    }
                }

        except HTTPException:
            raise
        except OllamaConnectionError as e:
            raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e}")
        except OllamaModelError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process message: {e}")


    async def stream_chat_response(self,
            ollama: OllamaService,
            session: ChatSession,
            conversation_history: List[Dict[str, str]],
            db: Session,
            session_id: int
    ):
        """
    Generator function for streaming chat responses.
    Handles real-time LLM response streaming and database updates.
        """
        response_content = ""

        try:
            async for chunk in ollama.chat_completion(
                session.model_name,
                conversation_history,
                stream=True
            ):
                # Extract content from chunk
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                    if content:
                        response_content += content
                        # Send chunk to frontend
                        yield f"data: {json.dumps({'content': content, 'type': 'chunk'})}\n\n"

                # Check if response is complete
                if chunk.get("done", False):
                    # Store complete assistant response in database
                    if response_content.strip():
                        add_message(db, session_id, "assistant", response_content)

                    # Generate title if this is the first exchange
                    message_count = len(get_session_messages(db, session_id))
                    if message_count <= 2:
                        asyncio.create_task(
                            update_session_title_async(
                                db, session_id, ollama, session.model_name,
                                conversation_history[-1]["content"]
                            )
                        )

                    # Send completion signal
                    yield f"data: {json.dumps({'type': 'done', 'message_id': None})}\n\n"
                    break

        except Exception as e:
            # Send error to frontend
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


    async def update_session_title_async(self,
            db: Session,
            session_id: int,
            ollama: OllamaService,
            model_name: str,
            first_message: str
    ):
        """
    Background task to generate and update session title.
        """
        try:
            title = await ollama.generate_title(model_name, first_message)
            update_session_title(db, session_id, title)
        except Exception:
            pass  # Silently fail title generation


    #@app.put("/api/sessions/{session_id}/title")
    async def update_chat_session_title(self,
            session_id: int,
            title_data: UpdateSessionTitleRequest,
            db: Session = Depends(get_db)
    ):
        """
    Update the title of a chat session.
        """
        try:
            success = update_session_title(db, session_id, title_data.title)
            if not success:
                raise HTTPException(status_code=404, detail="Chat session not found")

            return {"message": "Session title updated successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update title: {e}")


    #@app.delete("/api/sessions/{session_id}")
    async def delete_chat_session_endpoint(self,session_id: int, db: Session = Depends(get_db)):
        """
    Delete a chat session and all its messages.
        """
        try:
            success = delete_chat_session(db, session_id)
            if not success:
                raise HTTPException(status_code=404, detail="Chat session not found")

            return {"message": "Chat session deleted successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete session: {e}")


    # self.router.get("/api/sessions/{session_id}/export")
    async def export_chat_session(self,session_id: int, db: Session = Depends(get_db)):
        """
    Export chat session as JSON for backup or sharing.
        """
        try:
            session = get_chat_session(db, session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")

            messages = get_session_messages(db, session_id)

            export_data = {
                "session": {
                    "id": session.id,
                    "title": session.title,
                    "model_name": session.model_name,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat()
                },
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat()
                    }
                    for msg in messages
                ],
                "export_timestamp": datetime.utcnow().isoformat()
            }

            return JSONResponse(
                content=export_data,
                headers={"Content-Disposition": f"attachment; filename=chat_{session_id}.json"}
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to export session: {e}")


    # Error handlers for better user experience

    #@app.exception_handler(self,404)
    async def not_found_handler(request, exc):
        """Custom 404 error handler."""
        return JSONResponse(
            status_code=404,
            content={"detail": "Endpoint not found", "path": str(request.url)}
        )


    #@app.exception_handler(500)
    async def internal_error_handler(self,request, exc):
        """Custom 500 error handler."""
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error occurred"}
        )





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
        self.router.post("/concepts")(self.list_concepts) # use decorator
        self.router.post("/custom")(self.custom_lake) # use decorator
        self.router.post("/chat")(self.agent_chat) # use decorator
        self.router.post("/model_meter")(self.test_models) # use decorator
        self.router.post("/create_book")(self.create_book) # use decorator
        self.router.post("/models")(self.models) # use decorator



    #@app.post("/agent/concepts")
    def list_concepts(self, prompt: Prompt):


        if prompt.model is None:
          model = DEFAULT_MODEL
        else:
          model = prompt.model

        # Set the root concept and model
        if prompt.prompt is None:
          prompt.prompt = "How to make most easy revenue with AI Agents as a startup, convince seed investors"  # Default root concept

        if prompt.amount is None:
          prompt.amount = 10

        concept = prompt.prompt
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

        return JSONResponse(content=res.text)
        #return {}



    #@app.post("/custom")
    def custom_lake(self, prompt: Prompt):
      try:

        # Set the root concept and model
        if prompt.prompt is None:
          prompt.prompt = "How to make most easy revenue with AI Agents as a startup, convince seed investors"  # Default root concept

        if prompt.amount is None:
          prompt.amount = 10

        request_data = {"prompt":prompt.prompt, "amount":prompt.amount}

        agent_manager = AgentManager(max_retries=2, verbose=True)
        agent = agent_manager.get_agent("hub_products")


        out = agent.execute(agent_manager.model,prompt.prompt,1,prompt.amount)

        return JSONResponse(
                content=out
            )

      except Exception as e:
          logger.error(f"Error listing models: {str(e)}")
          raise HTTPException(status_code=500, detail=str(e))




    #@app.post("/agent/chat")
    def agent_chat(self, prompt: Prompt):
        try:

          if prompt.model is None:
            model = DEFAULT_MODEL
          else:
            model = prompt.model

          if prompt.prompt is None:
            prompt.prompt = message


          res = requests.post(f"{OLLAMA_BASE}/api/generate", json={
            "prompt": prompt.prompt,
            "stream" : False,
            "model" : model
          })

          return JSONResponse(content=res.text)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))





    #@app.post("/agent/test_models")
    async def test_models(self):

        agent_manager = AgentManager(max_retries=2, verbose=True)
        agent = agent_manager.get_agent("models_perf_tool")


        out = agent.execute(DEFAULT_MODEL)

        return JSONResponse(
                content=out
            )


    #@app.post("/agent/create_book")
    async def create_book(self, prompt: Prompt):
        try:

          if prompt.model is None:
            model = DEFAULT_MODEL
          else:
            model = prompt.model

          if prompt.prompt is None:
            prompt.prompt = message

          concept = prompt.prompt



          request_data = {"prompt":prompt}

          agent_manager = AgentManager(max_retries=2, verbose=True)
          agent = agent_manager.get_agent("write_book")


          out = agent.execute()

          return JSONResponse(
                  content=out
              )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    #@app.get("/agent/create_curse")
    async def create_curse(self):
        try:
          return {}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    #@app.get("/agent/create_tutorial")
    async def create_tutorial(self):
        try:
          return {}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    #@app.get("/agent/benchmark_models")
    def has_data(self, data):
        try:
          if len(data) > 0:
            return True
          else:
            return False
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))



    #@app.get("/agent/benchmark_models")
    def benchmark_models(self):
        try:
          return {}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    #@app.get("/agent/models")
    async def models(self):
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




    def agents(self):
        return {"agents": self.name}


class Base44:

    def __init__(self, name: str):
        self.name = name
        self.router = APIRouter()
        self.router.get("/products")(self.products) # use decorator
        self.router.get("/activity")(self.activity) # use decorator
        self.router.get("/subscription")(self.subscription) # use decorator

    def products(self):
      entities = self.make_api_request(f'apps/68aa4e39f2b74e241c8a6bd3/entities/Product')
      return entities#print(entities)



    def make_api_request(self, api_path, method='GET', data=None):
        url = f'https://app.base44.com/api/{api_path}'
        headers = {
            'api_key': 'ef5e2e3b8e524b6d91e6e85136f86d9f',
            'Content-Type': 'application/json'
        }
        if method.upper() == 'GET':
            response = requests.request(method, url, headers=headers, params=data)
        else:
            response = requests.request(method, url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()



    def activity(self):
      entities = self.make_api_request(f'apps/68aa4e39f2b74e241c8a6bd3/entities/ActivityLog')
      return entities

    def subscription(self):
        entities = self.make_api_request(f'apps/68aa4e39f2b74e241c8a6bd3/entities/Subscription')
        return entities



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
rooms     = Rooms("any")


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


