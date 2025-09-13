import  sys, os
from sqlalchemy.orm import Session
import requests
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from typing import Optional, List, Dict, Any
from fastapi import Depends, BackgroundTasks, APIRouter
from fastapi.responses import  StreamingResponse, JSONResponse, Response, FileResponse

from dotenv import load_dotenv

load_dotenv()

from .agents.agent_manager import AgentManager



OLLAMA_BASE = os.getenv('OLLAMA_BASE', 'http://ollama:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL',"qwen3:0.6b")
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
                        self.update_session_title_async,
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
                            self.update_session_title_async(
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
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
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
          model = OLLAMA_MODEL
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
            model = OLLAMA_MODEL
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


        out = agent.execute(OLLAMA_MODEL)

        return JSONResponse(
                content=out
            )


    #@app.post("/agent/create_book")
    async def create_book(self, prompt: Prompt):


        if prompt.model is None:
          model = OLLAMA_MODEL
        else:
          model = prompt.model

        if prompt.prompt is None:
          prompt.prompt = message

        concept = prompt.prompt



        request_data = {"prompt":prompt}

        agent_manager = AgentManager(max_retries=2, verbose=True)
        agent = agent_manager.get_agent("write_book")


        out = agent.execute(concept)

        return JSONResponse(
                content=out
            )



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



class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


class User(BaseModel):
    username: str
    full_name: str | None = None



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
