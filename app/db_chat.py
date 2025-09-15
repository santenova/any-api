"""
Database configuration and models for the LLM Network Gateway.

This module handles SQLite database setup, table definitions, and basic CRUD operations
for chat sessions and messages. Uses SQLAlchemy ORM for database interactions.
"""

import os
from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.exc import SQLAlchemyError

# Database URL - defaults to SQLite in data directory
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chat.db")

# SQLAlchemy engine with connection pooling for SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,  # Verify connections before use
    echo=False  # Set to True for SQL query logging during development
)

# Session factory for database operations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all database models
Base = declarative_base()


class ChatSession(Base):
    """
Represents a chat session/conversation.
Each session can use different models and maintains separate conversation history.
    """
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, default="New Chat")
    model_name = Column(String(100), nullable=False)  # The LLM model being used
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to messages - cascade delete when session is deleted
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatSession(id={self.id}, title='{self.title}', model='{self.model_name}')>"


class Message(Base):
    """
Represents individual messages within a chat session.
Stores both user inputs and LLM responses with timestamps.
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)  # The actual message content
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to the chat session
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role}', session_id={self.session_id})>"


def create_tables():
    """
Create all database tables if they don't exist.
Should be called on application startup.
    """
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except SQLAlchemyError as e:
        print(f"Error creating database tables: {e}")
        raise


def get_db() -> Session:
    """
Dependency function to get database session.
Ensures proper session cleanup after each request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database utility functions for common operations

def create_chat_session(db: Session, title: str, model_name: str) -> ChatSession:
    """Create a new chat session with specified model."""
    try:
        session = ChatSession(title=title, model_name=model_name)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Failed to create chat session: {e}")


def get_chat_session_by_title(db: Session, title: str) -> Optional[ChatSession]:
    """Retrieve a chat session by ID with its messages."""
    try:
        return db.query(ChatSession).filter(ChatSession.id == title).first()
    except SQLAlchemyError as e:
        raise Exception(f"Failed to retrieve chat session: {e}")


def get_chat_session(db: Session, session_id: int) -> Optional[ChatSession]:
    """Retrieve a chat session by ID with its messages."""
    try:
        return db.query(ChatSession).filter(ChatSession.id == session_id).first()
    except SQLAlchemyError as e:
        raise Exception(f"Failed to retrieve chat session: {e}")


def get_all_chat_sessions(db: Session) -> List[ChatSession]:
    """Retrieve all chat sessions ordered by most recent update."""
    try:
        return db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    except SQLAlchemyError as e:
        raise Exception(f"Failed to retrieve chat sessions: {e}")


def add_message(db: Session, session_id: int, role: str, content: str) -> Message:
    """Add a message to a chat session and update session timestamp."""
    try:
        message = Message(session_id=session_id, role=role, content=content)
        db.add(message)

        # Update session timestamp
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(message)
        return message
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Failed to add message: {e}")


def get_session_messages(db: Session, session_id: int, limit: int = 50) -> List[Message]:
    """Get messages for a session, limited and ordered by creation time."""
    try:
        return (db.query(Message)
                .filter(Message.session_id == session_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .all())
    except SQLAlchemyError as e:
        raise Exception(f"Failed to retrieve messages: {e}")


def delete_chat_session(db: Session, session_id: int) -> bool:
    """Delete a chat session and all its messages."""
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
            return True
        return False
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Failed to delete chat session: {e}")


def update_session_title(db: Session, session_id: int, new_title: str) -> bool:
    """Update the title of a chat session."""
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.title = new_title
            session.updated_at = datetime.utcnow()
            db.commit()
            return True
        return False
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Failed to update session title: {e}")
