from beanie import Document
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC

class User(Document):
    username: str
    display_name: Optional[str] = None
    created_at: datetime = datetime.now(UTC)
    last_login: datetime = datetime.now(UTC)
    is_active: bool = True
    
    # Session and Bot Management
    bot_id: Optional[str] = None  
    current_session_id: Optional[str] = None  
    
    
    facts: Dict[str, Any] = {
        "personal": {},      # name, age, location, etc.
        "preferences": {},   # likes, dislikes, interests
        "work": {},         # job, company, skills
        "family": {},       # family members, relationships
        "health": {},       # health info, conditions
        "goals": {},        # aspirations, targets
        "experiences": {},  # past experiences, memories
        "other": {}         # miscellaneous facts
    }
    
    # Metadata about facts
    facts_metadata: Dict[str, Any] = {
        "last_updated": None,
        "total_facts": 0,
        "sources": []  # Track which messages contributed facts
    }

    class Settings:
        name = "users"

class Bot(Document):
    bot_id: str  # Unique identifier for the bot
    bot_name: str  # Display name for the bot
    system_prompt: str  # Customized system prompt for this bot
    prompt_customized: bool = False  # Track if prompt has been personalized
    created_at: datetime = datetime.now(UTC)
    updated_at: datetime = datetime.now(UTC)
    
    # Bot configuration and metadata
    configuration: Dict[str, Any] = {
        "personality": "helpful",
        "style": "conversational",
        "knowledge_areas": [],
        "preferences": {}
    }

    class Settings:
        name = "bots"

class Session(Document):
    session_id: str  # Unique identifier for the session
    user_id: str  # References username
    bot_id: str  # Which bot was used in this session
    status: str = "active"  # active, ended, archived
    created_at: datetime = datetime.now(UTC)
    updated_at: datetime = datetime.now(UTC)
    ended_at: Optional[datetime] = None
    
    # Session metadata
    metadata: Dict[str, Any] = {
        "message_count": 0,
        "duration": None,
        "topics": []
    }

    class Settings:
        name = "sessions"

class ChatMessage(Document):
    user_id: str  # This will be the username
    session_id: str  # Links message to session
    bot_id: str  # Which bot generated the response
    message: str
    response: str
    timestamp: datetime = datetime.now(UTC)
    
    # Message metadata
    metadata: Dict[str, Any] = {
        "facts_extracted": 0,
        "response_time": None,
        "tokens_used": None
    }

    class Settings:
        name = "chat_messages" 