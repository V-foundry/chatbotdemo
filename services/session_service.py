 import uuid
from typing import Optional, List, Dict
from datetime import datetime, UTC
from models import User, Session, ChatMessage
from database import is_connected
from loguru import logger

async def create_session(user_id: str, bot_id: str = None) -> str:
    """Create a new conversation session for the user."""
    if not is_connected():
        return f"session_{user_id}_default"
    
    try:
        # End any existing active session
        await end_user_sessions(user_id)
        
        # Ensure user has a bot
        if not bot_id:
            from services.bot_service import ensure_user_has_bot
            bot_id = await ensure_user_has_bot(user_id)
        
        # Generate unique session_id
        session_id = f"session_{user_id}_{uuid.uuid4().hex[:8]}"
        
        # Create session document
        new_session = Session(
            session_id=session_id,
            user_id=user_id,
            bot_id=bot_id,
            status="active"
        )
        await new_session.insert()
        
        # Update user's current_session_id
        user = await User.find_one(User.username == user_id)
        if user:
            user.current_session_id = session_id
            await user.save()
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
        
    except Exception as e:
        logger.error(f"Error creating session for user {user_id}: {e}")
        return f"session_{user_id}_error"

async def get_active_session(user_id: str) -> Optional[Session]:
    """Get user's current active session."""
    if not is_connected():
        return None
    
    try:
        user = await User.find_one(User.username == user_id)
        if not user or not user.current_session_id:
            return None
        
        session = await Session.find_one(
            Session.session_id == user.current_session_id,
            Session.status == "active"
        )
        return session
        
    except Exception as e:
        logger.error(f"Error retrieving active session for user {user_id}: {e}")
        return None

async def end_user_sessions(user_id: str) -> bool:
    """End all active sessions for a user."""
    if not is_connected():
        return True
    
    try:
        # Find and end all active sessions for the user
        active_sessions = await Session.find(
            Session.user_id == user_id,
            Session.status == "active"
        ).to_list()
        
        for session in active_sessions:
            session.status = "ended"
            session.ended_at = datetime.now(UTC)
            session.updated_at = datetime.now(UTC)
            await session.save()
        
        # Clear user's current_session_id
        user = await User.find_one(User.username == user_id)
        if user:
            user.current_session_id = None
            await user.save()
        
        logger.info(f"Ended {len(active_sessions)} sessions for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error ending sessions for user {user_id}: {e}")
        return False

async def load_session_history(session_id: str, limit: int = 50) -> List[Dict]:
    """Load conversation history for a session."""
    if not is_connected():
        return []
    
    try:
        messages = await ChatMessage.find(
            ChatMessage.session_id == session_id
        ).sort(ChatMessage.timestamp).limit(limit).to_list()
        
        history = []
        for msg in messages:
            history.append({
                "user": msg.message,
                "assistant": msg.response,
                "timestamp": msg.timestamp
            })
        
        logger.debug(f"Loaded {len(history)} messages for session {session_id}")
        return history
        
    except Exception as e:
        logger.error(f"Error loading session history: {e}")
        return []

async def ensure_active_session(user_id: str) -> str:
    """Ensure user has an active session, create one if needed."""
    session = await get_active_session(user_id)
    if session:
        return session.session_id
    
    # Create new session
    session_id = await create_session(user_id)
    return session_id

async def start_new_session(username: str) -> str:
    """Start a new conversation session for the user"""
    if not is_connected():
        return "Database not connected - cannot create session"
    
    try:
        # Normalize username
        username = username.strip().lower()
        
        # Create new session (this will end existing ones)
        session_id = await create_session(username)
        logger.info(f"Started new session {session_id} for user {username}")
        return f"✅ New conversation session started!\nSession ID: {session_id[:12]}...\n\nYour chat history has been cleared. You can start a fresh conversation!"
        
    except Exception as e:
        logger.error(f"Error starting new session for {username}: {e}")
        return "❌ Error starting new session"