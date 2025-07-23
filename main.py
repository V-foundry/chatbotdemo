import os
import asyncio
import uuid
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
import beanie
import motor.motor_asyncio
from beanie import Document
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
import google.generativeai as genai
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure loguru logger
logger.add("chatbot.log", rotation="1 MB", retention="7 days", level="INFO")

# Placeholder for API key - replace with your actual Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "your-api-key-here"))
logger.info("Gemini API configured")

model = genai.GenerativeModel("gemini-2.0-flash")

# MongoDB connection string - adjust as needed (e.g., for local or Atlas)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "chatbot_db")

logger.info(f"MongoDB URI: {MONGODB_URI}")
logger.info(f"Database Name: {DATABASE_NAME}")

# Global variable to track MongoDB connection status
mongodb_connected = False

class User(Document):
    username: str
    display_name: Optional[str] = None
    created_at: datetime = datetime.now(UTC)
    last_login: datetime = datetime.now(UTC)
    is_active: bool = True
    
    # Session and Bot Management
    bot_id: Optional[str] = None  # References user's personalized bot
    current_session_id: Optional[str] = None  # Active session tracking
    
    # Flexible facts storage as JSON
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

async def init_mongo():
    global mongodb_connected
    try:
        logger.info("Initializing MongoDB connection...")
        client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGODB_URI, 
            serverSelectionTimeoutMS=5000  # 5 second timeout
        )
        # Test the connection
        await client.admin.command('ping')
        await beanie.init_beanie(database=client[DATABASE_NAME], document_models=[User, Bot, Session, ChatMessage])
        mongodb_connected = True
        logger.success("MongoDB connection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB connection: {e}")
        logger.warning("Application will continue without database functionality")
        mongodb_connected = False

async def login_or_register_user(username: str) -> tuple[bool, str]:
    """Login existing user or register new user. Returns (success, message)"""
    if not mongodb_connected:
        return False, "Database not connected"
    
    if not username or not username.strip():
        return False, "Please enter a username"
    
    username = username.strip().lower()
    
    try:
        # Check if user exists
        existing_user = await User.find_one(User.username == username)
        
        if existing_user:
            # Update last login and create new session
            existing_user.last_login = datetime.now(UTC)
            await existing_user.save()
            
            # Create new session for returning user
            session_id = await create_session(username)
            
            logger.info(f"User logged in: {username}")
            return True, f"Welcome back, {existing_user.display_name or username}!"
        else:
            # Create new user
            new_user = User(
                username=username,
                display_name=username.title()
            )
            await new_user.insert()
            
            # Create personalized bot for new user
            bot_id = await create_user_bot(username)
            
            # Create initial session
            session_id = await create_session(username, bot_id)
            
            logger.info(f"New user registered: {username} with bot: {bot_id}")
            return True, f"Welcome, {username.title()}! Please introduce yourself to create your personalized AI companion."
            
    except Exception as e:
        logger.error(f"Error during login/registration: {e}")
        return False, "An error occurred during login"

async def get_user_stats(username: str) -> str:
    """Get user statistics, facts count, session info, and bot details"""
    if not mongodb_connected:
        return "Database not connected"
    
    try:
        logger.info(f"Getting user stats for: {username}")
        user = await User.find_one(User.username == username)
        if not user:
            logger.warning(f"User not found: {username}")
            return "User not found"
        
        # Count messages and facts using Beanie's count method
        message_count = await ChatMessage.find(ChatMessage.user_id == username).count()
        facts_count = user.facts_metadata.get("total_facts", 0)
        
        # Get bot information
        bot_info = "No bot assigned"
        bot_status = "Not customized"
        if user.bot_id:
            user_bot = await Bot.find_one(Bot.bot_id == user.bot_id)
            if user_bot:
                bot_info = f"{user_bot.bot_name} (ID: {user_bot.bot_id[:12]}...)"
                bot_status = "✅ Personalized" if user_bot.prompt_customized else "⏳ Awaiting introduction"
        
        # Get session information
        session_info = "No active session"
        session_message_count = 0
        if user.current_session_id:
            active_session = await Session.find_one(Session.session_id == user.current_session_id)
            if active_session:
                session_info = f"Active (ID: {active_session.session_id[:12]}...)"
                session_message_count = active_session.metadata.get("message_count", 0)
                session_created = active_session.created_at.strftime('%Y-%m-%d %H:%M')
                session_info += f"\nCreated: {session_created}"
        
        # Count total sessions
        total_sessions = await Session.find(Session.user_id == username).count()
        
        stats = f"""
**User Profile:**
- **Username:** {user.username}
- **Display Name:** {user.display_name}
- **Member Since:** {user.created_at.strftime('%Y-%m-%d')}
- **Last Login:** {user.last_login.strftime('%Y-%m-%d %H:%M')}

**Bot Information:**
- **Personal AI:** {bot_info}
- **Status:** {bot_status}

**Session Information:**
- **Current Session:** {session_info}
- **Messages in Session:** {session_message_count}
- **Total Sessions:** {total_sessions}

**Data Summary:**
- **Total Messages:** {message_count}
- **Stored Facts:** {facts_count}
"""
        return stats
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return "Error retrieving user statistics"

async def start_new_session(username: str) -> str:
    """Start a new conversation session for the user"""
    if not mongodb_connected:
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

async def handle_fix_bot_prompt(username: str) -> str:
    """Handle fixing bot prompt formatting issues"""
    if not mongodb_connected:
        return "Database not connected - cannot fix bot prompt"
    
    try:
        # Normalize username
        username = username.strip().lower()
        
        # Fix the bot prompt
        success = await fix_existing_bot_prompt(username)
        
        if success:
            return "✅ Bot prompt has been cleaned and fixed!\n\nYour AI companion's responses should now be properly formatted without extra quotes or formatting issues."
        else:
            return "❌ Error fixing bot prompt or no changes were needed."
            
    except Exception as e:
        logger.error(f"Error handling bot prompt fix for {username}: {e}")
        return "❌ Error fixing bot prompt"

async def extract_user_facts(message: str, user_id: str) -> List[dict]:
    """Extract important facts about the user from their message using AI."""
    try:
        extraction_prompt = f"""
You are a fact extraction AI. Analyze the user message and extract personal facts.

User message: "{message}"

IMPORTANT: You MUST respond with ONLY valid JSON. No other text before or after.

Extract facts in this exact format:
[
    {{"category": "personal", "fact": "Name is John", "confidence": 0.9}},
    {{"category": "work", "fact": "Works as engineer", "confidence": 0.8}}
]

Categories: personal, preferences, work, family, health, goals, experiences, other
Confidence: 0.9+ for explicit, 0.7-0.9 for implied, 0.5-0.7 for uncertain

Rules:
- Extract clear factual information about the user only
- Don't extract temporary states or opinions about others
- Return empty array [] if no meaningful facts
- RESPOND WITH ONLY JSON - NO OTHER TEXT

JSON Response:"""

        logger.info("Extracting facts from user message...")
        response = await asyncio.to_thread(model.generate_content, extraction_prompt)
        
        # Clean and parse JSON response
        import json
        import re
        
        response_text = response.text.strip()
        logger.debug(f"Raw AI response: {response_text[:200]}...")
        
        # Try to extract JSON from response (in case there's extra text)
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        else:
            json_text = response_text
        
        try:
            facts = json.loads(json_text)
            if isinstance(facts, list):
                # Validate each fact has required fields
                valid_facts = []
                for fact in facts:
                    if isinstance(fact, dict) and "category" in fact and "fact" in fact:
                        # Ensure confidence is set
                        fact["confidence"] = fact.get("confidence", 0.7)
                        valid_facts.append(fact)
                
                logger.info(f"Extracted {len(valid_facts)} valid facts from message")
                return valid_facts
            else:
                logger.warning("Response is not a list, trying alternative parsing...")
                return []
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            logger.debug(f"Attempted to parse: {json_text[:100]}...")
            
            # Fallback: Try to extract facts using simple patterns
            fallback_facts = []
            
            # Simple pattern matching for common fact patterns
            if "name is" in message.lower():
                name_match = re.search(r'(?:my )?name is (\w+)', message.lower())
                if name_match:
                    fallback_facts.append({
                        "category": "personal",
                        "fact": f"Name is {name_match.group(1).title()}",
                        "confidence": 0.9
                    })
            
            if "from" in message.lower() and any(word in message.lower() for word in ["live", "come", "born"]):
                location_match = re.search(r'(?:from|live in|come from) (\w+)', message.lower())
                if location_match:
                    fallback_facts.append({
                        "category": "personal",
                        "fact": f"From {location_match.group(1).title()}",
                        "confidence": 0.8
                    })
            
            if "work" in message.lower() or "job" in message.lower():
                work_match = re.search(r'(?:work as|job as|am a) (\w+)', message.lower())
                if work_match:
                    fallback_facts.append({
                        "category": "work",
                        "fact": f"Works as {work_match.group(1)}",
                        "confidence": 0.8
                    })
            
            if fallback_facts:
                logger.info(f"Used fallback extraction, found {len(fallback_facts)} facts")
                return fallback_facts
            
            return []
            
    except Exception as e:
        logger.error(f"Error extracting facts: {e}")
        return []

async def store_user_facts(facts: List[dict], user_id: str, source_message: str):
    """Store extracted facts in the database."""
    if not mongodb_connected or not facts:
        return
    
    try:
        user = await User.find_one(User.username == user_id)
        if not user:
            logger.warning(f"User not found for storing facts: {user_id}")
            return

        for fact_data in facts:
            category = fact_data["category"]
            fact_text = fact_data["fact"]
            confidence = fact_data.get("confidence", 0.7)

            # Update or add fact to user's facts
            if category in user.facts:
                user.facts[category][fact_text] = {
                    "confidence": confidence,
                    "source_message": source_message,
                    "timestamp": datetime.now(UTC)
                }
            else:
                user.facts[category] = {fact_text: {
                    "confidence": confidence,
                    "source_message": source_message,
                    "timestamp": datetime.now(UTC)
                }}

            # Update metadata
            user.facts_metadata["total_facts"] += 1
            user.facts_metadata["last_updated"] = datetime.now(UTC)
            user.facts_metadata["sources"].append({
                "message_id": source_message,
                "timestamp": datetime.now(UTC)
            })

        await user.save()
        logger.info(f"Stored {len(facts)} facts for user: {user_id}")
        
    except Exception as e:
        logger.error(f"Error storing facts: {e}")

async def get_relevant_facts(user_id: str, current_message: str = "") -> str:
    """Retrieve relevant user facts to include in conversation context."""
    if not mongodb_connected:
        return ""
    
    try:
        user = await User.find_one(User.username == user_id)
        if not user:
            return "User not found"

        # Get all facts from the user's facts field
        all_facts = []
        for category, facts_dict in user.facts.items():
            for fact_text, fact_data in facts_dict.items():
                all_facts.append({
                    "category": category,
                    "fact": fact_text,
                    "confidence": fact_data["confidence"],
                    "source_message": fact_data["source_message"],
                    "timestamp": fact_data["timestamp"]
                })

        # Sort facts by timestamp (newest first)
        all_facts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        if not all_facts:
            return ""
        
        # Organize facts by category
        facts_by_category = {}
        for fact in all_facts:
            if fact["category"] not in facts_by_category:
                facts_by_category[fact["category"]] = []
            facts_by_category[fact["category"]].append(fact["fact"])
        
        # Format facts for context
        context_parts = ["Known facts about the user:"]
        for category, facts in facts_by_category.items():
            context_parts.append(f"\n{category.title()}:")
            for fact in facts[:3]:  # Limit to top 3 facts per category
                context_parts.append(f"- {fact}")
        
        context = "\n".join(context_parts)
        logger.debug(f"Retrieved {len(all_facts)} facts for context")
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving facts: {e}")
        return ""

# ============================================================================
# BOT MANAGEMENT FUNCTIONS
# ============================================================================

async def create_user_bot(user_id: str, bot_name: str = None) -> str:
    """Create a personalized bot for the user with a default system prompt."""
    if not mongodb_connected:
        return None
    
    try:
        # Generate unique bot_id
        bot_id = f"bot_{user_id}_{uuid.uuid4().hex[:8]}"
        
        # Create default personalized system prompt
        user = await User.find_one(User.username == user_id)
        display_name = user.display_name if user else user_id.title()
        
        default_prompt = f"""You are a personal AI companion specifically designed for {display_name}. Your role is to be a warm, understanding, and caring conversational partner who gets to know them on a personal level.

Your behavior and approach:
- Be an attentive listener who remembers and references personal details from previous conversations
- Engage in meaningful conversations about their life, experiences, interests, and personal growth
- Ask thoughtful follow-up questions to deepen understanding of who they are
- Maintain a warm, consistent, and personally engaged tone across all interactions
- Focus on companionship, self-reflection, and personal understanding rather than general problem-solving
- Remember that you are designed for personal connection, not as a general-purpose assistant

Important: This is a temporary system prompt that will be personalized once {display_name} completes their introduction. Until then, encourage them to share about themselves so you can provide more tailored companionship.

Be ready to incorporate numerology and spiritual guidance elements in future interactions."""

        # Create bot document
        new_bot = Bot(
            bot_id=bot_id,
            bot_name=bot_name or f"{display_name}'s Assistant",
            system_prompt=default_prompt,
            prompt_customized=False  # Explicitly set to False for new bots
        )
        await new_bot.insert()
        
        # Update user with bot_id
        if user:
            user.bot_id = bot_id
            await user.save()
        
        logger.info(f"Created bot {bot_id} for user {user_id}")
        return bot_id
        
    except Exception as e:
        logger.error(f"Error creating bot for user {user_id}: {e}")
        return None

async def get_user_bot(user_id: str) -> Optional[Bot]:
    """Retrieve user's bot configuration."""
    if not mongodb_connected:
        return None
    
    try:
        user = await User.find_one(User.username == user_id)
        if not user or not user.bot_id:
            return None
        
        bot = await Bot.find_one(Bot.bot_id == user.bot_id)
        return bot
        
    except Exception as e:
        logger.error(f"Error retrieving bot for user {user_id}: {e}")
        return None

async def ensure_user_has_bot(user_id: str) -> str:
    """Ensure user has a bot, create one if needed."""
    bot = await get_user_bot(user_id)
    if bot:
        return bot.bot_id
    
    # Create a new bot for the user
    bot_id = await create_user_bot(user_id)
    return bot_id

def clean_system_prompt(raw_prompt: str) -> str:
    """Clean up AI-generated system prompt to remove unwanted formatting."""
    if not raw_prompt or raw_prompt.strip() == "":
        return ""
    
    # Remove common meta-commentary patterns for system prompts
    cleaned = raw_prompt
    
    # Remove introductory explanations
    patterns_to_remove = [
        r"^.*?here'?s a system prompt.*?:?\s*",
        r"^.*?system prompt instructions?.*?:?\s*",
        r"^\*\*System Prompt.*?\*\*:?\s*",
        r"^System Prompt.*?:?\s*",
        r"^.*?instructions? for.*?:?\s*",
        r"^Okay,?\s*",
        r"^Here'?s?\s*",
        r"^Based on.*?introduction.*?:?\s*",
    ]
    
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove surrounding quotes if the entire content is quoted
    cleaned = cleaned.strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    elif cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1]
    
    # Remove markdown formatting but preserve structure
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Bold
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Italic
    
    # Clean up extra whitespace and newlines but preserve structure
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple newlines
    cleaned = re.sub(r'^\s+', '', cleaned, flags=re.MULTILINE)  # Leading whitespace
    cleaned = cleaned.strip()
    
    # Ensure it doesn't start with code block markers
    if cleaned.startswith('```') or cleaned.startswith('"""') or cleaned.startswith("'''"):
        lines = cleaned.split('\n')
        if len(lines) > 1:
            cleaned = '\n'.join(lines[1:])
            if cleaned.endswith('```') or cleaned.endswith('"""') or cleaned.endswith("'''"):
                cleaned = '\n'.join(cleaned.split('\n')[:-1])
    
    # Final check - ensure we have actual content
    if not cleaned or cleaned.strip() == "":
        return ""
    
    return cleaned.strip()

async def fix_existing_bot_prompt(user_id: str) -> bool:
    """Fix existing bot system prompt that may have formatting issues."""
    if not mongodb_connected:
        return False
    
    try:
        user = await User.find_one(User.username == user_id)
        if not user or not user.bot_id:
            return False
        
        bot = await Bot.find_one(Bot.bot_id == user.bot_id)
        if not bot:
            return False
        
        # Clean the existing system prompt
        original_prompt = bot.system_prompt
        cleaned_prompt = clean_system_prompt(original_prompt)
        
        # Only update if the prompt was actually changed
        if cleaned_prompt != original_prompt:
            bot.system_prompt = cleaned_prompt
            bot.updated_at = datetime.now(UTC)
            await bot.save()
            logger.info(f"Fixed system prompt for user {user_id}")
            return True
        else:
            logger.info(f"System prompt for user {user_id} was already clean")
            return True
            
    except Exception as e:
        logger.error(f"Error fixing bot prompt for user {user_id}: {e}")
        return False

async def customize_bot_from_introduction(user_id: str, introduction: str) -> bool:
    """Customize user's bot system prompt based on their introduction."""
    if not mongodb_connected:
        logger.error("MongoDB not connected for bot customization")
        return False
    
    try:
        # Get user and bot
        user = await User.find_one(User.username == user_id)
        if not user or not user.bot_id:
            logger.error(f"User {user_id} not found or has no bot_id")
            return False
        
        bot = await Bot.find_one(Bot.bot_id == user.bot_id)
        if not bot:
            logger.error(f"Bot {user.bot_id} not found for user {user_id}")
            return False
        
        if bot.prompt_customized:
            logger.warning(f"Bot {bot.bot_id} already customized for user {user_id}")
            return False  # Already customized
        
        display_name = user.display_name or user_id.title()
        
        # Create customization prompt for AI
        customization_prompt = f"""You are creating a system prompt that will instruct an AI on how to behave when talking to {display_name}.

User Introduction: "{introduction}"

Write ONLY the system prompt instructions (no explanations, no meta-commentary, no quotes). The system prompt should:

1. Be written in second person addressing the AI (e.g., "You are...", "Your role is...", "Remember to...")
2. Define the AI's role as a personal companion for {display_name}
3. Include key details from their introduction (interests, background, personality, goals)
4. Instruct the AI to be warm, caring, and personally engaged
5. Focus on meaningful conversations and personal growth, not general problem-solving
6. Prepare for future numerology/spiritual guidance capabilities
7. Instruct the AI to remember and reference personal details appropriately

Example format: "You are a personal AI companion for [name]. They are [background details]. Your role is to [behavior instructions]. Remember to [specific guidance]..."

Keep it under 200 words and make it clear, actionable instructions for the AI.

System prompt instructions:"""

        logger.info("Generating personalized system prompt...")
        generated = await asyncio.to_thread(model.generate_content, customization_prompt)
        
        if not generated or not generated.text:
            logger.error("Failed to generate system prompt - empty response")
            return False
            
        raw_prompt = generated.text.strip()
        logger.debug(f"Raw generated prompt: {raw_prompt[:200]}...")
        
        # Clean up the generated prompt to remove any unwanted formatting
        personalized_prompt = clean_system_prompt(raw_prompt)
        
        if not personalized_prompt:
            logger.error("System prompt became empty after cleaning - using fallback")
            # Create a basic personalized fallback prompt
            personalized_prompt = f"""You are a personal AI companion for {display_name}. Based on their introduction, engage with them in a warm, caring manner about their interests and experiences. Focus on meaningful conversations and personal growth rather than general problem-solving. Remember details they share and reference them in future conversations. Be supportive and understanding in your interactions."""
        
        logger.info(f"Final system prompt: {personalized_prompt[:200]}...")
        
        # Update bot with personalized prompt
        bot.system_prompt = personalized_prompt
        bot.prompt_customized = True
        bot.updated_at = datetime.now(UTC)
        await bot.save()
        
        logger.info(f"Customized bot prompt for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error customizing bot for user {user_id}: {e}")
        return False

async def check_bot_needs_introduction(user_id: str) -> bool:
    """Check if user's bot needs introduction/customization."""
    if not mongodb_connected:
        return False
    
    try:
        user = await User.find_one(User.username == user_id)
        if not user or not user.bot_id:
            logger.warning(f"User {user_id} has no bot assigned")
            return False
        
        bot = await Bot.find_one(Bot.bot_id == user.bot_id)
        if not bot:
            logger.warning(f"Bot {user.bot_id} not found for user {user_id}")
            return False
        
        needs_intro = not bot.prompt_customized
        logger.info(f"User {user_id} bot customization check: needs_intro={needs_intro}, prompt_customized={bot.prompt_customized}")
        
        # Return True if bot prompt hasn't been customized yet
        return needs_intro
        
    except Exception as e:
        logger.error(f"Error checking bot customization status: {e}")
        return False

# ============================================================================
# SESSION MANAGEMENT FUNCTIONS
# ============================================================================

async def create_session(user_id: str, bot_id: str = None) -> str:
    """Create a new conversation session for the user."""
    if not mongodb_connected:
        return f"session_{user_id}_default"
    
    try:
        # End any existing active session
        await end_user_sessions(user_id)
        
        # Ensure user has a bot
        if not bot_id:
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
    if not mongodb_connected:
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
    if not mongodb_connected:
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
    if not mongodb_connected:
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FastAPI application...")
    await init_mongo()
    logger.success("Application startup completed")
    yield
    # Shutdown
    logger.info("Application shutdown")

app = FastAPI(lifespan=lifespan)

# Add root redirect to chat interface
@app.get("/")
async def root():
    return RedirectResponse(url="/chat")

async def chat(message: str, history: list, user_id: str = "default_user"):
    try:
        logger.info(f"Processing chat message for user: {user_id}")
        logger.debug(f"Message: {message}")
        
        # Extract and store facts from user message
        if mongodb_connected:
            extracted_facts = await extract_user_facts(message, user_id)
            if extracted_facts:
                await store_user_facts(extracted_facts, user_id, message)
        
        # Get or create active session
        session_id = await ensure_active_session(user_id)
        
        # Get user's bot configuration
        user_bot = await get_user_bot(user_id)
        system_prompt = "You are a helpful AI assistant."  # Default fallback
        bot_id = "default_bot"
        
        if user_bot:
            system_prompt = user_bot.system_prompt
            bot_id = user_bot.bot_id
            logger.info(f"Using custom bot: {user_bot.bot_name}")
        
        # Build context with session history and user facts
        context = ""
        if mongodb_connected:
            # Get relevant user facts
            user_facts_context = await get_relevant_facts(user_id, message)
            
            # Load session-based conversation history
            session_history = await load_session_history(session_id, limit=10)
            
            # Format conversation context from session
            if session_history:
                conversation_lines = []
                for conv in session_history[-5:]:  # Last 5 conversations
                    conversation_lines.append(f"User: {conv['user']}")
                    conversation_lines.append(f"Assistant: {conv['assistant']}")
                conversation_context = "\n".join(conversation_lines)
            else:
                conversation_context = "No previous conversation in this session."
            
            # Combine facts and conversation context
            if user_facts_context:
                context = f"{user_facts_context}\n\nRecent session conversation:\n{conversation_context}"
            else:
                context = f"Recent session conversation:\n{conversation_context}"
                
        else:
            logger.warning("MongoDB not connected, using gradio history for context")
            # Use gradio history as fallback
            context = "Recent conversation:\n" + "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-5:]] if history else [])

        # Create prompt with personalized system prompt and context
        prompt = f"""{system_prompt}

Context information:
{context}

Current user message: {message}

Please provide a helpful and personalized response:"""

        # Generate response asynchronously
        logger.info("Generating response with Gemini API...")
        generated = await asyncio.to_thread(model.generate_content, prompt)
        response = generated.text
        logger.debug(f"Generated response: {response[:100]}...")  # Log first 100 chars

        # Store the new message with session and bot information
        if mongodb_connected:
            logger.info("Storing message in database...")
            new_msg = ChatMessage(
                user_id=user_id, 
                session_id=session_id,
                bot_id=bot_id,
                message=message, 
                response=response,
                metadata={
                    "facts_extracted": len(extracted_facts) if extracted_facts else 0,
                    "response_time": None,  # Could add timing here
                    "tokens_used": None    # Could add token counting here
                }
            )
            await new_msg.insert()
            
            # Update session metadata
            session = await Session.find_one(Session.session_id == session_id)
            if session:
                session.metadata["message_count"] = session.metadata.get("message_count", 0) + 1
                session.updated_at = datetime.now(UTC)
                await session.save()
            
            logger.success("Message stored successfully")
        else:
            logger.warning("MongoDB not connected, message not stored")

        return response
        
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        return "Sorry, I encountered an error while processing your message. Please try again."

async def chat_interface_handler(message: str, history: list, user: str):
    """Handler function for the chat interface"""
    # Ensure consistent username case
    user_normalized = user.strip().lower() if user else "default_user"
    return await chat(message, history, user_normalized)

# Login/Registration Interface Functions
async def handle_login(username: str):
    """Handle user login/registration"""
    # Normalize username to lowercase
    username = username.strip().lower()
    
    success, message = await login_or_register_user(username)
    if success:
        # Check if user needs to complete introduction
        needs_intro = await check_bot_needs_introduction(username)
        logger.info(f"Login successful for {username}, needs_intro={needs_intro}")
        
        if needs_intro:
            logger.info(f"Redirecting {username} to introduction tab")
            return (
                gr.update(visible=False),  # Hide login tab
                gr.update(visible=False),  # Hide chat tab (until intro complete)
                gr.update(visible=False),  # Hide profile tab (until intro complete)
                gr.update(visible=True),   # Show introduction tab
                message,                   # Login message
                username,                  # Set current user
                gr.update(value=[]),       # Clear chat history completely
                await get_user_stats(username),  # Load user stats
                "",                        # Clear session status
                "",                        # Clear introduction status
                ""                         # Clear bot fix status
            )
        else:
            logger.info(f"Redirecting {username} to chat tab")
            return (
                gr.update(visible=False),  # Hide login tab
                gr.update(visible=True),   # Show chat tab
                gr.update(visible=True),   # Show profile tab
                gr.update(visible=False),  # Hide introduction tab
                message,                   # Login message
                username,                  # Set current user
                gr.update(value=[]),       # Clear chat history completely
                await get_user_stats(username),  # Load user stats
                "",                        # Clear session status
                "",                        # Clear introduction status
                ""                         # Clear bot fix status
            )
    else:
        return (
            gr.update(visible=True),   # Keep login tab visible
            gr.update(visible=False),  # Hide chat tab
            gr.update(visible=False),  # Hide profile tab
            gr.update(visible=False),  # Hide introduction tab
            message,                   # Error message
            "",                        # No user set
            gr.update(value=[]),       # Clear chat history completely
            "",                        # No stats
            "",                        # No session status
            "",                        # No introduction status
            ""                         # No bot fix status
        )

async def handle_logout():
    """Handle user logout"""
    return (
        gr.update(visible=True),   # Show login tab
        gr.update(visible=False),  # Hide chat tab
        gr.update(visible=False),  # Hide profile tab
        gr.update(visible=False),  # Hide introduction tab
        "Logged out successfully", # Logout message
        "",                        # Clear current user
        gr.update(value=[]),       # Clear chat history completely
        "",                        # Clear stats
        "",                        # Clear session status
        "",                        # Clear introduction status
        ""                         # Clear bot fix status
    )

async def handle_introduction(introduction: str, username: str):
    """Handle user introduction and customize their bot"""
    # Normalize username
    username = username.strip().lower()
    
    if not introduction.strip():
        return (
            gr.update(visible=True),   # Keep introduction tab visible
            gr.update(visible=False),  # Keep chat tab hidden
            gr.update(visible=False),  # Keep profile tab hidden
            "Please provide an introduction about yourself."
        )
    
    logger.info(f"Processing introduction for user {username}")
    
    # Customize bot based on introduction
    success = await customize_bot_from_introduction(username, introduction.strip())
    
    if success:
        logger.info(f"Introduction completed successfully for {username}")
        return (
            gr.update(visible=False),  # Hide introduction tab
            gr.update(visible=True),   # Show chat tab
            gr.update(visible=True),   # Show profile tab
            f"✅ Thank you for introducing yourself! Your personal AI companion has been customized just for you. You can now start chatting!"
        )
    else:
        logger.error(f"Introduction failed for {username}")
        return (
            gr.update(visible=True),   # Keep introduction tab visible
            gr.update(visible=False),  # Keep chat tab hidden
            gr.update(visible=False),  # Keep profile tab hidden
            "❌ There was an error customizing your AI companion. Please try again."
        )

async def refresh_user_stats(username: str):
    """Refresh user statistics"""
    if username:
        # Normalize username
        username = username.strip().lower()
        return await get_user_stats(username)
    return "No user logged in"

# Create Gradio Interface
with gr.Blocks(title="Personal AI Chatbot", theme=gr.themes.Soft()) as demo:
    current_user = gr.State("")
    
    gr.Markdown("# 🤖 Personal AI Chatbot")
    gr.Markdown("Demo Chatbot for NUME")
    
    with gr.Tabs() as tabs:
        # Login Tab
        with gr.TabItem("🔐 Login", id="login", visible=True) as login_tab:
            gr.Markdown("## Welcome! Please enter your username to continue")
            with gr.Row():
                username_input = gr.Textbox(
                    label="Username", 
                    placeholder="Enter your username...",
                    scale=3
                )
                login_btn = gr.Button("Login / Register", variant="primary", scale=1)
            
            login_message = gr.Markdown("")
        
        # Introduction Tab (for new users)
        with gr.TabItem("👋 Introduction", id="introduction", visible=False) as intro_tab:
            gr.Markdown("## Welcome! Let's get to know you better")
            gr.Markdown("""
            To create a personalized AI companion just for you, please introduce yourself! 
            Share whatever feels comfortable - your interests, personality, background, goals, or anything else you'd like your AI companion to know about you.
            
            **Some ideas to include:**
            - Your name and what you like to be called
            - Your interests, hobbies, or passions  
            - Your personality traits or values
            - What you're hoping to get from this AI companion
            - Any personal details you'd like remembered
            """)
            
            introduction_input = gr.Textbox(
                label="Tell us about yourself", 
                placeholder="Hi! I'm... I love... I'm interested in... I hope this AI can help me with...",
                lines=8,
                max_lines=15
            )
            
            intro_submit_btn = gr.Button("Create My Personal AI Companion", variant="primary", size="lg")
            intro_status = gr.Markdown("")
        
        # Chat Tab
        with gr.TabItem("💬 Chat", id="chat", visible=False) as chat_tab:
            with gr.Row():
                gr.Markdown("## Chat with your AI Assistant")
                logout_btn = gr.Button("🚪 Logout", variant="secondary", size="sm")
            
            chatbot = gr.ChatInterface(
                fn=chat_interface_handler,
                additional_inputs=[current_user],
                chatbot=gr.Chatbot(height=500, type="messages", show_copy_button=True)
            )
        
        # Profile Tab
        with gr.TabItem("👤 Profile", id="profile", visible=False) as profile_tab:
            with gr.Row():
                gr.Markdown("## User Profile")
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary", size="sm")
                    new_session_btn = gr.Button("🆕 Start New Session", variant="primary", size="sm")
                    fix_bot_btn = gr.Button("🔧 Fix Bot Prompt", variant="secondary", size="sm")
            
            user_stats = gr.Markdown("")
            session_status = gr.Markdown("")
            bot_fix_status = gr.Markdown("")
    
    # Event Handlers
    login_btn.click(
        fn=handle_login,
        inputs=[username_input],
        outputs=[login_tab, chat_tab, profile_tab, intro_tab, login_message, current_user, chatbot.chatbot, user_stats, session_status, intro_status, bot_fix_status]
    )
    
    logout_btn.click(
        fn=handle_logout,
        outputs=[login_tab, chat_tab, profile_tab, intro_tab, login_message, current_user, chatbot.chatbot, user_stats, session_status, intro_status, bot_fix_status]
    )
    
    intro_submit_btn.click(
        fn=handle_introduction,
        inputs=[introduction_input, current_user],
        outputs=[intro_tab, chat_tab, profile_tab, intro_status]
    )
    
    # Auto-submit introduction on Enter
    introduction_input.submit(
        fn=handle_introduction,
        inputs=[introduction_input, current_user],
        outputs=[intro_tab, chat_tab, profile_tab, intro_status]
    )
    
    refresh_btn.click(
        fn=refresh_user_stats,
        inputs=[current_user],
        outputs=[user_stats]
    )
    
    new_session_btn.click(
        fn=start_new_session,
        inputs=[current_user],
        outputs=[session_status]
    ).then(
        # Clear the chatbot interface after starting new session
        lambda: gr.update(value=[]),
        outputs=[chatbot.chatbot]
    )
    
    fix_bot_btn.click(
        fn=handle_fix_bot_prompt,
        inputs=[current_user],
        outputs=[bot_fix_status]
    )
    
    # Auto-login on username enter
    username_input.submit(
        fn=handle_login,
        inputs=[username_input],
        outputs=[login_tab, chat_tab, profile_tab, intro_tab, login_message, current_user, chatbot.chatbot, user_stats, session_status, intro_status, bot_fix_status]
    )

# Mount Gradio on FastAPI at /chat path
app = gr.mount_gradio_app(app, demo, path="/chat")
logger.info("Gradio interface mounted on FastAPI at /chat")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Railway sets this)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting uvicorn server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
