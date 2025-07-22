import os
import asyncio
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

class ChatMessage(Document):
    user_id: str  # This will be the username
    message: str
    response: str
    timestamp: datetime = datetime.now(UTC)

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
        await beanie.init_beanie(database=client[DATABASE_NAME], document_models=[User, ChatMessage])
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
            # Update last login
            existing_user.last_login = datetime.now(UTC)
            await existing_user.save()
            logger.info(f"User logged in: {username}")
            return True, f"Welcome back, {existing_user.display_name or username}!"
        else:
            # Create new user
            new_user = User(
                username=username,
                display_name=username.title()
            )
            await new_user.insert()
            logger.info(f"New user registered: {username}")
            return True, f"Welcome, {username.title()}! Your account has been created."
            
    except Exception as e:
        logger.error(f"Error during login/registration: {e}")
        return False, "An error occurred during login"

async def get_user_stats(username: str) -> str:
    """Get user statistics and facts count"""
    if not mongodb_connected:
        return "Database not connected"
    
    try:
        user = await User.find_one(User.username == username)
        if not user:
            return "User not found"
        
        # Count messages and facts using Beanie's count method
        message_count = await ChatMessage.find(ChatMessage.user_id == username).count()
        facts_count = user.facts_metadata.get("total_facts", 0)
        
        stats = f"""
**User Profile:**
- **Username:** {user.username}
- **Display Name:** {user.display_name}
- **Member Since:** {user.created_at.strftime('%Y-%m-%d')}
- **Last Login:** {user.last_login.strftime('%Y-%m-%d %H:%M')}
- **Total Messages:** {message_count}
- **Stored Facts:** {facts_count}
"""
        return stats
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return "Error retrieving user statistics"

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
        
        # Build context with conversation history and user facts
        context = ""
        if mongodb_connected:
            # Get relevant user facts
            user_facts_context = await get_relevant_facts(user_id, message)
            
            # Retrieve previous messages for conversation context
            previous_messages = await ChatMessage.find(ChatMessage.user_id == user_id).sort(-ChatMessage.timestamp).to_list()
            logger.debug(f"Retrieved {len(previous_messages)} previous messages for context")
            
            conversation_context = "\n".join([f"User: {msg.message}\nAI: {msg.response}" for msg in previous_messages[-5:]])  # Last 5 for brevity
            
            # Combine facts and conversation context
            if user_facts_context:
                context = f"{user_facts_context}\n\nRecent conversation:\n{conversation_context}"
            else:
                context = f"Recent conversation:\n{conversation_context}"
                
        else:
            logger.warning("MongoDB not connected, using session history for context")
            # Use gradio history as fallback
            context = "Recent conversation:\n" + "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-5:]] if history else [])

        # Create prompt with enhanced context
        prompt = f"""You are a helpful AI assistant. Use the information below to provide personalized, contextual responses.

{context}

Current user message: {message}

AI Response:"""

        # Generate response asynchronously
        logger.info("Generating response with Gemini API...")
        generated = await asyncio.to_thread(model.generate_content, prompt)
        response = generated.text
        logger.debug(f"Generated response: {response[:100]}...")  # Log first 100 chars

        # Store the new message if MongoDB is connected
        if mongodb_connected:
            logger.info("Storing message in database...")
            new_msg = ChatMessage(user_id=user_id, message=message, response=response)
            await new_msg.insert()
            logger.success("Message stored successfully")
        else:
            logger.warning("MongoDB not connected, message not stored")

        return response
        
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        return "Sorry, I encountered an error while processing your message. Please try again."

async def chat_interface_handler(message: str, history: list, user: str):
    """Handler function for the chat interface"""
    return await chat(message, history, user)

# Login/Registration Interface Functions
async def handle_login(username: str):
    """Handle user login/registration"""
    success, message = await login_or_register_user(username)
    if success:
        return (
            gr.update(visible=False),  # Hide login tab
            gr.update(visible=True),   # Show chat tab
            gr.update(visible=True),   # Show profile tab
            message,                   # Login message
            username,                  # Set current user
            gr.update(value=""),       # Clear chat history
            await get_user_stats(username)  # Load user stats
        )
    else:
        return (
            gr.update(visible=True),   # Keep login tab visible
            gr.update(visible=False),  # Hide chat tab
            gr.update(visible=False),  # Hide profile tab
            message,                   # Error message
            "",                        # No user set
            gr.update(value=""),       # Clear chat history
            ""                         # No stats
        )

async def handle_logout():
    """Handle user logout"""
    return (
        gr.update(visible=True),   # Show login tab
        gr.update(visible=False),  # Hide chat tab
        gr.update(visible=False),  # Hide profile tab
        "Logged out successfully", # Logout message
        "",                        # Clear current user
        gr.update(value=""),       # Clear chat history
        ""                         # Clear stats
    )

async def refresh_user_stats(username: str):
    """Refresh user statistics"""
    if username:
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
        
        # Chat Tab
        with gr.TabItem("💬 Chat", id="chat", visible=False) as chat_tab:
            with gr.Row():
                gr.Markdown("## Chat with your AI Assistant")
                logout_btn = gr.Button("🚪 Logout", variant="secondary", size="sm")
            
            chatbot = gr.ChatInterface(
                fn=chat_interface_handler,
                additional_inputs=[current_user],
                chatbot=gr.Chatbot(height=500, type="messages")
            )
        
        # Profile Tab
        with gr.TabItem("👤 Profile", id="profile", visible=False) as profile_tab:
            with gr.Row():
                gr.Markdown("## User Profile")
                refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary", size="sm")
            
            user_stats = gr.Markdown("")
    
    # Event Handlers
    login_btn.click(
        fn=handle_login,
        inputs=[username_input],
        outputs=[login_tab, chat_tab, profile_tab, login_message, current_user, chatbot.chatbot, user_stats]
    )
    
    logout_btn.click(
        fn=handle_logout,
        outputs=[login_tab, chat_tab, profile_tab, login_message, current_user, chatbot.chatbot, user_stats]
    )
    
    refresh_btn.click(
        fn=refresh_user_stats,
        inputs=[current_user],
        outputs=[user_stats]
    )
    
    # Auto-login on username enter
    username_input.submit(
        fn=handle_login,
        inputs=[username_input],
        outputs=[login_tab, chat_tab, profile_tab, login_message, current_user, chatbot.chatbot, user_stats]
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
