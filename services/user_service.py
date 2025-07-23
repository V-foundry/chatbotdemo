from typing import List, Dict, Any
from datetime import datetime, UTC
from models import User, ChatMessage, Session
from database import is_connected
from loguru import logger

async def login_or_register_user(username: str) -> tuple[bool, str]:
    """Login existing user or register new user. Returns (success, message)"""
    if not is_connected():
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
            
            # Import here to avoid circular imports
            from services.session_service import create_session
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
            
            # Import here to avoid circular imports
            from services.bot_service import create_user_bot
            from services.session_service import create_session
            
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
    if not is_connected():
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
        
        # Import here to avoid circular imports
        from models import Bot
        
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

async def store_user_facts(facts: List[dict], user_id: str, source_message: str):
    """Store extracted facts in the database."""
    if not is_connected() or not facts:
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
    if not is_connected():
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