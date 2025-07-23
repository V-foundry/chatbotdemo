import uuid
import re
import asyncio
from typing import Optional
from datetime import datetime, UTC
from models import User, Bot
from database import is_connected
from loguru import logger
import google.generativeai as genai

# Initialize the model for bot customization
model = genai.GenerativeModel("gemini-2.0-flash")

async def create_user_bot(user_id: str, bot_name: str = None) -> str:
    """Create a personalized bot for the user with a default system prompt."""
    if not is_connected():
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
    if not is_connected():
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
    if not is_connected():
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
    if not is_connected():
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
    if not is_connected():
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