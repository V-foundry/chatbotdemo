import asyncio
import json
import re
from typing import List
from datetime import datetime, UTC
from models import ChatMessage, Session
from database import is_connected
from services.user_service import store_user_facts, get_relevant_facts
from services.session_service import load_session_history, ensure_active_session
from services.bot_service import get_user_bot
from loguru import logger
import google.generativeai as genai

# Initialize the model for chat responses
model = genai.GenerativeModel("gemini-2.0-flash")

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

async def chat(message: str, history: list, user_id: str = "default_user"):
    """Main chat function that processes user messages and generates responses"""
    try:
        logger.info(f"Processing chat message for user: {user_id}")
        logger.debug(f"Message: {message}")
        
        # Extract and store facts from user message
        if is_connected():
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
        if is_connected():
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
        if is_connected():
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