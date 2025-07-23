import motor.motor_asyncio
import beanie
from config import MONGODB_URI, DATABASE_NAME, mongodb_connected
from models import User, Bot, Session, ChatMessage
from loguru import logger

async def init_mongo():
    """Initialize MongoDB connection and Beanie ODM"""
    global mongodb_connected
    try:
        logger.info("Initializing MongoDB connection...")
        client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGODB_URI, 
            serverSelectionTimeoutMS=5000  # 5 second timeout
        )
        # Test the connection
        await client.admin.command('ping')
        await beanie.init_beanie(
            database=client[DATABASE_NAME], 
            document_models=[User, Bot, Session, ChatMessage]
        )
        mongodb_connected = True
        logger.success("MongoDB connection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB connection: {e}")
        logger.warning("Application will continue without database functionality")
        mongodb_connected = False

def is_connected() -> bool:
    """Check if MongoDB is connected"""
    return mongodb_connected 