import os
import google.generativeai as genai
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure loguru logger
logger.add("chatbot.log", rotation="1 MB", retention="7 days", level="INFO")

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
genai.configure(api_key=GEMINI_API_KEY)
logger.info("Gemini API configured")

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "chatbot_db")

logger.info(f"MongoDB URI: {MONGODB_URI}")
logger.info(f"Database Name: {DATABASE_NAME}")

# Server Configuration
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

# Global variable to track MongoDB connection status
mongodb_connected = False 