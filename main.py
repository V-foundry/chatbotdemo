import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr

# Import configuration and database
from config import PORT, HOST
from database import init_mongo
from models import User, Bot, Session, ChatMessage

# Import services
from services.user_service import login_or_register_user, get_user_stats
from services.bot_service import check_bot_needs_introduction, customize_bot_from_introduction, fix_existing_bot_prompt
from services.session_service import start_new_session
from services.chat_service import chat

from loguru import logger

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

async def handle_fix_bot_prompt(username: str) -> str:
    """Handle fixing bot prompt formatting issues"""
    # Normalize username
    username = username.strip().lower()
    
    # Fix the bot prompt
    success = await fix_existing_bot_prompt(username)
    
    if success:
        return "✅ Bot prompt has been cleaned and fixed!\n\nYour AI companion's responses should now be properly formatted without extra quotes or formatting issues."
    else:
        return "❌ Error fixing bot prompt or no changes were needed."

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
    logger.info(f"Starting uvicorn server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
