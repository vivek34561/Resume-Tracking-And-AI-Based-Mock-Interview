# auth/google_auth.py

import streamlit as st
from streamlit_google_auth import Authenticate
import os
import json
from pathlib import Path

def init_google_auth():
    """
    Initialize Google OAuth authentication.
    
    Returns:
        Authenticate: Google authenticator instance
    """
    # Get credentials from environment or file
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "client_secret.json")
    
    # Check if credentials file exists
    if not Path(credentials_path).exists():
        # Try to create credentials from environment variables
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")
        
        if client_id and client_secret:
            # Create temporary credentials dict
            credentials = {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uris": [redirect_uri],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
            }
            # Save to temp file
            with open("temp_client_secret.json", "w") as f:
                json.dump(credentials, f)
            credentials_path = "temp_client_secret.json"
        else:
            st.error("❌ Google OAuth credentials not configured!")
            st.info("Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables or provide client_secret.json file.")
            st.stop()
    
    # Initialize authenticator
    try:
        authenticator = Authenticate(
            secret_credentials_path=credentials_path,
            cookie_name='resumate_auth_cookie',
            cookie_key=os.getenv("COOKIE_SECRET_KEY", "your-secret-cookie-key-change-in-production"),
            redirect_uri=os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501"),
        )
        return authenticator
    except Exception as e:
        st.error(f"❌ Failed to initialize Google Auth: {e}")
        st.stop()


def get_google_user_info(authenticator):
    """
    Get authenticated user information from Google.
    
    Args:
        authenticator: Google authenticator instance
        
    Returns:
        dict: User information with keys: email, name, picture, google_id
    """
    if st.session_state.get('connected'):
        user_info = st.session_state.get('user_info', {})
        return {
            'email': user_info.get('email'),
            'name': user_info.get('name'),
            'picture': user_info.get('picture'),
            'google_id': user_info.get('sub') or user_info.get('id'),
        }
    return None


def google_login_button(authenticator):
    """
    Display Google login button and handle authentication.
    
    Args:
        authenticator: Google authenticator instance
        
    Returns:
        dict: User info if authenticated, None otherwise
    """
    # Check if already authenticated
    authenticator.check_authentification()
    
    # Show login/logout based on state
    if st.session_state.get('connected'):
        user_info = get_google_user_info(authenticator)
        return user_info
    else:
        # Show login button
        authenticator.login()
        return None


def google_logout(authenticator):
    """
    Handle Google logout.
    
    Args:
        authenticator: Google authenticator instance
    """
    if st.session_state.get('connected'):
        authenticator.logout()
        # Clean up session state
        if 'user' in st.session_state:
            del st.session_state.user
        if 'user_settings' in st.session_state:
            del st.session_state.user_settings
        st.rerun()
