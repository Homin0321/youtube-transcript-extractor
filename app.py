import re
import os
from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import google.generativeai as genai

# Configuration constants
GEMINI_MODEL_NAME = 'gemini-flash-lite-latest'
# Prompt template for Gemini AI to analyze video transcripts
TRANSCRIPT_ANALYSIS_PROMPT = """Please analyze the following video transcript and provide a piece of organized content with the following structure:

1.  **Title:** (A concise and descriptive title for the video)
2.  **Executive Summary:** (2-3 sentences providing a high-level overview of the video's purpose and key takeaways)
3.  **Detailed Breakdown:**  Organize the transcript into coherent paragraphs, elaborating on the key points. Each paragraph must have a subtitle. Remove any filler words, greetings, or repetitive phrases that do not contribute to a clear understanding of the video's core message.

Transcript content: """

def initialize_session_state():
    """Initialize Streamlit session state variables with default values"""
    default_values = {
        'youtube_url': "",  # Store the current YouTube URL
        'gemini_api_key': os.getenv("GEMINI_API_KEY", ""),  # API key for Gemini AI
        'extracted_text': "",  # Store the extracted transcript
        'summarized_text': "",  # Store the AI-generated summary
        'chat_session': None,  # Store the Gemini chat session
        'chat_display_history': []  # Store chat message history
    }
    
    # Set default values if not already in session state
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data
def get_youtube_id(url):
    """
    Extracts video ID from YouTube URL (both standard and shorts).
    Args:
        url (str): YouTube video URL
    Returns:
        str: Video ID if found, None otherwise
    """
    # Check standard YouTube URL format
    match = re.search(r'(?:v=|youtu\.be/)([^&?]+)', url)
    if match:
        return match.group(1)
    # Check YouTube Shorts URL format
    match = re.search(r'shorts/([^?]+)', url)
    if match:
        return match.group(1)
    return None

@st.cache_data
def get_youtube_transcript(youtube_url, video_id, selected_lang_code):
    """
    Fetches transcript for a given YouTube video ID and language.
    Updates session state with extracted text or error messages.
    """
    if not youtube_url or not video_id:
        st.sidebar.error("Invalid YouTube URL or video ID.")
        return
    
    if not selected_lang_code:
        st.sidebar.error("Please select a subtitle language before extracting the transcript.")
        return
        
    with st.spinner("Extracting transcript..."):
        try:
            # Fetch transcript using YouTube API
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[selected_lang_code])
            cleaned_text = "\n".join(item['text'] for item in transcript)
            st.session_state.extracted_text = cleaned_text
            st.sidebar.success("Transcript successfully extracted!")

        except NoTranscriptFound:
            st.sidebar.error("No transcript found for the selected language.")
            st.session_state.extracted_text = ""
        except TranscriptsDisabled:
            st.sidebar.error("Transcripts are disabled for this video.")
            st.session_state.extracted_text = ""
        except Exception as e:
            st.sidebar.error(f"An error occurred while extracting the transcript: {str(e)}")
            st.session_state.extracted_text = ""

@st.cache_resource
def get_gemini_model():
    """
    Initialize and return Gemini AI model instance.
    Returns:
        GenerativeModel: Configured Gemini model instance
    """
    if not st.session_state.gemini_api_key:
        st.error("GEMINI_API_KEY is not set. Please check your .env file.")
        return None

    genai.configure(api_key=st.session_state.gemini_api_key)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)

def get_gemini_chat(context):
    """
    Initialize a new chat session with Gemini AI.
    Args:
        context (str): Transcript context for the chat session
    Returns:
        Chat: Initialized chat session
    """
    with st.spinner("Starting chat with Gemini..."):
        model = get_gemini_model()
        chat = model.start_chat(history=[])
        # Set initial context for the chat
        chat.send_message(
            f"You are an assistant that answers questions about this transcript:\n{context}"
        )
    return chat

def summarize_text_with_gemini(extracted_text):
    """
    Process transcript using Gemini AI to generate a structured summary.
    Args:
        extracted_text (str): Raw transcript text
    Returns:
        str: Formatted summary or None if error occurs
    """
    try:
        text = extracted_text.strip()
        if not text:
            st.warning("Extracted transcript is empty. Please extract the transcript first.")
            return None
        with st.spinner("Summarizing text with Gemini..."):
            model = get_gemini_model()
            response = model.generate_content(TRANSCRIPT_ANALYSIS_PROMPT + text)

        if not hasattr(response, 'text'):
            st.error("Gemini API response does not contain text.")
            return None
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

def initialize_chat_session(context):
    """
    Initialize or reset chat session with transcript context.
    Args:
        context (str): The transcript text to provide context for the chat
    """
    try:
        # Create new chat session with Gemini
        st.session_state["chat_session"] = get_gemini_chat(context)
        st.session_state["chat_display_history"] = []
    except Exception as e:
        st.error(f"Failed to initialize Gemini chat: {e}")
        # Reset session state on error
        st.session_state["chat_session"] = None
        st.session_state["chat_display_history"] = []

def chat_with_gemini(extracted_text):
    """
    Implements chat interface for transcript Q&A using Gemini API.
    Args:
        extracted_text (str): The transcript text to chat about
    """
    context = extracted_text
    if not context:
        st.info("Please extract a transcript first before using the chat feature.")
        return

    # Initialize chat session if not exists or was reset
    if "chat_session" not in st.session_state or st.session_state["chat_session"] is None:
        initialize_chat_session(context)
        if st.session_state["chat_session"] is None:
            return

    # Create chat input interface
    user_input = st.chat_input("Ask something about the video...")

    # Process user message and get AI response
    if user_input:
        # Add user message to chat history
        st.session_state["chat_display_history"].append({"role": "user", "content": user_input})
        try:
            # Get AI response
            response = st.session_state["chat_session"].send_message(user_input)
            answer = response.text.strip()
            # Add AI response to chat history
            st.session_state["chat_display_history"].append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Gemini Q&A Error: {e}")
            # Reset chat session on error
            initialize_chat_session(context)
            return

    # Display chat messages in reverse chronological order
    for message in reversed(st.session_state["chat_display_history"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def render_sidebar():
    """
    Renders the sidebar UI components including URL input and language selection.
    Handles transcript extraction workflow.
    """
    st.sidebar.title("YouTube Transcript Extractor")
    youtube_url = st.sidebar.text_input("Enter YouTube URL:")

    if youtube_url:
        # Reset state if URL changes
        if youtube_url != st.session_state.youtube_url:
            st.session_state.extracted_text = ""
            st.session_state.summarized_text = ""
            st.session_state["chat_session"] = None
            st.session_state.youtube_url = youtube_url

        # Validate and extract video ID
        video_id = get_youtube_id(youtube_url)
        if not video_id:
            st.sidebar.error("Invalid YouTube URL.")
            return

        try:
            # Get available transcripts for the video
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_languages = {t.language_code: t for t in transcript_list}

            if not available_languages:
                st.sidebar.warning("No subtitles available for this video.")
                return

            # Create language selection dropdown
            selected_lang_code = st.sidebar.selectbox(
                "Select subtitle language:",
                sorted(list(available_languages.keys()))
            )

            # Extract transcript button
            if st.sidebar.button("Extract Transcript", use_container_width=True):
                get_youtube_transcript(youtube_url, video_id, selected_lang_code)

        except (NoTranscriptFound, TranscriptsDisabled) as e:
            st.sidebar.warning(f"Subtitle error: {e}")
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred: {e}")

def render_main_content():
    """
    Renders the main content area with three tabs:
    1. Transcript - Shows raw extracted transcript
    2. AI Summary - Shows Gemini-generated summary
    3. Chatbot - Interactive Q&A interface
    """
    tab1, tab2, tab3 = st.tabs(["Transcript", "AI Summary", "Chatbot"])

    # Tab 1: Raw Transcript
    with tab1:
        if st.session_state.extracted_text:
            st.code(st.session_state.extracted_text, language='text', height=800)
        else:
            st.info("Transcript will appear here after extraction.")

    # Tab 2: AI Summary
    with tab2:
        if st.session_state.extracted_text:
            # Generate AI summary if transcript exists
            summarized = summarize_text_with_gemini(st.session_state.extracted_text)
            if summarized:
                st.session_state.summarized_text = summarized
                st.sidebar.success("AI summary generated successfully!")

        if st.session_state.summarized_text:
            st.markdown(st.session_state.summarized_text)
        else:
            st.info("Organized summary will appear here after processing with Gemini.")

    # Tab 3: Interactive Chatbot
    with tab3:
        if st.session_state.extracted_text:
            chat_with_gemini(st.session_state.extracted_text)
        else:
            st.info("Chatbot will appear here after extraction.")

def main():
    """
    Main application entry point.
    Initializes environment, configures page settings, and renders UI components.
    """
    load_dotenv()  # Load environment variables
    # Configure Streamlit page settings
    st.set_page_config(layout="wide", page_title="YouTube Transcript Extractor", page_icon="🎥")
    initialize_session_state()  # Initialize session state variables

    # Render UI components
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()