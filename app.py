import re
import os
from dotenv import load_dotenv
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import google.generativeai as genai

# Configure Streamlit page
st.set_page_config(layout="wide", page_title="YouTube Transcript Extractor")

# Load environment variables
load_dotenv()

# Initialize session state variables
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'organized_text' not in st.session_state:
    st.session_state.organized_text = ""

# Sidebar setup
with st.sidebar:
    st.title("YouTube Transcript Extractor")
    youtube_url = st.text_input("Enter YouTube URL:")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

def get_youtube_id(url):
    """Extracts video ID from YouTube URL (both standard and shorts)."""
    # Check standard YouTube URL
    match = re.search(r'(?:v=|youtu\.be/)([^&?]+)', url)
    if match:
        return match.group(1)
    # Check YouTube Shorts URL
    match = re.search(r'shorts/([^?]+)', url)
    if match:
        return match.group(1)
    return None

@st.cache_resource
def get_gemini_model(api_key):
    """Returns a cached instance of the Gemini model."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash-lite')

@st.cache_resource
def get_gemini_chat(api_key, context):
    model = get_gemini_model(api_key)
    chat = model.start_chat(history=[])
    # Initialize with transcript context
    chat.send_message(
        f"You are an assistant that answers questions about this transcript:\n{context}"
    )
    return chat

def organize_text_with_gemini(text, api_key):
    """Process transcript using Gemini AI API."""
    try:
        model = get_gemini_model(api_key)
        prompt = """Please analyze the following video transcript and provide a piece of organized content with the following structure:

1.  **Title:** (A concise and descriptive title for the video)
2.  **Executive Summary:** (2-3 sentences providing a high-level overview of the video's purpose and key takeaways)
3.  **Detailed Breakdown:**  Organize the transcript into coherent paragraphs, elaborating on the key points. Each paragraph must have a subtitle. Remove any filler words, greetings, or repetitive phrases that do not contribute to a clear understanding of the video's core message.

Transcript content: """
        response = model.generate_content(prompt + text)
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

@st.dialog(title="Chat with Gemini", width="large")
def chat_with_gemini(context, api_key):
    """Chat interface to ask questions about the video transcript using Gemini API."""
    user_input = st.chat_input("Ask something about the video...")

    if "chat_session" not in st.session_state:
        st.session_state["chat_session"] = get_gemini_chat(api_key, context)
    if "chat_display_history" not in st.session_state:
        st.session_state["chat_display_history"] = []

    if user_input:
        st.session_state["chat_display_history"].append({"role": "user", "content": user_input})
        try:
            response = st.session_state["chat_session"].send_message(user_input)
            answer = response.text.strip()
            st.session_state["chat_display_history"].append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Gemini Q&A Error: {e}")

    # Display previous messages in reverse order
    for message in reversed(st.session_state["chat_display_history"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Process YouTube URL if provided
if youtube_url:
    with st.sidebar:
        if not gemini_api_key:
            st.warning("Gemini API Key not found. Please add it to your .env file.")

        video_id = get_youtube_id(youtube_url)
        if video_id:
            try:
                # Get available transcripts
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                available_languages = {t.language_code: t for t in transcript_list}

                if available_languages:
                    # Language selection
                    selected_lang_code = st.selectbox(
                        "Select subtitle language:",
                        sorted(list(available_languages.keys()))
                    )

                    # Extract transcript
                    if selected_lang_code:
                        if st.button("Extract Transcript", width="stretch"):
                            with st.spinner("Extracting transcript..."):
                                try:
                                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[selected_lang_code])
                                    cleaned_text = "\n".join(item['text'] for item in transcript)
                                    st.session_state.extracted_text = cleaned_text

                                except NoTranscriptFound:
                                    st.error(f"No transcript found for the selected language.")
                                    st.session_state.extracted_text = ""
                                except Exception as e:
                                    st.error(f"An error occurred while extracting the transcript: {e}")
                                    st.session_state.extracted_text = ""
                else:
                    st.warning("No subtitles are available for this video.")
                    st.session_state.extracted_text = ""

            except (NoTranscriptFound, TranscriptsDisabled) as e:
                st.warning(f"Subtitle error: {e}")
                st.session_state.extracted_text = ""
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.extracted_text = ""
        else:
            st.error("Invalid YouTube URL.")
            st.session_state.extracted_text = ""

        if st.session_state.extracted_text:
            if st.button("Organize by Gemini", width="stretch"):
                with st.spinner("Organizing by Gemini..."):
                    st.session_state.organized_text = organize_text_with_gemini(st.session_state.extracted_text, gemini_api_key)
            if st.button("Chat with Gemini", width="stretch"):
                chat_with_gemini(st.session_state.extracted_text, gemini_api_key)

if st.session_state.organized_text:
    st.markdown(st.session_state.organized_text)
elif st.session_state.extracted_text:
    st.write(st.session_state.extracted_text)
else:
    st.info("Enter YouTube URL and click 'Extract Transcript' to begin.")