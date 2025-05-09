import streamlit as st
import os
import re
import json
import time
import schedule
import threading
from google import genai
from google.genai import types

# --------------------------------------------
# SET YOUR CONFIGURATIONS AND KEYS HERE
# --------------------------------------------
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
model_name = "models/gemini-2.0-flash-001"  # Use the same model for both generate and cache

# Base prompt for the Hyundai IONIQ 5 Sales Chatbot.
base_prompt = """
You are a professional automotive sales consultant.

VERY IMPORTANT INSTRUCTION:-
**DO NOT REPLY TO ANY OF THE QUESTION ANYTIME OTHER THAN IONIQ5. YOU ARE JUST SALES AGENT FOR IONIQ 5. THATS IT. DO NOT GO OUT OF THIS. JUST TALK ABOUT THE CAR**

*IMPORTANT INSTRUCTION:-*
**Use bullet points in giving answer about the question wherever necessary. Keep it short and concise.**
**After your answer to a question, in the next line suggest 1–2 questions that can help the customer based on the current question.**
**DON’T ADD SUGGESTED QUESTIONS IN BULLET POINTS.**

**Always greet the customer warmly before starting any conversation. Do not use structured response formats while greeting.**

Engage naturally in a multi-turn dialogue and always refer to previous conversation details to maintain continuity. Your communication must always be in ENGLISH. If the user asks a question in another language, politely ask them to continue in English.

Your primary role is to guide the customer towards making a confident and informed decision by:
1. Understanding their needs,
2. Providing relevant, clear answers,
3. Keeping the conversation engaging and friendly.

Your tone should be warm, helpful, and professional. Never rush to the end—build rapport as you go. Ensure that your final output is always a valid JSON object with **exactly one key**: "answer".

**Session Management:**
- If the user says goodbye (e.g., "bye", "goodbye", "see you", "talk later"), you must respond with a friendly closing and END the session.
- If the user is inactive for 2 minutes, politely end the session with a goodbye message.

**PRODUCT-SPECIFIC INSTRUCTION (Hyundai IONIQ 5 ONLY):**
You are representing the Hyundai IONIQ 5.
Do not answer any questions about other vehicles or unrelated topics. Focus solely on this model—its features, benefits, pricing, performance, interior/exterior, EV technology, financing, warranty, or test-drive process.

Your key objectives:
1. Close the sale by addressing the customer's concerns and creating a sense of urgency.
2. Be the customer's trusted expert on the Hyundai IONIQ 5.

If the customer's query is not related to the Hyundai IONIQ 5, politely refuse to answer.
"""

# --------------------------------------------
# STREAMLIT APP LAYOUT
# --------------------------------------------
st.set_page_config(page_title="Hyundai IONIQ 5 Chatbot", layout="centered")
st.title("Hyundai IONIQ 5 Sales Chatbot")

# --------------------------------------------
# INITIALISE THE CLIENT AND CREATE THE CACHE
# --------------------------------------------
@st.cache_resource
def init_genai_cache():
    client = genai.Client()
    data_file_path = "data (1) (1).csv"  # Ensure the CSV is in the same directory
    uploaded_file = client.files.upload(file=data_file_path)

    # Wait for processing
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    cache = client.caches.create(
        model=model_name,
        config=types.CreateCachedContentConfig(
            display_name="hyundai_sales_data",
            system_instruction=base_prompt,
            contents=[uploaded_file],
            ttl="3600s"
        )
    )
    return client, cache

client, cache = init_genai_cache()

# --------------------------------------------
# SCHEDULER TO REFRESH CACHE TTL EVERY DAY
# --------------------------------------------
def refresh_cache():
    client.caches.update(
        name=cache.name,
        config=types.UpdateCachedContentConfig(ttl="86400s")
    )

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

if "scheduler_started" not in st.session_state:
    schedule.every().day.do(refresh_cache)
    threading.Thread(target=run_scheduler, daemon=True).start()
    st.session_state["scheduler_started"] = True

# --------------------------------------------
# CONVERSATION MANAGEMENT
# --------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # Each message: {"role": "user" or "assistant", "content": ...}

# ---------------------------------------------------------
# GENERATOR FUNCTION: generate_response_stream
# ---------------------------------------------------------
def generate_response_stream(question: str):
    recent = st.session_state.messages[-4:] if len(st.session_state.messages) >= 4 else st.session_state.messages
    conversation_context = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent
    )
    full_prompt = (conversation_context + "\nCustomer: " + question) if conversation_context else ("Customer: " + question)

    response_stream = client.models.generate_content_stream(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.1,
            cached_content=cache.name
        )
    )

    for chunk in response_stream:
        raw = chunk.text or ""
        # remove markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw)
        # strip any leading "json" label
        cleaned = re.sub(r"(?i)^json\s*", "", cleaned)
        yield cleaned

# ---------------------------------------------------------
# HANDLING THE USER QUERY AND DISPLAYING THE RESPONSE
# ---------------------------------------------------------
def handle_user_query(user_query: str):
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Collect full response
    full_response = ""
    for chunk in generate_response_stream(user_query):
        full_response += chunk

    # Parse out the "answer" field
    try:
        parsed = json.loads(full_response)
        answer_text = parsed.get("answer", full_response).strip()
    except json.JSONDecodeError:
        answer_text = full_response.strip()

    # Append and display only the clean answer
    st.session_state.messages.append({"role": "assistant", "content": answer_text})

# --------------------------------------------
# STREAMLIT CHAT INTERFACE
# --------------------------------------------
user_query = st.chat_input("Ask something about the Hyundai IONIQ 5...")
if user_query:
    handle_user_query(user_query)

# Display the conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
