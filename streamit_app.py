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
model_name = "models/gemini-1.5-flash-001"

# Base prompt for the Hyundai IONIQ 5 Sales Chatbot.
base_prompt = """
Greet the customer warmly before anything else.  
You are a professional automotive sales consultant representing the Hyundai IONIQ 5.

— **Answer Format**  
• **Always** respond **only** with bullet points.  
• **Never** use paragraphs or run-on prose.  
• Keep each bullet short, clear, and impactful.  
• After your final bullet, on a new line, list **2–3 suggested follow-up questions** the customer might ask, each as its own bullet prefixed with “Follow-up:”.  

— **Tone & Content**  
• Friendly, helpful, professional.  
• Always in English; if asked otherwise, politely ask them to continue in English.  
• Refer back to earlier conversation context to stay coherent.  

— **Product Scope**  
• You speak **only** about the Hyundai IONIQ 5 (features, specs, pricing, test drives, financing, etc.).  
• If asked about anything else, reply:  
  • “I’m sorry, I can only provide information about the Hyundai IONIQ 5.”  

— **Key Objectives**  
1. Close the sale by addressing concerns and creating urgency.  
2. Thoroughly answer questions while keeping bullets concise.  
3. Encourage next steps (test drive, financing).  

— **Conversation Summary**  
At the end of every answer, append **one** bullet summarising the conversation so far, prefixed with “Summary:”.

IMPORTANT: Output **must** be a valid JSON object with exactly one key, `"answer"`.  
Do **not** include any code fences, markdown syntax, or extra keys.
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
    """
    Upload a CSV file (named data.csv in the same directory) and create a cache for the model.
    Adjust the CSV file path as needed.
    """
    client = genai.Client()
    data_file_path = ("data (1) (1).csv") # Ensure your CSV file is in the same directory as this script
    uploaded_file = client.files.upload(file=data_file_path)
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
# SCHEDULER TO REFRESH CACHE TTL EVERY 55 MINUTES
# --------------------------------------------
def refresh_cache():
    client.caches.update(
        name=cache.name,
        config=types.UpdateCachedContentConfig(ttl="3600s")
    )

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

if "scheduler_started" not in st.session_state:
    schedule.every(55).minutes.do(refresh_cache)
    threading.Thread(target=run_scheduler, daemon=True).start()
    st.session_state["scheduler_started"] = True

# --------------------------------------------
# CONVERSATION MANAGEMENT
# --------------------------------------------
if "messages" not in st.session_state:
    # Each message is stored as a dict: {"role": "user" or "assistant", "content": ...}
    st.session_state.messages = []

# ---------------------------------------------------------
# GENERATOR FUNCTION: generate_response_stream
# ---------------------------------------------------------
def generate_response_stream(question: str):
    if len(st.session_state.messages) >= 4:
        conversation_context = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[-4:]]
        )
    else:
        conversation_context = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
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
    
    full_text = ""
    for chunk in response_stream:
        raw = chunk.text or ""
        cleaned_chunk = re.sub(r"```(?:json)?", "", raw)
        full_text += cleaned_chunk
        yield cleaned_chunk


# ---------------------------------------------------------
# HANDLING THE USER QUERY AND DISPLAYING THE RESPONSE
# ---------------------------------------------------------
def handle_user_query(user_query: str):
    # Append user's message to the conversation history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    placeholder = st.empty()
    full_response = ""
    # Stream the answer using the generator function
    for chunk in generate_response_stream(user_query):
        full_response += chunk
        placeholder.text(full_response)  # Update the UI with streamed text
    
    # Process the full response to extract only the "answer" field if present
    answer_text = full_response
    try:
        parsed = json.loads(full_response)
        answer_text = parsed.get("answer", full_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", full_response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                answer_text = data.get("answer", full_response)
            except (json.JSONDecodeError, KeyError):
                answer_text = full_response
    
    # Append the assistant's plain text answer to conversation history
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    placeholder.empty()  # Clear the placeholder

# --------------------------------------------
# STREAMLIT CHAT INTERFACE
# --------------------------------------------
user_query = st.chat_input("Ask something about the Hyundai IONIQ 5...")

if user_query:
    handle_user_query(user_query)

# Display the conversation in a chat-like UI
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:  # "assistant"
        with st.chat_message("assistant"):
            st.write(msg["content"])
