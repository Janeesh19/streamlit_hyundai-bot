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
model_name = "models/gemini-2.0-flash-001"

# --------------------------------------------
# BASE PROMPT FOR THE HYUNDAI IONIQ 5 SALES CHATBOT
# --------------------------------------------
base_prompt = """
You are a professional automotive sales consultant.

VERY IMPORTANT INSTRUCTION:-
**DO NOT REPLY TO ANY QUESTION ABOUT ANY VEHICLE OTHER THAN THE HYUNDAI IONIQ 5.**
**STAY FOCUSED ON THE IONIQ 5 ONLY.**

*IMPORTANT INSTRUCTION:*
- Use bullet points where necessary, keeping answers short and concise.
- After your answer, on the next line suggest 1–2 follow-up questions the customer might ask.
- Do not bullet-point those follow-up suggestions.
- Always greet the customer warmly before your first answer.
- Maintain a warm, helpful, professional tone and build rapport.

Always output exactly one key in your final response JSON: `"answer"`.

Session rules:
- If the user says “bye”, “goodbye” etc., respond with a friendly closing and end the session.
- If there is 2 minutes of inactivity, end the session politely.

Your role: guide the customer to a confident purchase decision by understanding their needs, providing clear answers, and building trust.
"""

# --------------------------------------------
# INITIALISE GENAI CLIENT & CACHE (CACHED PER SESSION)
# --------------------------------------------
@st.cache_resource
def init_genai_cache():
    client = genai.Client()
    data_file_path = "data (1) (1).csv"  # your dataset
    uploaded_file = client.files.upload(file=data_file_path)

    # wait for the file to finish processing
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    cache = client.caches.create(
        model=model_name,
        config=types.CreateCachedContentConfig(
            display_name="hyundai_sales_data",
            system_instruction=base_prompt,
            contents=[uploaded_file],
            ttl="86400s"  # 24 hours
        )
    )
    return client, cache

client, cache = init_genai_cache()

# --------------------------------------------
# SCHEDULER TO REFRESH CACHE TTL DAILY
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
# SET UP CHAT HISTORY
# --------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"|"assistant", "content": str}

# --------------------------------------------
# GENERATE A SINGLE, CLEAN RESPONSE
# --------------------------------------------
def generate_response(question: str) -> str:
    # assemble the last few turns for context
    recent = st.session_state.messages[-4:]
    conversation = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
    full_prompt = (conversation + "\nCustomer: " + question) if conversation else "Customer: " + question

    resp = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.1,
            cached_content=cache.name
        )
    )

    raw = resp.text or ""
    # strip markdown fences and any leading “json” labels
    cleaned = re.sub(r"```(?:json)?", "", raw)
    cleaned = re.sub(r"(?i)^json\s*", "", cleaned).strip()
    return cleaned

# --------------------------------------------
# HANDLE A USER QUERY
# --------------------------------------------
def handle_user_query(user_query: str):
    # record user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # get the model output
    full_response = generate_response(user_query)

    # try to parse JSON and extract the answer
    try:
        parsed = json.loads(full_response)
        answer_text = parsed.get("answer", "")
    except json.JSONDecodeError:
        answer_text = full_response

    # record assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer_text})

# --------------------------------------------
# STREAMLIT UI
# --------------------------------------------
st.set_page_config(page_title="Hyundai IONIQ 5 Chatbot", layout="centered")
st.title("Hyundai IONIQ 5 Sales Chatbot")

if user_input := st.chat_input("Ask something about the Hyundai IONIQ 5…"):
    handle_user_query(user_input)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
