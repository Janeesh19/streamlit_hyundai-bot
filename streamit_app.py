import streamlit as st

# ────────────────────────────────────────────────────────────────────────
# Must be the very first Streamlit command:
st.set_page_config(page_title="Hyundai IONIQ 5 Chatbot", layout="centered")
# ────────────────────────────────────────────────────────────────────────

import os
import re
import time
import schedule
import threading
from google import genai
from google.genai import types

# --------------------------------------------
# YOUR ORIGINAL BASE PROMPT (WITH JSON LINE REMOVED)
# --------------------------------------------
base_prompt = """
You are a professional automotive sales consultant.

VERY IMPORTANT INSTRUCTION:-
**DO NOT REPLY TO ANY OF THE QUESTION ANYTIME OTHER THAN IONIQ5. YOU ARE JUST SALES AGENT FOR IONIQ 5. THATS IT.DO NOT GO OUT OF THIS.JUST TALK ABOUT THE CAR**

*IMPORTANT INSTRUCTION:-*
**Use bullet points in giving answer about the question where ever necessary.keep it short and concise**
**after your answer to a question, in the next line suggest 1–2 questions that can help the customer based on the current question.**
**DON’T ADD SUGGESTED QUESTIONS IN BULLET POINTS.**

**Always greet the customer warmly before starting any conversation. Do not use structured response formats while greeting.**

Engage naturally in a multi-turn dialogue and always refer to previous conversation details to maintain continuity. Your communication must always be in ENGLISH. If the user asks a question in another language, politely ask them to continue in English.

Your primary role is to guide the customer towards making a confident and informed decision by:
1. Understanding their needs,
2. Providing relevant, clear answers,
3. Keeping the conversation engaging and friendly.

Your tone should be warm, helpful, and professional. Never rush to the end—build rapport as you go.

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

model_name = "models/gemini-2.0-flash-001"
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --------------------------------------------
# FUNCTIONS TO MANAGE A SINGLE CACHE
# --------------------------------------------
def delete_cache(name: str):
    """Delete the named cache and remove it from session state."""
    try:
        client = st.session_state.client
        client.caches.delete(name=name)
    except Exception:
        pass
    st.session_state.pop("cache", None)

def init_genai_cache():
    client = genai.Client()

    CACHE_DISPLAY = "hyundai_sales_data"

    # 1) Look for an existing cache
    for c in client.caches.list():
        if c.display_name == CACHE_DISPLAY:
            return client, c

    # 2) Upload the CSV and wait until active
    data_file = "data (1) (1).csv"
    uploaded = client.files.upload(file=data_file)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)

    # 3) Create a new cache with 30-minute TTL
    cache = client.caches.create(
        model=model_name,
        config=types.CreateCachedContentConfig(
            display_name=CACHE_DISPLAY,
            system_instruction=base_prompt,
            contents=[uploaded],
            ttl="1800s"    # 30 minutes
        )
    )

    # 4) Schedule automatic deletion in 30 minutes
    schedule.every(30).minutes.do(lambda: delete_cache(cache.name))

    return client, cache

# --------------------------------------------
# INITIALISE CLIENT & CACHE ONCE
# --------------------------------------------
if "client" not in st.session_state:
    st.session_state.client, st.session_state.cache = init_genai_cache()

client = st.session_state.client
cache  = st.session_state.cache

# --------------------------------------------
# RUN THE SCHEDULER IN THE BACKGROUND
# --------------------------------------------
def _run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

if "scheduler_started" not in st.session_state:
    threading.Thread(target=_run_scheduler, daemon=True).start()
    st.session_state.scheduler_started = True

# --------------------------------------------
# CHAT HISTORY SETUP
# --------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------
# GENERATE A SINGLE, PLAIN-TEXT RESPONSE
# --------------------------------------------
def generate_response(question: str) -> str:
    recent = st.session_state.messages[-4:]
    context = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
    prompt  = f"{context}\nCustomer: {question}" if context else f"Customer: {question}"

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.1,
            cached_content=cache.name
        )
    )

    raw = resp.text or ""
    cleaned = re.sub(r"```(?:json)?", "", raw)
    cleaned = re.sub(r"(?i)^json\s*", "", cleaned).strip()
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = re.sub(r"(?m)^\*\s*", "- ", cleaned)
    return cleaned

# --------------------------------------------
# HANDLE USER INPUT
# --------------------------------------------
def handle_user_query(user_query: str):
    st.session_state.messages.append({"role": "user", "content": user_query})
    answer = generate_response(user_query)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --------------------------------------------
# RENDER THE CHAT UI
# --------------------------------------------
st.title("Hyundai IONIQ 5 Sales Chatbot")

if user_input := st.chat_input("Ask something about the Hyundai IONIQ 5…"):
    handle_user_query(user_input)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
