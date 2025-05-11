import streamlit as st

# ───────────────────────────────────────────────────────────────────────
# Must be the very first Streamlit command in your script:
st.set_page_config(page_title="Hyundai IONIQ 5 Chatbot", layout="centered")
# ───────────────────────────────────────────────────────────────────────

import os
import re
import json
import time
import schedule
import threading
from google import genai
from google.genai import types

# --------------------------------------------
# YOUR ORIGINAL BASE PROMPT (UNCHANGED)
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

model_name = "models/gemini-2.0-flash-001"
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --------------------------------------------
# INITIALISE GENAI CLIENT & CACHE ONCE
# --------------------------------------------
def init_genai_cache():
    client = genai.Client()
    data_file_path = "data (1) (1).csv"
    uploaded_file = client.files.upload(file=data_file_path)

    # wait until processing finishes
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    cache = client.caches.create(
        model=model_name,
        config=types.CreateCachedContentConfig(
            display_name="hyundai_sales_data",
            system_instruction=base_prompt,
            contents=[uploaded_file],
            ttl="86400s"
        )
    )
    return client, cache

if "client" not in st.session_state:
    st.session_state.client, st.session_state.cache = init_genai_cache()

client = st.session_state.client
cache = st.session_state.cache

# --------------------------------------------
# SCHEDULER TO REFRESH CACHE TTL DAILY
# --------------------------------------------
def _refresh_cache():
    client.caches.update(
        name=cache.name,
        config=types.UpdateCachedContentConfig(ttl="86400s")
    )

def _run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

if "scheduler_started" not in st.session_state:
    schedule.every().day.do(_refresh_cache)
    threading.Thread(target=_run_scheduler, daemon=True).start()
    st.session_state.scheduler_started = True

# --------------------------------------------
# PREPARE CHAT HISTORY
# --------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------
# SYNCHRONOUS RESPONSE (NO STREAMING)
# --------------------------------------------
def generate_response(question: str) -> str:
    recent = st.session_state.messages[-4:]
    conversation = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
    prompt = (conversation + "\nCustomer: " + question) if conversation else "Customer: " + question

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
    return cleaned

# --------------------------------------------
# HANDLE USER INPUT
# --------------------------------------------
def handle_user_query(user_query: str):
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 1. Get the raw cleaned response:
    raw = generate_response(user_query)

    # 2. If it’s a pure JSON blob, pull out the {...} bit
    m = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    json_str = m.group(1) if m else raw

    # 3. Try to parse it
    try:
        parsed = json.loads(json_str)
        answer_text = parsed.get("answer", "")
    except json.JSONDecodeError:
        answer_text = raw

    # 4. Post-process escapes and markdown bullets
    #    - Turn any literal “\n” into real newlines
    answer_text = answer_text.replace("\\n", "\n")
    #    - Convert leading “* ” bullets into hyphens
    answer_text = re.sub(r"(?m)^\*\s*", "- ", answer_text)
    #    - Strip any stray whitespace
    answer_text = answer_text.strip()

    # 5. Append the final clean answer
    st.session_state.messages.append({"role": "assistant", "content": answer_text})

# --------------------------------------------
# RENDER THE CHAT UI
# --------------------------------------------
st.title("Hyundai IONIQ 5 Sales Chatbot")

if user_input := st.chat_input("Ask something about the Hyundai IONIQ 5…"):
    handle_user_query(user_input)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
