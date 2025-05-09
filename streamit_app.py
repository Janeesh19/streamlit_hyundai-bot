import streamlit as st
import os
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
model_name = "models/gemini-2.0-flash-001"  # same model for generate and cache

# Base prompt for the Hyundai IONIQ 5 Sales Chatbot.
base_prompt = """
You are a professional automotive sales consultant.

VERY IMPORTANT INSTRUCTION:- 
**DO NOT REPLY TO ANY OF THE QUESTION ANYTIME OTHER THAN IONIQ5. YOU ARE JUST SALES AGENT FOR IONIQ 5. THATS IT. DO NOT GO OUT OF THIS. JUST TALK ABOUT THE CAR**

*IMPORTANT INSTRUCTION:-*
**Use bullet points in giving answer about the question wherever necessary. Keep it short and concise.**
**After your answer to a question, in the next line suggest 1–2 questions that can help the customer, based on their current question.**
**DON’T ADD SUGGESTED QUESTIONS IN BULLET POINTS.**

**Always greet the customer warmly before starting any conversation. Do not use structured response formats while greeting.**

Engage naturally in a multi-turn dialogue and always refer to previous conversation details to maintain continuity. Your communication must always be in ENGLISH. If the user asks a question in another language, politely ask them to continue in English.

Your primary role is to guide the customer towards making a confident and informed decision by:
1. Understanding their needs,
2. Providing relevant, clear answers,
3. Keeping the conversation engaging and friendly.

Your tone should be warm, helpful, and professional. Never rush to the end—build rapport as you go. Ensure that your final output is always a valid JSON object with **exactly one key**: `"answer"`.

**Session Management:**
- If the user says goodbye (e.g., "bye", "goodbye", "see you", "talk later"), you must respond with a friendly closing and END the session.
- If the user is inactive for 2 minutes, politely end the session with a goodbye message.

**PRODUCT-SPECIFIC INSTRUCTION (Hyundai IONIQ 5 ONLY):**

You are representing the Hyundai IONIQ 5. Do not answer any questions about other vehicles or unrelated topics. Focus solely on this model—its features, benefits, pricing, performance, interior/exterior, EV technology, financing, warranty, or test-drive process.

Your key objectives:
1. Close the sale by addressing the customer’s concerns and creating a sense of urgency.
2. Be the customer’s trusted expert on the Hyundai IONIQ 5.

If the customer’s query is not related to the Hyundai IONIQ 5, politely refuse to answer.
"""

# --------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------
st.set_page_config(page_title="Hyundai IONIQ 5 Chatbot", layout="centered")
st.title("Hyundai IONIQ 5 Sales Chatbot")

# --------------------------------------------
# CACHE THE CLIENT & DATA FOR 1 HOUR
# --------------------------------------------
@st.cache_resource
def init_genai_cache():
    client = genai.Client()
    data_file_path = "data (1) (1).csv"  # adjust if needed
    uploaded = client.files.upload(file=data_file_path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    cache = client.caches.create(
        model=model_name,
        config=types.CreateCachedContentConfig(
            display_name="hyundai_sales_data",
            system_instruction=base_prompt,
            contents=[uploaded],
            ttl="3600s"
        )
    )
    return client, cache

client, cache = init_genai_cache()

# --------------------------------------------
# REFRESH CACHE DAILY AT 00:00 UTC
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
    schedule.every().day.at("00:00").do(refresh_cache)
    threading.Thread(target=run_scheduler, daemon=True).start()
    st.session_state["scheduler_started"] = True

# --------------------------------------------
# INITIALISE CONVERSATION HISTORY
# --------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------
# HANDLE USER QUERY (ONE-SHOT JSON PARSE)
# --------------------------------------------
def handle_user_query(user_query: str):
    # 1. save user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. build the last 4 messages as context
    recent = st.session_state.messages[-4:]
    conv = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
    prompt = (conv + "\nCustomer: " + user_query) if conv else ("Customer: " + user_query)

    # 3. one-shot generate the JSON response
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.1,
            cached_content=cache.name
        )
    )
    raw = response.text

    # 4. extract only the "answer" field
    try:
        answer = json.loads(raw)["answer"]
    except (json.JSONDecodeError, KeyError):
        answer = raw.strip()

    # 5. save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --------------------------------------------
# STREAMLIT CHAT INTERFACE
# --------------------------------------------
user_input = st.chat_input("Ask something about the Hyundai IONIQ 5…")
if user_input:
    handle_user_query(user_input)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        # If it looks like JSON, parse & extract "answer"
        if msg["role"] == "assistant":
            try:
                parsed = json.loads(content)
                st.write(parsed.get("answer", content))
            except json.JSONDecodeError:
                st.write(content)
        else:
            st.write(content)
