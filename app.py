# app.py

import os
import json
import re
import joblib
import streamlit as st
from dotenv import load_dotenv

# NEW: use the new Google Gen AI SDK (with web search tools)
from google import genai

# ------------ Load environment variables ------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ------------ Configure Gemini client (with Search support) ------------
gemini_client = None
if GOOGLE_API_KEY:
    try:
        # New SDK client
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print("Error configuring Gemini client:", e)
else:
    print("⚠️ GOOGLE_API_KEY is not set in .env")

# ------------ Load local trained model ------------
MODEL_PATH = os.path.join("models", "fake_news_model.pkl")

local_model = None
if os.path.exists(MODEL_PATH):
    try:
        local_model = joblib.load(MODEL_PATH)
    except Exception as e:
        print("Error loading local model:", e)
else:
    print(f"⚠️ Local model not found at {MODEL_PATH}. Run trained_model.py first.")

# ------------ Helper: Local model prediction ------------
def predict_with_local_model(text: str):
    if local_model is None:
        return {
            "label": "N/A",
            "probability": None,
            "error": "Local model not loaded. Please train model first."
        }

    try:
        proba = local_model.predict_proba([text])[0]
        pred = local_model.predict([text])[0]

        # Assuming 0 = FAKE, 1 = REAL
        label = "REAL" if pred == 1 else "FAKE"
        confidence = float(max(proba))

        return {
            "label": label,
            "probability": confidence,
            "error": None
        }
    except Exception as e:
        return {
            "label": "N/A",
            "probability": None,
            "error": str(e)
        }

# ------------ Helper: Extract JSON from plain text (fallback) ------------
def extract_json(text: str):
    # safety: None ya non-string aayega to handle kar lo
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)

    # Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find JSON-like block using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception:
            return None
    return None

# ------------ Helper: Gemini prediction (with LIVE Google Search) ------------
def predict_with_gemini(text: str):
    """
    Uses Gemini 2.5 Flash + Google Search grounding.
    Model will actually hit web search and then answer.
    """
    if gemini_client is None:
        return {
            "label": "N/A",
            "confidence": None,
            "reason": "Gemini client not configured.",
            "raw": None,
            "error": "Gemini model not available."
        }

    prompt = f"""
You are a fake news detection assistant.

Use LIVE Google Search (via the google_search tool) to check if this news headline has been reported
by reliable news sources. Decide if the news appears "FAKE" or "REAL" based on what you find.

Your JSON "reason" MUST be very short: maximum 2 sentences and no more than 200 characters.

Headline:
\"\"\"{text}\"\"\"


Reply ONLY in this JSON format (no extra text, no markdown):

{{
  "label": "FAKE" or "REAL",
  "confidence": 0.0 to 1.0,
  "reason": "short 1–2 line explanation in English (max 200 characters)"
}}
"""

    try:
        # response_mime_type nahi use kar rahe (400 error se bachne ke liye)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "tools": [{"google_search": {}}],  # Google Search tool
            },
        )

        resp_text = getattr(response, "text", None)

        # Try to parse JSON directly
        parsed = None
        if resp_text is not None:
            try:
                parsed = json.loads(resp_text)
            except Exception:
                parsed = extract_json(resp_text)

        if parsed is None:
            # Fallback if something goes wrong
            return {
                "label": "UNKNOWN",
                "confidence": None,
                "reason": "Could not parse JSON from Gemini response.",
                "raw": resp_text,
                "error": "JSON parsing failed."
            }

        label = parsed.get("label", "UNKNOWN")
        confidence = parsed.get("confidence", None)
        reason = parsed.get("reason", "")

        return {
            "label": label,
            "confidence": confidence,
            "reason": reason,
            "raw": parsed,
            "error": None
        }

    except Exception as e:
        return {
            "label": "N/A",
            "confidence": None,
            "reason": "",
            "raw": None,
            "error": str(e)
        }

# ------------ Helper: Overall verdict (Local + Gemini) ------------
def compute_overall_verdict(local_label: str, gemini_label: str):
    """
    Tumhare rules:
      - Dono FAKE  -> FAKE
      - Dono REAL  -> REAL
      - Local FAKE & Gemini REAL -> REAL
      - Local REAL & Gemini FAKE -> FAKE

    In short: jab dono valid hon, overall verdict Gemini ke label ke equal hai.
    """
    valid = {"FAKE", "REAL"}

    if local_label in valid and gemini_label in valid:
        return gemini_label

    if gemini_label in valid:
        return gemini_label
    if local_label in valid:
        return local_label

    return "UNKNOWN"

# ------------ Streamlit UI ------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write(
    "This app gives predictions from two sources:\n"
    "1. 🧠 Your own trained ML model (TF-IDF + Logistic Regression)\n"
    "2. 🤖 Google Gemini (LLM) + Google Search (live web)\n\n"
    "Enter a headline and see what both of them say!"
)

headline = st.text_area(
    "News headline / title:",
    placeholder="e.g. Government announces free college for all students from next month...",
    height=100
)

if st.button("Check News"):
    if not headline.strip():
        st.warning("Please koi headline to likho bhai 🙂")
    else:
        with st.spinner("Analyzing..."):

            # Local model prediction
            local_result = predict_with_local_model(headline)

            # Gemini prediction (with live web search)
            gemini_result = predict_with_gemini(headline)

        # Show results side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 Local ML Model")
            if local_result["error"]:
                st.error(local_result["error"])
            else:
                st.markdown(f"**Prediction:** `{local_result['label']}`")
                if local_result["probability"] is not None:
                    st.markdown(
                        f"**Confidence:** `{local_result['probability']:.2f}`"
                    )

        with col2:
            st.subheader("🤖 Gemini (LLM) + 🌐 Web Search")
            if gemini_result["error"]:
                st.error(gemini_result["error"])
            else:
                st.markdown(f"**Prediction:** `{gemini_result['label']}`")
                if gemini_result["confidence"] is not None:
                    st.markdown(
                        f"**Confidence:** `{gemini_result['confidence']:.2f}`"
                    )
                if gemini_result.get("reason"):
                    st.markdown("**Reason (short):**")
                    st.write(gemini_result["reason"])

        # Overall verdict
        local_label = local_result.get("label", "UNKNOWN")
        gemini_label = gemini_result.get("label", "UNKNOWN")

        overall_label = compute_overall_verdict(local_label, gemini_label)

        conflict = (
            local_label in ("FAKE", "REAL")
            and gemini_label in ("FAKE", "REAL")
            and local_label != gemini_label
        )

        st.markdown("---")
        st.subheader("📊 Overall Verdict")

        if overall_label == "FAKE":
            st.error("**Overall Prediction: FAKE** — news seems misleading / false.")
        elif overall_label == "REAL":
            st.success("**Overall Prediction: REAL** — news seems likely genuine.")
        else:
            st.info("**Overall Prediction: UNKNOWN** — models are not confident enough.")

        if conflict:
            st.caption(
                "Note: Local ML Model aur Gemini (LLM) ke predictions different the, "
                "isliye overall verdict **Gemini ke according** liya gaya hai (tumhare rules ke hisaab se)."
            )

        # Detailed raw Gemini JSON/text (optional)
        with st.expander("🔍 See Gemini raw output / JSON"):
            st.write(gemini_result["raw"] if gemini_result.get("raw") else "No raw data.")

st.markdown("---")
st.caption(
    "Tip:  First run `trained_model.py` to train the model, then run `streamlit run app.py`.\n"
    "Note: Gemini uses the Google Search tool here, which fetches information from the live web to give answers."
)
