# app.py
import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ─── Page Config & Style ───────────────────────────────
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📨",
    layout="centered"
)

st.markdown(
    """
    <style>
    body { background-color: #f0f2f6; }
    .stApp { 
        background-image: linear-gradient(135deg, #ffffff 25%, #e6f2ff 100%);
    }
    .stButton>button {
        background-color: #4c8bf5; color: white; border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff; border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Load Models ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vectorizer():
    with open("model/vectorizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_model():
    with open("model/spam_model.pkl", "rb") as f:
        return pickle.load(f)

vectorizer = load_vectorizer()
model = load_model()

# Ensure stopwords are available
nltk.download("stopwords", quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ─── Text Transformation ───────────────────────────────
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    cleaned = [
        ps.stem(tok)
        for tok in tokens
        if tok.isalnum() and tok not in stop_words
    ]
    return " ".join(cleaned)

# ─── Main App ──────────────────────────────────────────
st.title("📨 SMS Spam Classifier")
st.write("Enter an SMS below to see if it’s **Spam** or **Ham**.")

with st.form(key="classify_form"):
    message = st.text_area("Your message", height=150)
    submit = st.form_submit_button("✅ Classify")

if submit:
    if not message.strip():
        st.warning("Please enter some text to classify.")
    else:
        processed = transform_text(message)
        vect = vectorizer.transform([processed])
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0].max()

        if str(pred) in ("1", "spam"):
            st.error(f"⚠️ Spam (confidence: {proba:.1%})")
        else:
            st.success(f"✅ Ham (confidence: {proba:.1%})")

st.markdown("---")
st.caption("Powered by: Streamlit • TF–IDF • Naive Bayes")
