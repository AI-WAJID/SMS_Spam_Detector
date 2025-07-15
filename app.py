import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ─── Page Setup ──────────────────────────────────────────────────────────────────
# Set the page configuration for a more professional look
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📨",
    layout="centered",
    initial_sidebar_state="auto",
)

# ─── Custom CSS for Advanced Styling ─────────────────────────────────────────────
# Inject custom CSS for a modern, responsive, and theme-friendly design.
# This CSS creates a card-like layout, improves form elements, and styles the
# result display for better readability on any background.
st.markdown("""
<style>
    /* --- General & Themeing --- */
    /* Using Streamlit's theme variables for better adaptability */
    body {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* --- Main App Container (Card) --- */
    .main-container {
        background-color: var(--secondary-background-color);
        padding: 2rem 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--separator-color);
        margin-top: 2rem;
    }

    /* --- Header & Title --- */
    .title {
        text-align: center;
        color: var(--primary-color, #007bff); /* Use theme primary or fallback */
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: var(--text-color);
        margin-bottom: 2rem;
    }

    /* --- Form & Input Elements --- */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid var(--separator-color);
        background-color: var(--background-color);
        color: var(--text-color);
        font-size: 1.1rem;
        min-height: 150px;
    }
    .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px var(--primary-color-20, rgba(0,123,255,.25));
    }

    /* --- Button Styling --- */
    div.stButton > button {
        background-color: var(--primary-color, #007bff);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75em 1.5em;
        width: 100%;
        transition: all 0.2s ease-in-out;
        border: none;
    }
    div.stButton > button:hover {
        background-color: var(--primary-color-darker, #0056b3); /* Needs a defined theme variable or fallback */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    div.stButton > button:active {
        transform: translateY(0);
        box-shadow: none;
    }

    /* --- Custom Result Display --- */
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        text-align: center;
        font-size: 1.25rem;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px; /* Space between icon and text */
        border: 1px solid;
    }
    .result-ham {
        background-color: #e7f5e8;
        color: #28a745;
        border-color: #b8e0bb;
    }
    .result-spam {
        background-color: #f8d7da;
        color: #dc3545;
        border-color: #f5c6cb;
    }
    .result-warning {
        background-color: #fff3cd;
        color: #856404;
        border-color: #ffeeba;
    }
    
    /* --- Footer --- */
    .stCaption {
        text-align: center;
        color: var(--text-color-light, #6c757d);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Models & NLP Tools (Cached) ────────────────────────────────────────────
# Use st.cache_resource to load models only once, improving performance.
@st.cache_resource(show_spinner="Loading NLP model...")
def load_model_assets():
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model/spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    nltk.download("stopwords", quiet=True)
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    return vectorizer, model, ps, stop_words

vectorizer, model, ps, stop_words = load_model_assets()

# ─── Text Transformation Function ──────────────────────────────────────────────
def transform_text(text: str) -> str:
    """Cleans, tokenizes, stems, and removes stopwords from text."""
    # Find all word characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    # Stem words that are alphanumeric and not in the stop words list
    stemmed_tokens = [
        ps.stem(tok) for tok in tokens 
        if tok.isalnum() and tok not in stop_words
    ]
    return " ".join(stemmed_tokens)

# ─── UI Layout & Logic ───────────────────────────────────────────────────────────
# Wrap the main content in a styled container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Header ---
st.markdown('<h1 class="title">📨 SMS Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a message to check if it\'s spam or not.</p>', unsafe_allow_html=True)

# --- Input Form ---
with st.form(key="classify_form"):
    message = st.text_area(
        "Message Content", 
        height=150, 
        placeholder="e.g., 'Congratulations! You've won a $1000 gift card. Click here...'"
    )
    submit_button = st.form_submit_button(label="✅ Classify Message")

# --- Processing and Displaying the Result ---
if submit_button:
    if not message.strip():
        # Display a custom-styled warning if the input is empty
        st.markdown(
            '<div class="result-box result-warning">⚠️ Please enter a message to classify.</div>', 
            unsafe_allow_html=True
        )
    else:
        # 1. Preprocess the input text
        transformed_message = transform_text(message)
        
        # 2. Vectorize the text using the loaded TF-IDF vectorizer
        vector_input = vectorizer.transform([transformed_message])
        
        # 3. Predict using the Naive Bayes model
        prediction = model.predict(vector_input)[0]
        
        # 4. Get prediction probability for confidence score
        probability = model.predict_proba(vector_input)[0].max()

        # 5. Display the result in a custom-styled box
        if str(prediction) == "1": # Assuming '1' is Spam
            st.markdown(
                f'<div class="result-box result-spam">🚨 Spam (Confidence: {probability:.1%})</div>',
                unsafe_allow_html=True
            )
        else: # Otherwise, it's Ham
            st.markdown(
                f'<div class="result-box result-ham">✅ Ham (Confidence: {probability:.1%})</div>',
                unsafe_allow_html=True
            )

# Close the main container div
st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("🔍 Built with Streamlit · TF-IDF Vectorizer · Naive Bayes Classifier")
