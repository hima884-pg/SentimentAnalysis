import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("SentimentalAnalysis/sentiment_svm_model.pkl")
vectorizer = joblib.load("SentimentalAnalysis/tfidf_vectorizer.pkl")

# Text cleaner
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    return text

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("📝 Product Review Sentiment Analyzer")

review = st.text_area("Enter your product review here:")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        cleaned_review = clean_text(review)
        vectorized_input = vectorizer.transform([cleaned_review])
        prediction = model.predict(vectorized_input)[0]

        sentiment = "🟢 Positive" if prediction == 1 else "🔴 Negative"
        st.subheader("Predicted Sentiment:")
        st.success(sentiment)