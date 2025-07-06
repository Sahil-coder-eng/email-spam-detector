# app.py

import streamlit as st
import joblib
from transform import transform_text
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# Load model and vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📧 Spam Detector App")
    st.markdown("Built by **Sahil Samal**")
    st.markdown("🚀 Using Python, NLP, and Naive Bayes")
    st.markdown("🔗 [GitHub](https://github.com/Sahil-coder-eng)")
    st.markdown("---")
    st.markdown("💡 Paste any email text to classify as **Spam** or **Not Spam**")
    st.markdown("✅ Confidence score, word cloud, and transformed text included!")

# ---------- Page Title ----------
st.title("📨 Email Spam Detection")
st.markdown("Enter an email message below to detect if it's **Spam** or **Not Spam**.")

# ---------- Input Email ----------
email_input = st.text_area("✉️ Your Email Text Here", height=200)

if st.button("📌 Predict"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter a valid email message.")
    else:
        # Transform and vectorize
        transformed = transform_text(email_input)
        vector_input = tfidf.transform([transformed])
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]

        # Result display
        if result == 1:
            st.error("🚫 This email is **SPAM**.")
        else:
            st.success("✅ This email is **NOT SPAM**.")

        # Show prediction confidence
        st.subheader("🔍 Prediction Confidence")
        st.write(f"Spam: {proba[1]:.2%} | Not Spam: {proba[0]:.2%}")
        st.progress(float(proba[1]))

        # Show transformed text
        if st.checkbox("🔎 Show Transformed Text"):
            st.code(transformed)

        # Show word cloud
        if st.checkbox("🌀 Show Word Cloud"):
            wc = WordCloud(width=500, height=250, background_color='white').generate(transformed)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

# ---------- Bulk Prediction ----------
st.markdown("---")
st.header("📋 Bulk Email Prediction")
multi_input = st.text_area("Paste multiple emails (one per line):")

if st.button("📍 Predict All"):
    if multi_input.strip() == "":
        st.warning("Please enter at least one email.")
    else:
        lines = [line.strip() for line in multi_input.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            transformed = transform_text(line)
            vector = tfidf.transform([transformed])
            result = model.predict(vector)[0]
            label = "🚫 SPAM" if result == 1 else "✅ Not Spam"
            st.write(f"**Email {i+1}:** {label}")
