# 📧 Email Spam Detection using Machine Learning

## 🎯 Objective
To build a machine learning model to classify emails as **Spam** or **Not Spam** using Natural Language Processing techniques.

## 🛠️ Tech Stack
- Python, Pandas, NLTK, Scikit-learn
- TF-IDF for text vectorization
- Multinomial Naive Bayes for classification

## 📊 Results
- Accuracy: ~97%
- Precision: ~96%
- Recall: ~94%

## 📁 Project Structure



# 📧 Email Spam Detection App

A machine learning web app that classifies emails as **Spam** or **Not Spam** using NLP and Naive Bayes, built with Streamlit.

🔗 **Live Demo:** [Click to Open](https://email-spam-detector-bqycpm2xoznmcmdsqp7vj4.streamlit.app/)

## 🚀 Features
- Predict if an email is spam
- Show prediction confidence
- Visualize text as word cloud
- Preprocessing using NLTK
- Built with Streamlit UI

## 🧠 Tech Stack
- Python, Scikit-learn, Pandas
- NLTK for text cleaning
- Streamlit for the frontend

## 📁 Project Structure
- `app.py`: Streamlit interface
- `transform.py`: Text preprocessing
- `train_and_save.py`: Model training
- `model.pkl`, `tfidf.pkl`: Saved model/vectorizer
- `requirements.txt`: Dependencies

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

