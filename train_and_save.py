# train_and_save.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
from transform import transform_text

# 1️⃣ Load dataset
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['transformed_text'] = df['text'].apply(transform_text)

# 2️⃣ Add extra spam examples (for unseen short messages)
extra_spam = [
    "Hi you won a lottery! Claim your ₹10,000 reward now.",
    "Congratulations! You are the lucky winner.",
    "Claim your prize now.",
    "You have been selected to receive a free gift.",
    "Win exciting rewards by clicking this link!",
    "Earn money fast from home!",
    "You are our lucky winner, click here to claim.",
    "Your KYC is pending, verify now to avoid account suspension.",
    "Get 90% off today only! Limited offer.",
    "Your account will be blocked if not verified immediately.",
    "Click to claim your cashback now.",
    "Update your Paytm KYC to continue using your wallet.",
    "Your Netflix account has been locked. Verify now.",
    "Your bank account will be blocked. Confirm your details here.",
    "Win a free iPhone 15 by completing a short survey!",
    "Your recharge failed, click to retry.",
]

extra_df = pd.DataFrame({
    'label': [1] * len(extra_spam),
    'text': extra_spam,
    'transformed_text': [transform_text(t) for t in extra_spam]
})

# Merge with original data
df = pd.concat([df, extra_df], ignore_index=True)

# 3️⃣ Custom stopwords – keep spam indicators
custom_stopwords = set(ENGLISH_STOP_WORDS) - {
    "free", "won", "claim", "prize", "congratulations", "offer", "click", "lottery""free", "won", "claim", "prize", "congratulations", 
     "click", "reward", "verify", "update"
}

# 4️⃣ TF-IDF with n-grams (better context for short spam)
tfidf = TfidfVectorizer(max_features=4000, stop_words=custom_stopwords, ngram_range=(1, 3))
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['label'].values

# 5️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6️⃣ Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 7️⃣ Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
print("✅ Model and vectorizer saved successfully with improved spam handling!")
