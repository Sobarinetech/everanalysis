import streamlit as st
import pandas as pd
import re
from email import policy
from email.parser import BytesParser
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from gtts import gTTS
import os
import tempfile

# Load Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Function to parse emails
def parse_email(email_bytes):
    try:
        email = BytesParser(policy=policy.default).parsebytes(email_bytes)
        sender = email.get("From", "Unknown Sender")
        receiver = email.get("To", "Unknown Receiver")
        date = email.get("Date", None)
        subject = email.get("Subject", "No Subject")
        body = email.get_body(preferencelist=('plain', 'html')).get_content() if email.get_body() else ""
        return sender, receiver, date, subject, body
    except Exception as e:
        st.error(f"Error parsing email: {e}")
        return None, None, None, None, ""

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] > 0.05:
        return "Positive"
    elif sentiment['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function to perform RCA
def perform_rca(df):
    if df.empty or len(df) < 2:
        return (
            "Root Cause Analysis Summary:\n"
            "Insufficient data to perform a meaningful Root Cause Analysis.\n"
            "Please upload more emails or ensure the emails contain relevant content."
        )

    escalation_triggers = []
    for _, row in df.iterrows():
        if "urgent" in row["Body"].lower() or "immediate" in row["Body"].lower():
            escalation_triggers.append(f"Escalation Trigger in email from {row['Sender']} to {row['Receiver']}")

    top_senders = df["Sender"].value_counts().idxmax()
    top_receivers = df["Receiver"].value_counts().idxmax()

    if "Response Time (hours)" in df.columns:
        avg_response_time = df["Response Time (hours)"].mean()
        max_response_time = df["Response Time (hours)"].max()
        response_issues = (
            f"Average Response Time: {avg_response_time:.2f} hours\n"
            f"Longest Response Time: {max_response_time:.2f} hours\n"
        )
    else:
        response_issues = "Response time data unavailable.\n"

    sentiment_counts = df["Sentiment"].value_counts().to_dict()
    sentiment_summary = f"Sentiment Overview: {sentiment_counts}\n"

    escalation_summary = (
        f"Escalation Triggers: {'; '.join(escalation_triggers) if escalation_triggers else 'None Found'}\n"
    )

    rca_summary = (
        f"Root Cause Analysis Summary:\n"
        f"Total Emails Analyzed: {len(df)}\n"
        f"Top Sender: {top_senders}\n"
        f"Top Receiver: {top_receivers}\n"
        f"{response_issues}"
        f"{sentiment_summary}"
        f"{escalation_summary}"
    )

    return rca_summary

# Function to generate text-to-speech narration
def generate_tts(text):
    try:
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Error generating TTS: {e}")
        return None

# Streamlit App
st.title("Email Analysis and Root Cause Analysis (RCA) Tool")
st.write("Upload your email files (EML format) for analysis.")

uploaded_files = st.file_uploader("Upload Emails", type=["eml"], accept_multiple_files=True)

if uploaded_files:
    data = []
    for file in uploaded_files:
        email_bytes = file.read()
        sender, receiver, date, subject, body = parse_email(email_bytes)
        sentiment = analyze_sentiment(body)
        data.append({
            "Sender": sender,
            "Receiver": receiver,
            "Date": date,
            "Subject": subject,
            "Body": body,
            "Sentiment": sentiment
        })

    df = pd.DataFrame(data)

    st.subheader("Email Data Overview")
    st.dataframe(df)

    st.subheader("Sentiment Analysis")
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("Root Cause Analysis (RCA)")
    rca_summary = perform_rca(df)
    st.text(rca_summary)

    audio_file = generate_tts(rca_summary)
    if audio_file:
        st.subheader("RCA Narration")
        st.audio(audio_file, format="audio/mp3")
        os.unlink(audio_file)  # Clean up the temporary file
