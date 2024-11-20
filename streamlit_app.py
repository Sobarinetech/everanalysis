import streamlit as st
import pandas as pd
import re
from email import policy
from email.parser import BytesParser
from io import BytesIO
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import networkx as nx
from gtts import gTTS
import tempfile
import os

# Load Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Helper: Parse emails
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

# Helper: Analyze sentiment
def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] > 0.05:
        return "Positive"
    elif sentiment['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Helper: Calculate response times
def calculate_response_times(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values(by="Date", inplace=True)
    df["Response Time (hours)"] = df["Date"].diff().dt.total_seconds() / 3600
    return df

# Helper: Perform Root Cause Analysis
def perform_rca(df):
    if df.empty or len(df) < 2:
        return (
            "Root Cause Analysis Summary:\n"
            "Insufficient data to perform a meaningful analysis.\n"
            "Please upload more emails with sufficient interaction."
        )

    # Escalation triggers
    escalation_triggers = []
    for _, row in df.iterrows():
        if re.search(r"\burgent\b|\bASAP\b|\bimmediate\b", row["Body"], re.IGNORECASE):
            escalation_triggers.append(f"Trigger in email from {row['Sender']} to {row['Receiver']}")

    # Key metrics
    top_senders = df["Sender"].value_counts().idxmax()
    top_receivers = df["Receiver"].value_counts().idxmax()
    avg_response_time = df["Response Time (hours)"].mean() if "Response Time (hours)" in df.columns else "N/A"

    # Sentiment summary
    sentiment_counts = df["Sentiment"].value_counts().to_dict()
    negative_emails = df[df["Sentiment"] == "Negative"]

    rca_summary = (
        f"Root Cause Analysis Summary:\n"
        f"Total Emails Analyzed: {len(df)}\n"
        f"Top Sender: {top_senders}\n"
        f"Top Receiver: {top_receivers}\n"
        f"Average Response Time: {avg_response_time:.2f} hours\n"
        f"Sentiment Overview: {sentiment_counts}\n"
        f"Escalation Triggers: {'; '.join(escalation_triggers) if escalation_triggers else 'None Found'}\n"
        f"Negative Emails: {len(negative_emails)} instances.\n"
    )
    return rca_summary

# Helper: Generate TTS
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
st.title("Enhanced Email RCA Tool")
st.write("Upload your email files (EML format) for detailed analysis.")

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
    df = calculate_response_times(df)

    st.subheader("Email Data Overview")
    st.dataframe(df)

    st.subheader("Visualizations")
    # Sentiment Pie Chart
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    # Timeline Chart
    st.line_chart(df["Response Time (hours)"])

    # Network Graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["Sender"], row["Receiver"])
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
    st.pyplot(plt)

    st.subheader("Root Cause Analysis (RCA)")
    rca_summary = perform_rca(df)
    st.text(rca_summary)

    audio_file = generate_tts(rca_summary)
    if audio_file:
        st.subheader("RCA Narration")
        st.audio(audio_file, format="audio/mp3")
        os.unlink(audio_file)  # Clean up the temporary file
