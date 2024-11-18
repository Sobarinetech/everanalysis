import streamlit as st
from email import message_from_string
from datetime import datetime, timedelta
from textblob import TextBlob
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
from gtts import gTTS
import tempfile

# Page Configuration
st.title("Comprehensive Email Analytics & RCA Tool")
st.markdown("""
This tool provides advanced analytics for email conversations, focusing on key metrics, RCA, and qualitative insights.
""")

# Upload Files
uploaded_files = st.file_uploader("Upload Email Files (EML format):", type=["eml"], accept_multiple_files=True)

# Helper Functions
def parse_email(email_content):
    """Parses an email and extracts required information."""
    email = message_from_string(email_content.decode("utf-8"))
    sender = email.get("From", "Unknown Sender")
    receiver = email.get("To", "Unknown Receiver")
    date = email.get("Date", None)
    subject = email.get("Subject", "No Subject")
    body = email.get_payload(decode=True).decode(errors="ignore") if email.is_multipart() else email.get_payload()
    return sender, receiver, date, subject, body

def analyze_sentiment(text):
    """Performs sentiment analysis using TextBlob."""
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.1:
        return "Positive"
    elif sentiment < -0.1:
        return "Negative"
    else:
        return "Neutral"

def formality_analysis(text):
    """Analyzes formality level using keywords."""
    informal_words = {"hey", "lol", "thanks", "bye", "ok"}
    words = set(text.lower().split())
    if words & informal_words:
        return "Informal"
    return "Formal"

def analyze_response_times(email_df):
    """Analyzes response times between emails."""
    email_df["Date"] = pd.to_datetime(email_df["Date"], errors="coerce")
    email_df = email_df.sort_values("Date")
    email_df["Response Time"] = email_df["Date"].diff().apply(lambda x: x.total_seconds() / 3600 if pd.notnull(x) else None)
    return email_df

def extract_roles(email_data):
    """Extracts roles and responsibilities based on sender and receiver patterns."""
    role_map = Counter([email["Sender"] for email in email_data])
    responsibilities = {k: f"Frequently communicates ({v} messages)" for k, v in role_map.items()}
    return responsibilities

def identify_entry_exit(email_data):
    """Identifies entry and exit points in the conversation."""
    participants = [email["Sender"] for email in email_data] + [email["Receiver"] for email in email_data]
    unique_participants = list(set(participants))
    entry_points = {p: email_data[0]["Date"] for p in unique_participants if p in email_data[0]["Sender"]}
    exit_points = {p: email_data[-1]["Date"] for p in unique_participants if p in email_data[-1]["Sender"]}
    return entry_points, exit_points

def analyze_contextual_factors(email_df):
    """Analyzes contextual factors based on timeline and communication frequency."""
    timeline = email_df["Date"].dt.date.value_counts().sort_index()
    st.line_chart(timeline)

def qualitative_metrics(email_data):
    """Analyzes qualitative metrics like tone shifts and miscommunication."""
    tone_shifts = []
    for i in range(1, len(email_data)):
        prev_sentiment = email_data[i-1]["Sentiment"]
        current_sentiment = email_data[i]["Sentiment"]
        if prev_sentiment != current_sentiment:
            tone_shifts.append(email_data[i])
    return tone_shifts

def generate_wordcloud(text):
    """Generates a word cloud."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

def generate_rca_audio(narration):
    """Generates RCA narration as audio."""
    tts = gTTS(text=narration, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# Analysis Process
if st.button("Analyze Emails"):
    if not uploaded_files:
        st.error("Please upload at least one email file.")
    else:
        email_data = []
        combined_text = ""
        for uploaded_file in uploaded_files:
            content = uploaded_file.read()
            sender, receiver, date, subject, body = parse_email(content)
            sentiment = analyze_sentiment(body)
            formality = formality_analysis(body)
            email_data.append({
                "Sender": sender,
                "Receiver": receiver,
                "Date": date,
                "Subject": subject,
                "Body": body,
                "Sentiment": sentiment,
                "Formality": formality
            })
            combined_text += body

        email_df = pd.DataFrame(email_data)
        email_df = analyze_response_times(email_df)

        st.subheader("Key Metrics")
        st.metric("Total Emails", len(email_df))
        st.metric("Unique Participants", len(set(email_df["Sender"])))
        avg_response_time = email_df["Response Time"].mean()
        st.metric("Average Response Time (hrs)", round(avg_response_time, 2))

        st.subheader("Roles and Responsibilities")
        roles = extract_roles(email_data)
        st.json(roles)

        st.subheader("Entry and Exit Points")
        entry_points, exit_points = identify_entry_exit(email_data)
        st.write("Entry Points", entry_points)
        st.write("Exit Points", exit_points)

        st.subheader("Tone and Sentiment Analysis")
        tone_shifts = qualitative_metrics(email_data)
        st.write("Tone Shifts Identified:")
        st.dataframe(pd.DataFrame(tone_shifts))

        st.subheader("Contextual Factors")
        analyze_contextual_factors(email_df)

        st.subheader("Word Cloud")
        wordcloud_fig = generate_wordcloud(combined_text)
        st.pyplot(wordcloud_fig)

        st.subheader("RCA Narration")
        narration = f"""
        The analysis revealed {len(email_df)} emails with an average response time of {round(avg_response_time, 2)} hours.
        Key roles and responsibilities were identified, with notable tone shifts in {len(tone_shifts)} instances.
        Contextual factors, such as communication gaps, may indicate external influences or organizational challenges.
        """
        st.write(narration)
        rca_audio = generate_rca_audio(narration)
        st.audio(rca_audio, format="audio/mp3")
