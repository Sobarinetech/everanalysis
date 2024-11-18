import os
import email
from email import message_from_string
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import pandas as pd
import streamlit as st
from gtts import gTTS

# --- Helper Functions ---
def parse_email(email_content):
    """Parses an email and extracts required information."""
    email_msg = message_from_string(email_content.decode("utf-8"))
    sender = email_msg.get("From", "Unknown Sender")
    receiver = email_msg.get("To", "Unknown Receiver")
    date = email_msg.get("Date", None)
    subject = email_msg.get("Subject", "No Subject")
    
    # Safely handle payload
    payload = email_msg.get_payload(decode=True)
    if payload is not None:
        body = payload.decode(errors="ignore")
    else:
        body = "No Body Found"
    
    return sender, receiver, date, subject, body

def analyze_sentiment(text):
    """Analyzes sentiment of the given text."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive", polarity
    elif polarity < 0:
        return "Negative", polarity
    else:
        return "Neutral", polarity

def generate_wordcloud(text):
    """Generates and saves a word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud", fontsize=16)
    plt.savefig("wordcloud.png")
    plt.close()

def calculate_response_times(df):
    """Calculates response times between email exchanges."""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values(by="Date", inplace=True)
    df["Response Time (hours)"] = df["Date"].diff().dt.total_seconds() / 3600
    return df

def generate_narration(summary):
    """Generates a TTS audio narration for the summary."""
    tts = gTTS(text=summary, lang="en")
    tts.save("narration.mp3")

# --- Main Analysis Logic ---
def analyze_emails(email_folder):
    all_emails = []
    all_bodies = []
    stats = {"Positive": 0, "Negative": 0, "Neutral": 0}

    # Parse each email in the folder
    for filename in os.listdir(email_folder):
        if filename.endswith(".eml"):
            with open(os.path.join(email_folder, filename), "rb") as f:
                email_content = f.read()
            sender, receiver, date, subject, body = parse_email(email_content)
            sentiment, polarity = analyze_sentiment(body)
            all_emails.append({"Sender": sender, "Receiver": receiver, "Date": date, "Subject": subject,
                               "Body": body, "Sentiment": sentiment, "Polarity": polarity})
            all_bodies.append(body)
            stats[sentiment] += 1

    # Convert to DataFrame
    email_df = pd.DataFrame(all_emails)

    # Calculate response times
    email_df = calculate_response_times(email_df)

    # Generate word cloud
    combined_text = " ".join(all_bodies)
    generate_wordcloud(combined_text)

    # Root Cause Analysis and Summary
    rca_summary = perform_rca(email_df)
    generate_narration(rca_summary)

    return email_df, stats, rca_summary

def perform_rca(df):
    """Performs Root Cause Analysis."""
    escalation_triggers = []
    for _, row in df.iterrows():
        if "urgent" in row["Body"].lower() or "immediate" in row["Body"].lower():
            escalation_triggers.append(f"Escalation Trigger in email from {row['Sender']} to {row['Receiver']}")

    top_senders = df["Sender"].value_counts().idxmax()
    top_receivers = df["Receiver"].value_counts().idxmax()

    rca_summary = (
        f"Root Cause Analysis Summary:\n"
        f"Total Emails Analyzed: {len(df)}\n"
        f"Top Sender: {top_senders}\n"
        f"Top Receiver: {top_receivers}\n"
        f"Escalation Triggers: {'; '.join(escalation_triggers) if escalation_triggers else 'None Found'}\n"
    )

    return rca_summary

# --- Streamlit Interface ---
st.title("Email Root Cause Analysis Tool")
st.sidebar.title("Configuration")
email_folder = st.sidebar.text_input("Email Folder Path", "path/to/emails")

if st.sidebar.button("Analyze Emails"):
    if not os.path.exists(email_folder):
        st.error("Invalid email folder path!")
    else:
        st.info("Analyzing emails, please wait...")
        email_df, stats, rca_summary = analyze_emails(email_folder)

        st.success("Analysis Complete!")
        st.write("### Email Data")
        st.dataframe(email_df)

        st.write("### Sentiment Statistics")
        st.bar_chart(pd.DataFrame.from_dict(stats, orient="index", columns=["Count"]))

        st.write("### RCA Summary")
        st.text(rca_summary)

        st.write("### Word Cloud")
        st.image("wordcloud.png")

        st.write("### Narration")
        audio_file = open("narration.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")
