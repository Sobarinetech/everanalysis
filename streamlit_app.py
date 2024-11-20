import streamlit as st
import pandas as pd
import re
from email import policy
from email.parser import BytesParser
from io import BytesIO
from datetime import datetime
import spacy  # Using spaCy for NLP and Named Entity Recognition
import matplotlib.pyplot as plt
import networkx as nx
from gtts import gTTS
import tempfile
import os

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

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

# Helper: Analyze sentiment (based on polarity)
def analyze_sentiment(text):
    if not text:
        return "Neutral"
    # Basic sentiment analysis (could be enhanced by more advanced techniques)
    polarity = text.lower().count("urgent") - text.lower().count("please")  # Simplified sentiment
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    return "Neutral"

# Helper: Analyze escalation triggers and root cause (enhanced)
def analyze_escalation_and_rca(df):
    escalation_triggers = []
    rca_summary = []
    unresolved_issues = []

    for _, row in df.iterrows():
        body = row['Body']
        doc = nlp(body)  # Using spaCy for NER (Named Entity Recognition)
        
        # Detect escalation-related words and entities
        if re.search(r"\burgent\b|\bASAP\b|\bimmediate\b", body, re.IGNORECASE):
            escalation_triggers.append(f"Escalation triggered in email from {row['Sender']} to {row['Receiver']}")
        
        # Detect named entities that might indicate responsibility (e.g., project names, individuals)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
        if entities:
            unresolved_issues.append(f"Unresolved issues detected in email from {row['Sender']} mentioning: {', '.join(entities)}")

        # Check if this email appears to be a follow-up without resolution
        if re.search(r"still waiting|no response|following up", body, re.IGNORECASE):
            unresolved_issues.append(f"Unresolved issue detected: {row['Subject']} (from {row['Sender']} to {row['Receiver']})")

    # Provide summary based on escalations and unresolved issues
    rca_summary.append(f"Escalation Triggers Detected: {', '.join(escalation_triggers) if escalation_triggers else 'None'}")
    rca_summary.append(f"Unresolved Issues: {', '.join(unresolved_issues) if unresolved_issues else 'None'}")

    return '\n'.join(rca_summary)

# Helper: Generate TTS (Text-to-Speech)
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
st.title("Enhanced Email Escalation and RCA Tool")
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
    
    # Perform escalation and RCA analysis
    rca_summary = analyze_escalation_and_rca(df)

    st.subheader("Email Data Overview")
    st.dataframe(df)

    st.subheader("Root Cause Analysis (RCA)")
    st.text(rca_summary)

    audio_file = generate_tts(rca_summary)
    if audio_file:
        st.subheader("RCA Narration")
        st.audio(audio_file, format="audio/mp3")
        os.unlink(audio_file)  # Clean up the temporary file
