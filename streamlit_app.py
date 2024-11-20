import streamlit as st
import pandas as pd
import spacy
import re
import matplotlib.pyplot as plt
import networkx as nx
from email import policy
from email.parser import BytesParser
from io import BytesIO
from datetime import datetime
import google.generativeai as genai

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to call Gemini API for content generation or analysis
def call_gemini_api(text):
    try:
        # Load and configure the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response from the model using the provided text
        response = model.generate_content(text)
        
        # Return the generated response text
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# Streamlit App Setup
st.title("Enhanced Email RCA Tool")
st.write("Upload your email files (EML format) for detailed analysis.")

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
    sentiment = nlp(text)._.sentiment
    return sentiment

# Helper: Perform escalation analysis on email body
def analyze_escalation(text):
    escalation_keywords = r"\burgent\b|\bASAP\b|\bimmediate\b"
    if re.search(escalation_keywords, text, re.IGNORECASE):
        return True
    return False

# Helper: Perform Named Entity Recognition (NER)
def extract_entities(text):
    doc = nlp(text)
    entities = {
        "PERSON": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
        "ORG": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
        "GPE": [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    }
    return entities

# Helper: Calculate response times
def calculate_response_times(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values(by="Date", inplace=True)
    df["Response Time (hours)"] = df["Date"].diff().dt.total_seconds() / 3600
    return df

# Helper: Perform Root Cause Analysis (RCA)
def perform_rca(df):
    if df.empty or len(df) < 2:
        return "Insufficient data for Root Cause Analysis."

    # Identify escalation emails
    escalation_emails = df[df['Escalation'] == True]
    
    # Sentiment Analysis Overview
    sentiment_counts = df["Sentiment"].value_counts().to_dict()
    negative_emails = df[df["Sentiment"] == "Negative"]

    rca_summary = (
        f"Root Cause Analysis Summary:\n"
        f"Total Emails Analyzed: {len(df)}\n"
        f"Escalation Emails: {len(escalation_emails)}\n"
        f"Sentiment Overview: {sentiment_counts}\n"
        f"Negative Emails: {len(negative_emails)}\n"
        f"Escalation Triggers: {escalation_emails[['Sender', 'Receiver', 'Subject']].to_string(index=False)}"
    )
    
    # Optionally use Gemini to analyze or summarize RCA
    gemini_analysis = call_gemini_api(rca_summary)
    if gemini_analysis:
        rca_summary += "\nGemini Analysis:\n" + gemini_analysis
    
    return rca_summary

# Streamlit file uploader
uploaded_files = st.file_uploader("Upload Emails", type=["eml"], accept_multiple_files=True)

if uploaded_files:
    data = []
    
    for file in uploaded_files:
        email_bytes = file.read()
        sender, receiver, date, subject, body = parse_email(email_bytes)
        
        # Analyze sentiment
        sentiment = analyze_sentiment(body)
        
        # Analyze escalation triggers
        escalation = analyze_escalation(body)
        
        # Extract entities (persons, organizations, etc.)
        entities = extract_entities(body)
        
        # Append to data
        data.append({
            "Sender": sender,
            "Receiver": receiver,
            "Date": date,
            "Subject": subject,
            "Body": body,
            "Sentiment": sentiment,
            "Escalation": escalation,
            "Entities": entities
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    df = calculate_response_times(df)

    # Display Email Data
    st.subheader("Email Data Overview")
    st.dataframe(df)

    # Visualizations
    st.subheader("Visualizations")
    # Sentiment Pie Chart
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    # Response Time Chart
    st.line_chart(df["Response Time (hours)"])

    # Network Graph (Sender -> Receiver)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["Sender"], row["Receiver"])
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
    st.pyplot(plt)

    # Root Cause Analysis Summary
    st.subheader("Root Cause Analysis (RCA)")
    rca_summary = perform_rca(df)
    st.text(rca_summary)
