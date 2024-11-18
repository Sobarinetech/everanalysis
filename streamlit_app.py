import streamlit as st
from email import message_from_string
from datetime import datetime
from textblob import TextBlob
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gtts import gTTS
from PyPDF2 import PdfReader
import networkx as nx
import os
import tempfile

# Streamlit Page Setup
st.title("Advanced RCA & Escalation Analysis")
st.markdown("""
This tool analyzes emails to extract key metrics, perform RCA, and provide actionable insights using advanced data storytelling.
""")

# Upload Files
uploaded_files = st.file_uploader("Upload Email Files (EML, MSG, TXT, PDF):", type=["eml", "msg", "txt", "pdf"], accept_multiple_files=True)

# Helper Functions
def extract_email_body(email):
    """Extracts email body from EML file."""
    if email.is_multipart():
        for part in email.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode(errors="ignore")
    else:
        return email.get_payload(decode=True).decode(errors="ignore")

def extract_pdf_text(pdf_file):
    """Extracts text from PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def analyze_sentiment(text):
    """Performs sentiment analysis using TextBlob."""
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

def extract_date(email):
    """Extracts the date from an email."""
    date = email.get("Date", "Unknown Date")
    try:
        return datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")
    except ValueError:
        return None

def generate_wordcloud(text):
    """Generates a word cloud."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

# Analysis Variables
email_data = []
sentiments = []
word_frequencies = Counter()

# Process Uploaded Files
if st.button("Run Analysis"):
    if not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        # Iterate over uploaded files
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_content = uploaded_file.read()
            
            if file_name.endswith(".eml"):
                email = message_from_string(file_content.decode("utf-8"))
                body = extract_email_body(email)
                subject = email.get("Subject", "No Subject")
                sender = email.get("From", "Unknown Sender")
                date = extract_date(email)
            elif file_name.endswith(".pdf"):
                body = extract_pdf_text(uploaded_file)
                subject = "No Subject"
                sender = "Unknown Sender"
                date = None
            else:
                st.error(f"Unsupported file type: {file_name}")
                continue

            sentiment = analyze_sentiment(body)
            sentiments.append(sentiment)
            word_frequencies.update(body.split())

            email_data.append({
                "File Name": file_name,
                "Sender": sender,
                "Subject": subject,
                "Body": body,
                "Date": date,
                "Sentiment": sentiment
            })

        # Convert to DataFrame
        email_df = pd.DataFrame(email_data)

        # Display Data
        st.subheader("Email Data")
        st.dataframe(email_df)

        # Sentiment Analysis
        st.subheader("Sentiment Analysis")
        sentiment_counts = pd.Series(sentiments).value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", ax=ax, color=["green", "red", "gray"])
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Word Cloud
        st.subheader("Word Cloud")
        wordcloud_fig = generate_wordcloud(" ".join(word_frequencies.keys()))
        st.pyplot(wordcloud_fig)

        # Escalation Triggers
        st.subheader("Escalation Triggers")
        escalation_triggers = email_df[email_df["Sentiment"] == "Negative"]
        if not escalation_triggers.empty:
            st.write("Emails with Negative Sentiment:")
            st.dataframe(escalation_triggers)
        else:
            st.write("No escalation triggers found.")

        # Network Diagram
        st.subheader("Participant Network")
        participants = email_df["Sender"].value_counts()
        G = nx.DiGraph()
        for sender in participants.index:
            G.add_node(sender)
        for _, row in email_df.iterrows():
            G.add_edge(row["Sender"], "Receiver")
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(G, with_labels=True, node_color="lightblue", ax=ax, font_size=10)
        st.pyplot(fig)

        # Culpability Analysis
        st.subheader("Culpability Analysis")
        top_senders = participants.head(5)
        st.write("Top Senders of Negative Emails:")
        st.bar_chart(top_senders)

        # Generate Narration
        st.subheader("Generate RCA Narration")
        if st.button("Create RCA Narration"):
            narration_text = f"""
            The analysis revealed {len(email_data)} emails. The sentiment distribution showed 
            {sentiment_counts['Positive']} positive, {sentiment_counts['Negative']} negative, 
            and {sentiment_counts['Neutral']} neutral emails. The most frequent sender is 
            {participants.idxmax()} with {participants.max()} emails.
            """
            st.write(narration_text)

            # Text-to-Speech
            tts = gTTS(text=narration_text, lang="en")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)
            st.audio(temp_file.name, format="audio/mp3")
