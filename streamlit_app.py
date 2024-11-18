import streamlit as st
import google.generativeai as genai
from email import message_from_string
from datetime import datetime
import pandas as pd
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import Counter
from PyPDF2 import PdfReader
from gtts import gTTS
import os

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("RCA & Escalation Analysis from Email Data")
st.write("""
An advanced solution for analyzing root causes (RCA) and escalation patterns from email content and attachments.
""")

# File Upload
uploaded_files = st.file_uploader("Upload Email Files (supports .eml, .msg, .txt, .pdf):", type=["eml", "msg", "txt", "pdf"], accept_multiple_files=True)

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"

# Extracting Email Body and PDF Attachment
def extract_email_body(email):
    if email.is_multipart():
        for part in email.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    return part.get_payload(decode=True).decode(errors="ignore")
                except:
                    continue
    else:
        try:
            return email.get_payload(decode=True).decode(errors="ignore")
        except:
            return "No content available."

def extract_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

# Extracting Date from Email
def extract_date(email):
    date = email.get("Date", "Unknown Date")
    try:
        return datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")
    except:
        return datetime.now()

# Process Uploaded Files
if st.button("Generate RCA Insights"):
    if not uploaded_files:
        st.error("Please upload at least one email file.")
    else:
        all_emails_summary = []
        sentiment_data = []

        # Process each uploaded file
        for file in uploaded_files:
            content = file.read().decode("utf-8") if file.name.endswith(('eml', 'msg', 'txt')) else ""
            email = message_from_string(content) if content else None
            subject = email.get("Subject", "No Subject") if email else "No Subject"
            from_email = email.get("From", "Unknown Sender") if email else "Unknown Sender"
            sent_time = extract_date(email) if email else datetime.now()
            body = extract_email_body(email) if email else ""
            pdf_text = extract_pdf_text(file) if file.name.endswith("pdf") else ""

            # Combine email body and PDF text
            full_body = body + "\n" + pdf_text
            sentiment = analyze_sentiment(full_body)

            # Append data for summary
            all_emails_summary.append({
                "Subject": subject,
                "From": from_email,
                "Sent Time": sent_time,
                "Body": full_body,
                "Sentiment": sentiment
            })
            sentiment_data.append(sentiment)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_emails_summary)

        # Sentiment Analysis
        st.write("### Sentiment Analysis")
        sentiment_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'yellow', 'red'])
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        # RCA Analysis and Identification of Culprits
        st.write("### RCA and Culprit Identification")
        negative_emails = df[df['Sentiment'] == 'Negative']
        
        if not negative_emails.empty:
            st.write("#### Negative Sentiment Emails (Potential Escalations):")
            st.dataframe(negative_emails)
            
            # Identify potential culprits (frequent senders in negative sentiment emails)
            culprit_counts = negative_emails['From'].value_counts().head(5)
            st.write("#### Top 5 Culprits (Frequent Senders of Negative Emails):")
            st.bar_chart(culprit_counts)

            # Culpability Analysis
            culpability_analysis = {}
            for sender, count in culprit_counts.items():
                # Analyze the frequency and tone of communication
                sender_emails = negative_emails[negative_emails['From'] == sender]
                tone_shift = sender_emails['Sentiment'].value_counts().to_dict()  # Track sentiment shifts
                avg_response_time = sender_emails['Sent Time'].diff().mean()  # Calculate average response time
                
                culpability_analysis[sender] = {
                    "Frequency": count,
                    "Tone Shifts": tone_shift,
                    "Avg Response Time": avg_response_time
                }
                
            st.write("#### Culpability Analysis (Based on Frequency, Tone Shifts, and Response Time):")
            st.write(culpability_analysis)

        # RCA Narrative Generation (using AI Model)
        if st.button("Generate RCA Narrative"):
            rca_data = negative_emails[['Subject', 'From', 'Sent Time', 'Body']].to_dict(orient='records')
            prompt = f"Analyze these emails for root cause patterns and escalations:\n{rca_data}"
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            st.write("### RCA Narrative:")
            st.write(response.text)

            # Text-to-Speech for the RCA Narrative
            tts = gTTS(text=response.text, lang='en', slow=False)
            audio_path = "rca_narrative.mp3"
            tts.save(audio_path)

            # Provide the link for the user to download the audio
            st.audio(audio_path, format="audio/mp3")
