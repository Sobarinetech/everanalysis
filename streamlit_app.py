import streamlit as st
import google.generativeai as genai
from email import message_from_string
from datetime import datetime
import pandas as pd
from textblob import TextBlob
import re
import matplotlib.pyplot as plt

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("RCA and Escalation Analysis for Email Data")
st.write("""
A focused analysis tool to extract actionable insights from email data 
to help with RCA and escalation analysis.
""")

# File Upload
uploaded_files = st.file_uploader(
    "Upload Email Files (supports .eml, .msg, or .txt):", 
    type=["eml", "msg", "txt"], 
    accept_multiple_files=True
)

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"

# Function to extract email body
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

# Process Uploaded Files
if st.button("Generate RCA Insights"):
    if not uploaded_files:
        st.error("Please upload at least one email file.")
    else:
        all_emails_summary = []
        sentiment_data = []

        # Process each uploaded file
        for file in uploaded_files:
            content = file.read().decode("utf-8")
            email = message_from_string(content)
            subject = email.get("Subject", "No Subject")
            from_email = email.get("From", "Unknown Sender")
            sent_time = email.get("Date", "Unknown Date")
            body = extract_email_body(email)

            # Sentiment Analysis
            sentiment = analyze_sentiment(body)

            # Append data for summary
            all_emails_summary.append({
                "Subject": subject,
                "From": from_email,
                "Sent Time": sent_time,
                "Body": body,
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

        # Email Activity Over Time
        st.write("### Email Activity Over Time")
        df['Sent Time'] = pd.to_datetime(df['Sent Time'], errors='coerce')
        activity_trend = df['Sent Time'].dt.date.value_counts().sort_index()
        st.line_chart(activity_trend)

        # Escalation Metrics
        st.write("### Escalation Metrics")
        escalations = df[df['Sentiment'] == 'Negative']
        if not escalations.empty:
            st.write("Escalations (Negative Sentiment Emails):")
            st.dataframe(escalations)
        else:
            st.write("No negative sentiment emails found.")

        # Extracting key RCA insights
        if st.button("Extract RCA Insights"):
            rca_insights = escalations[['Subject', 'From', 'Sent Time', 'Body']].to_dict(orient='records')
            st.json(rca_insights)

            # Generate RCA Narrative using Gemini API
            prompt = f"Analyze these emails for root cause patterns and escalations:\n{rca_insights}"
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            st.write("### RCA Narrative")
            st.write(response.text)
