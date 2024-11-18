import streamlit as st
import google.generativeai as genai
from email import message_from_string
from datetime import datetime
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("RCA & Sentiment Analysis")
st.write("""
An advanced solution for analyzing root causes (RCA) and sentiment from email content and generating insights.
""")

# File Upload
uploaded_files = st.file_uploader("Upload Email Files (supports .eml, .msg, .txt):", type=["eml", "msg", "txt"], accept_multiple_files=True)

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"

# Extracting Email Body
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

# Extracting Date from Email
def extract_date(email):
    date = email.get("Date", "Unknown Date")
    try:
        return datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")
    except:
        return datetime.now()

# Process Uploaded Files
if st.button("Generate RCA Insights and Sentiment Analysis"):
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

        # Process RCA (Root Cause Analysis)
        st.write("### Root Cause Analysis (RCA) Insights")
        if not df.empty:
            # Check for frequent patterns in Subject
            subject_keywords = df['Subject'].str.extract(r'(\b[A-Za-z]{4,}\b)', expand=False)
            keyword_counts = subject_keywords.value_counts().head(10)
            st.write("#### Top 10 Keywords in Email Subjects (Potential Root Cause Indicators):")
            st.bar_chart(keyword_counts)

            # Analyze escalation patterns based on negative sentiment
            negative_emails = df[df['Sentiment'] == 'Negative']
            st.write("#### Negative Sentiment Emails (Potential Escalations):")
            st.dataframe(negative_emails)

            # Identify top 5 most frequent culprits (senders of negative emails)
            culprit_counts = negative_emails['From'].value_counts().head(5)
            st.write("#### Top 5 Culprits (Frequent Senders of Negative Emails):")
            st.bar_chart(culprit_counts)

            # Response Time Analysis for Escalation
            negative_emails['Response Time'] = negative_emails['Sent Time'].diff().fillna(pd.Timedelta(seconds=0))
            average_response_time = negative_emails['Response Time'].mean()
            st.write(f"#### Average Response Time for Negative Sentiment Emails: {average_response_time}")

        else:
            st.write("No data available for RCA insights.")
