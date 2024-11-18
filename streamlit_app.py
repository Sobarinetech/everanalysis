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
        else:
            st.write("No negative sentiment emails found.")

        # Topic Modeling with LDA (Latent Dirichlet Allocation)
        st.write("### Topic Modeling (LDA) for Root Cause Insights")
        
        # Filter out empty documents
        valid_emails = negative_emails['Body'].dropna().str.strip()
        valid_emails = valid_emails[valid_emails.str.split().str.len() > 1]  # Remove too short documents

        # Vectorize with TF-IDF
        if valid_emails.empty:
            st.write("No valid emails for topic modeling.")
        else:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
            X = vectorizer.fit_transform(valid_emails)

            # Check if the vocabulary is empty
            if X.shape[0] > 0:
                lda = LDA(n_components=3, random_state=42)
                lda.fit(X)

                topics = []
                for idx, topic in enumerate(lda.components_):
                    topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
                    topics.append(f"Topic {idx + 1}: " + ", ".join(topic_words))

                st.write("#### Topics Detected in Negative Sentiment Emails:")
                for topic in topics:
                    st.write(topic)
            else:
                st.write("Topic modeling failed due to lack of content.")

        # Escalation Pattern Detection
        st.write("### Escalation Patterns Over Time")
        df['Weekday'] = df['Sent Time'].dt.day_name()
        escalations_by_day = negative_emails.groupby('Weekday').size().sort_values(ascending=False)
        st.write("#### Escalations by Day of the Week:")
        st.bar_chart(escalations_by_day)

        # Frequency of Escalation Emails
        st.write("### Frequency of Escalation Emails")
        escalation_frequency = negative_emails.groupby(negative_emails['Sent Time'].dt.date).size()
        st.line_chart(escalation_frequency)

        # Email Thread Analysis (Root Cause in Email Chains)
        st.write("### Email Thread Analysis")
        # Assuming emails are part of a thread if the subject line has 'Re:'
        thread_emails = df[df['Subject'].str.contains('Re:', na=False)]
        st.write("#### Emails in Threads (Possible Escalations in Ongoing Conversations):")
        st.dataframe(thread_emails)

        # Escalation Specific Metrics: Email Response Time
        st.write("### Email Response Time Analysis")
        df['Response Time'] = df['Sent Time'].diff().fillna(pd.Timedelta(seconds=0))
        average_response_time = df['Response Time'].mean()
        st.write(f"#### Average Email Response Time: {average_response_time}")

        # Root Cause Insights (Based on Sentiment & Subject)
        st.write("### Root Cause Insights Based on Sentiment & Subject")
        subject_keywords = negative_emails['Subject'].str.extract(r'(\b[A-Za-z]{4,}\b)', expand=False)
        keyword_counts = subject_keywords.value_counts().head(10)
        st.write("#### Top 10 Keywords in Email Subjects (Root Cause Indicators):")
        st.bar_chart(keyword_counts)

        # Generate RCA Narrative using Gemini API
        if st.button("Generate RCA Narrative"):
            rca_data = negative_emails[['Subject', 'From', 'Sent Time', 'Body']].to_dict(orient='records')
            prompt = f"Analyze these emails for root cause patterns and escalations:\n{rca_data}"
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            st.write("### RCA Narrative:")
            st.write(response.text)
