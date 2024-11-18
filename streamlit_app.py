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
st.title("RCA & Email Data Storytelling")
st.write("""
An advanced solution for analyzing root causes (RCA) and escalation patterns from email content and generating a detailed AI-based data story.
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
if st.button("Generate RCA Insights and Data Story"):
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

        # Ensure Sent Time column is in datetime format
        df['Sent Time'] = pd.to_datetime(df['Sent Time'], errors='coerce')

        # Check if Sent Time column exists and convert to datetime format
        if df['Sent Time'].isnull().any():
            st.error("There is an issue with the Sent Time column. It contains invalid dates.")

        # Process RCA & Escalation Analysis

        # Root Cause Analysis (RCA) Insights
        st.write("### RCA Insights")
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

            # Email Thread Analysis (Root Cause in Email Chains)
            st.write("### Email Thread Analysis (Escalation in Ongoing Conversations)")
            thread_emails = df[df['Subject'].str.contains('Re:', na=False)]
            st.write("#### Emails in Threads (Potential Escalations in Ongoing Conversations):")
            st.dataframe(thread_emails)

        else:
            st.write("No data available for RCA insights.")

        # Topic Modeling with LDA (Latent Dirichlet Allocation) for identifying root causes
        st.write("### Topic Modeling (LDA) for Root Cause Insights")

        # Filter out empty documents
        valid_emails = df['Body'].dropna().str.strip()
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

        # AI-based Data Storytelling
        st.write("### AI-Generated Data Story")

        # Prepare the RCA data for storytelling
        rca_data = df[['Subject', 'From', 'Sent Time', 'Body']].to_dict(orient='records')

        # Generate a prompt for the AI model to generate a data story
        prompt = f"""
        Based on the following email data, generate a comprehensive AI-based narrative that tells the story of the root causes and escalation patterns:
        {rca_data}

        The narrative should include:
        - Key insights from the email subjects, such as common themes or recurring issues.
        - Identification of potential causes for escalations or negative sentiments.
        - A detailed breakdown of who is involved in escalations and any response delays.
        - An actionable conclusion with recommendations for preventing future issues.
        """

        # Call the Gemini model to generate the data story
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)

        # Display the AI-generated story
        st.write("### AI-Generated RCA Narrative:")
        st.write(response.text)
