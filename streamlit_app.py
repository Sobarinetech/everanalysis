import streamlit as st
import google.generativeai as genai
from email import message_from_string
from datetime import datetime
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import seaborn as sns

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Email Escalation and Sentiment Analysis")
st.write("""
This tool performs detailed analysis on email exchanges, identifying escalation triggers, sentiment shifts, response times, and key topics to help root cause escalation issues.
""")

# File Upload
uploaded_files = st.file_uploader("Upload Email Files (supports .eml, .msg, .txt):", type=["eml", "msg", "txt"], accept_multiple_files=True)

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"

# Extract Email Body and Sent Date
def extract_email_content_and_date(email):
    if email.is_multipart():
        for part in email.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    return body
                except:
                    continue
    else:
        try:
            return email.get_payload(decode=True).decode(errors="ignore")
        except:
            return "No content available."
    
    return "No content available."

def extract_date(email):
    date = email.get("Date", "Unknown Date")
    try:
        return datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")
    except:
        return datetime.now()

def extract_sender_and_receiver(email):
    sender = email.get("From", "Unknown Sender")
    receiver = email.get("To", "Unknown Receiver")
    return sender, receiver

# Process Uploaded Files
if st.button("Generate RCA Insights and Sentiment Analysis"):
    if not uploaded_files:
        st.error("Please upload at least one email file.")
    else:
        all_emails_summary = []
        sentiment_data = []
        response_times = []
        email_participants = []
        topics_counter = Counter()
        timelines = []
        senders = []
        receivers = []

        # Process each uploaded file
        for file in uploaded_files:
            content = file.read().decode("utf-8") if file.name.endswith(('eml', 'msg', 'txt')) else ""
            email = message_from_string(content) if content else None
            subject = email.get("Subject", "No Subject") if email else "No Subject"
            from_email, to_email = extract_sender_and_receiver(email) if email else ("Unknown Sender", "Unknown Receiver")
            sent_time = extract_date(email) if email else datetime.now()
            body = extract_email_content_and_date(email) if email else ""
            
            sentiment = analyze_sentiment(body)

            # Time elapsed between responses
            if len(all_emails_summary) > 0:
                time_diff = (sent_time - all_emails_summary[-1]['Sent Time']).total_seconds()
            else:
                time_diff = 0

            # Extract topics for further analysis
            topics = body.split()
            topics_counter.update(topics)

            # Append data for summary
            all_emails_summary.append({
                "Subject": subject,
                "From": from_email,
                "To": to_email,
                "Sent Time": sent_time,
                "Body": body,
                "Sentiment": sentiment,
                "Time Diff (seconds)": time_diff,
            })
            sentiment_data.append(sentiment)
            response_times.append(time_diff)
            email_participants.append((from_email, to_email))
            senders.append(from_email)
            receivers.append(to_email)
            timelines.append(sent_time)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_emails_summary)

        # Sentiment Analysis Visualization
        st.write("### Sentiment Analysis")
        sentiment_counts = pd.Series(sentiment_data).value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'yellow', 'red'])
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        # Response Time Analysis
        st.write("### Response Time Analysis")
        avg_response_time = pd.Series(response_times).mean()
        longest_response_time = pd.Series(response_times).max()
        shortest_response_time = pd.Series(response_times).min()
        st.write(f"Average Response Time: {avg_response_time:.2f} seconds")
        st.write(f"Longest Response Time: {longest_response_time:.2f} seconds")
        st.write(f"Shortest Response Time: {shortest_response_time:.2f} seconds")

        # Escalation Triggers: Analyze for key phrases and changes in tone
        st.write("### Escalation Triggers")
        negative_emails = df[df['Sentiment'] == 'Negative']
        if not negative_emails.empty:
            st.write("#### Negative Sentiment Emails (Potential Escalations):")
            st.dataframe(negative_emails)
        
        # Plot timeline of email exchanges
        st.write("### Timeline of Email Exchanges")
        fig, ax = plt.subplots()
        ax.plot(timelines, range(len(timelines)), marker='o')
        ax.set_title('Email Exchange Timeline')
        ax.set_xlabel('Time')
        ax.set_ylabel('Email Index')
        st.pyplot(fig)

        # Network Diagram: Participants in email exchange
        st.write("### Network Diagram of Participants")
        G = nx.Graph()
        for sender, receiver in email_participants:
            G.add_edge(sender, receiver)
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
        st.pyplot()

        # Heatmap for Topic Frequency Analysis
        st.write("### Heatmap for Topic Frequency Analysis")
        top_topics = topics_counter.most_common(20)
        topics_df = pd.DataFrame(top_topics, columns=["Topic", "Frequency"])
        fig, ax = plt.subplots()
        sns.heatmap(topics_df.set_index("Topic").T, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
        st.pyplot()

        # Culpability Analysis (Sender Contribution to Escalations)
        st.write("### Culpability Analysis")
        top_senders = negative_emails['From'].value_counts().head(5)
        st.write("#### Top 5 Senders (Frequent Sources of Negative Emails):")
        st.bar_chart(top_senders)

        # Additional Metrics
        st.write("### Additional Metrics")
        total_emails = len(df)
        total_participants = len(set(senders + receivers))
        st.write(f"Total Emails Processed: {total_emails}")
        st.write(f"Total Participants: {total_participants}")

        # Identifying unrealistic expectations or miscommunications
        unrealistic_expectations = df[df['Body'].str.contains("unrealistic expectation", case=False, na=False)]
        st.write("### Unrealistic Expectations or Miscommunications")
        if not unrealistic_expectations.empty:
            st.dataframe(unrealistic_expectations)
        else:
            st.write("No instances of unrealistic expectations or miscommunications identified.")
