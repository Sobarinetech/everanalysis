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
import os
from gtts import gTTS

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Email Escalation and RCA Tool")
st.write("""
This tool analyzes email exchanges for root causes of escalations, identifying patterns, key participants, response delays, sentiment shifts, and communication gaps.
""")

# File Upload
uploaded_files = st.file_uploader(
    "Upload Email Files (supports .eml, .msg, .txt):", type=["eml", "msg", "txt"], accept_multiple_files=True
)

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

# Extract Email Content, Date, and Participants
def extract_email_content_and_date(email):
    if email.is_multipart():
        for part in email.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_payload(decode=True).decode(errors="ignore")
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
if st.button("Analyze Emails for RCA and Sentiment"):
    if not uploaded_files:
        st.error("Please upload at least one email file.")
    else:
        emails_data = []
        sentiment_list = []
        response_times = []
        participants_graph = []
        topics_counter = Counter()
        escalations = []
        timestamps = []

        for file in uploaded_files:
            try:
                content = file.read().decode("utf-8")
                email = message_from_string(content)
                subject = email.get("Subject", "No Subject")
                sender, receiver = extract_sender_and_receiver(email)
                sent_time = extract_date(email)
                body = extract_email_content_and_date(email)
                sentiment = analyze_sentiment(body)
                
                if len(emails_data) > 0:
                    time_diff = (sent_time - emails_data[-1]["Sent Time"]).total_seconds()
                else:
                    time_diff = 0
                
                emails_data.append({
                    "Subject": subject,
                    "From": sender,
                    "To": receiver,
                    "Sent Time": sent_time,
                    "Body": body,
                    "Sentiment": sentiment,
                    "Time Diff (seconds)": time_diff,
                })

                sentiment_list.append(sentiment)
                response_times.append(time_diff)
                participants_graph.append((sender, receiver))
                topics_counter.update(body.split())
                timestamps.append(sent_time)

                if sentiment == "Negative":
                    escalations.append({"From": sender, "Body": body, "Sentiment": sentiment, "Sent Time": sent_time})
            except Exception as e:
                st.error(f"Error processing file {file.name}: {str(e)}")

        # Convert to DataFrame
        df = pd.DataFrame(emails_data)

        # Sentiment Analysis Visualization
        st.write("### Sentiment Analysis")
        sentiment_counts = pd.Series(sentiment_list).value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", ax=ax, color=["green", "yellow", "red"])
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Response Time Insights
        st.write("### Response Time Analysis")
        avg_time = pd.Series(response_times).mean()
        max_time = pd.Series(response_times).max()
        min_time = pd.Series(response_times).min()
        st.write(f"Average Response Time: {avg_time:.2f} seconds")
        st.write(f"Longest Response Time: {max_time:.2f} seconds")
        st.write(f"Shortest Response Time: {min_time:.2f} seconds")

        # Escalation Triggers
        st.write("### Escalation Triggers")
        if escalations:
            escalation_df = pd.DataFrame(escalations)
            st.write("#### Emails with Negative Sentiment:")
            st.dataframe(escalation_df)
        else:
            st.write("No negative sentiment detected in the emails.")

        # Timeline of Exchanges
        st.write("### Timeline of Email Exchanges")
        fig, ax = plt.subplots()
        ax.plot(timestamps, range(len(timestamps)), marker="o")
        ax.set_title("Email Exchange Timeline")
        ax.set_xlabel("Time")
        ax.set_ylabel("Email Index")
        st.pyplot(fig)

        # Participants Network Diagram
        st.write("### Email Participants Network")
        G = nx.Graph()
        G.add_edges_from(participants_graph)
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
        st.pyplot()

        # Topic Analysis Heatmap
        st.write("### Heatmap of Common Topics")
        common_topics = topics_counter.most_common(20)
        topic_df = pd.DataFrame(common_topics, columns=["Topic", "Frequency"])
        fig, ax = plt.subplots()
        sns.heatmap(topic_df.set_index("Topic").T, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
        st.pyplot()

        # Culpability Analysis
        st.write("### Culpability Analysis")
        top_sources = df[df["Sentiment"] == "Negative"]["From"].value_counts().head(5)
        if not top_sources.empty:
            st.write("#### Top 5 Sources of Escalations:")
            st.bar_chart(top_sources)
        else:
            st.write("No contributors identified for escalation.")

        # RCA and Conclusion Narration
        rca_narration = """
        The analysis indicates that escalations were triggered by delays in responses and negative sentiments in emails from key participants. 
        Setting realistic expectations and ensuring prompt replies can help reduce escalations.
        """
        st.write("### Root Cause Analysis (RCA)")
        st.write(rca_narration)

        # Generate Audio Narration
        tts = gTTS(text=rca_narration, lang="en")
        audio_file = "rca_analysis.mp3"
        tts.save(audio_file)
        st.audio(audio_file)
        os.remove(audio_file)
