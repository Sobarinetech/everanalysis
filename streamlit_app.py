import streamlit as st
import google.generativeai as genai
from email import message_from_string
from datetime import datetime
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Enterprise AI-Powered Data Storytelling")
st.write("""
An enterprise-grade solution for analyzing and transforming email data 
into actionable insights and engaging narratives.
""")

# File Upload
uploaded_files = st.file_uploader(
    "Upload Email Files (supports .eml, .msg, .txt, or zip archives):", 
    type=["eml", "msg", "txt", "zip"], 
    accept_multiple_files=True
)

# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# Button to process and generate insights
if st.button("Generate Enterprise Insights"):
    if not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        try:
            all_emails_summary = []
            sentiment_data = []

            # Process each uploaded file
            for file in uploaded_files:
                content = file.read().decode("utf-8")
                email = message_from_string(content)
                
                # Extract metadata
                subject = email.get("Subject", "No Subject")
                from_email = email.get("From", "Unknown Sender")
                to_email = email.get("To", "Unknown Recipient")
                sent_time = email.get("Date", "Unknown Date")
                reply_to = email.get("Reply-To", "No Reply-To Address")
                body = email.get_payload()

                # Sentiment Analysis
                sentiment = analyze_sentiment(body)

                # Save data for summary and visualization
                all_emails_summary.append({
                    "Subject": subject,
                    "From": from_email,
                    "To": to_email,
                    "Sent Time": sent_time,
                    "Reply-To": reply_to,
                    "Sentiment": sentiment
                })
                sentiment_data.append(sentiment)

            # Convert to DataFrame for visualization
            df = pd.DataFrame(all_emails_summary)

            # Generate Visual Insights
            st.write("### Sentiment Analysis Breakdown")
            sentiment_counts = pd.Series(sentiment_data).value_counts()
            st.bar_chart(sentiment_counts)

            st.write("### Email Metadata")
            st.dataframe(df)

            # Generate Narrative with Gemini API
            prompt = (
                "Using the following email data, analyze for timing, replies, "
                "sent details, sentiments, and create an engaging enterprise-level narrative:\n\n"
                f"{df.to_string(index=False)}"
            )
            response = genai.generate_text(model="models/text-bison-001", prompt=prompt)

            # Ensure response.result is handled properly
            if isinstance(response.result, list):
                narrative = "\n".join(response.result)  # Join list items into a single string
            else:
                narrative = response.result  # Directly use as a string

            st.write("### AI-Generated Narrative:")
            st.write(narrative)

            # Export Option
            if st.button("Download Insights as Excel"):
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Email Insights")
                output.seek(0)
                st.download_button(
                    label="Download Insights",
                    data=output,
                    file_name="email_insights.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
