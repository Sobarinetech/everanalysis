import streamlit as st
import google.generativeai as genai
from email import message_from_string
from datetime import datetime
from io import BytesIO
import pandas as pd
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
    "Upload Email Files (supports .eml, .msg, or .txt):", 
    type=["eml", "msg", "txt"], 
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

# Function to extract email body safely
def extract_email_body(email):
    """Extracts the email body, handling multipart emails and None content."""
    if email.is_multipart():
        for part in email.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            # Only process text/plain parts
            if content_type == "text/plain" and "attachment" not in content_disposition:
                try:
                    return part.get_payload(decode=True).decode(errors="ignore")
                except:
                    continue
    else:
        # Handle non-multipart emails
        payload = email.get_payload()
        if payload:
            try:
                return payload.decode(errors="ignore") if isinstance(payload, bytes) else payload
            except:
                pass
    return "No content available."

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
                body = extract_email_body(email)

                # Sentiment Analysis
                sentiment = analyze_sentiment(body)

                # Save data for summary and visualization
                all_emails_summary.append({
                    "Subject": subject,
                    "From": from_email,
                    "To": to_email,
                    "Sent Time": sent_time,
                    "Reply-To": reply_to,
                    "Body": body,
                    "Sentiment": sentiment
                })
                sentiment_data.append(sentiment)

            # Convert to DataFrame for visualization
            df = pd.DataFrame(all_emails_summary)

            # Display email metadata
            st.write("### Email Metadata")
            st.dataframe(df)

            # Generate Narrative with Gemini API
            prompt = (
                "Using the following email data, analyze for timing, replies, "
                "sent details, sentiments, and create an engaging enterprise-level narrative:\n\n"
                f"{df.to_string(index=False)}"
            )
            
            # Call Gemini API for content generation
            response = genai.generate_content(
                model="models/text-bison-001", 
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract the generated narrative
            narrative = response.content if hasattr(response, 'content') else "No content generated."

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
