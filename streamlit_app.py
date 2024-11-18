import streamlit as st
import google.generativeai as genai
from email import message_from_string
from datetime import datetime
from io import BytesIO
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

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

# Additional Features: Functions
def keyword_analysis(body):
    """Extract frequently occurring keywords."""
    words = body.split()
    keywords = pd.Series(words).value_counts().head(10)
    return keywords

def email_count_by_sender(df):
    """Count emails grouped by sender."""
    return df['From'].value_counts()

def sentiment_distribution(sentiments):
    """Visualize sentiment distribution."""
    sentiment_df = pd.DataFrame(sentiments, columns=["Sentiment"])
    distribution = sentiment_df["Sentiment"].value_counts()
    return distribution

# Process Uploaded Files
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
            
            # Load and configure the model
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Generate response from the model
            response = model.generate_content(prompt)

            # Extract and display the generated narrative
            st.write("### AI-Generated Narrative:")
            st.write(response.text)

            # 1. Keyword Analysis
            st.write("### Frequently Occurring Keywords")
            all_bodies = " ".join(df["Body"].dropna().tolist())
            keywords = keyword_analysis(all_bodies)
            st.bar_chart(keywords)

            # 2. Email Count by Sender
            st.write("### Email Count by Sender")
            email_counts = email_count_by_sender(df)
            st.bar_chart(email_counts)

            # 3. Sentiment Distribution
            st.write("### Sentiment Distribution")
            sentiment_distribution_data = sentiment_distribution(sentiment_data)
            st.bar_chart(sentiment_distribution_data)

            # 4. Weekly Email Trends
            st.write("### Weekly Email Trends")
            df["Sent Time"] = pd.to_datetime(df["Sent Time"], errors="coerce")
            df["Week"] = df["Sent Time"].dt.isocalendar().week
            weekly_emails = df.groupby("Week").size()
            st.line_chart(weekly_emails)

            # 5. Top 5 Longest Emails
            st.write("### Top 5 Longest Emails")
            df["Body Length"] = df["Body"].apply(lambda x: len(x) if x else 0)
            top_longest = df.nlargest(5, "Body Length")
            st.dataframe(top_longest[["Subject", "From", "Body Length"]])

            # 6. Customizable Filters
            st.write("### Filter Emails by Sender")
            unique_senders = df["From"].dropna().unique()
            sender_filter = st.selectbox("Select a Sender to View Emails:", unique_senders)
            filtered_emails = df[df["From"] == sender_filter]
            st.dataframe(filtered_emails)

            # 7. Export Filtered Emails to CSV
            if st.button("Download Filtered Emails"):
                filtered_output = BytesIO()
                filtered_emails.to_csv(filtered_output, index=False)
                filtered_output.seek(0)
                st.download_button(
                    label="Download Filtered Emails as CSV",
                    data=filtered_output,
                    file_name="filtered_emails.csv",
                    mime="text/csv"
                )

            # 8. Reply Time Analysis
            st.write("### Average Reply Time")
            reply_times = df["Sent Time"].dropna().diff().mean()
            st.write(f"Average Reply Time: {reply_times}")

            # 9. Time-Zone Based Analysis
            st.write("### Emails by Time Zone")
            time_zones = df["Sent Time"].dt.tz_localize(None).dt.hour.value_counts()
            st.bar_chart(time_zones)

            # 10. Visualize Sentiment Over Time
            st.write("### Sentiment Over Time")
            df["Sentiment Score"] = df["Body"].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)
            df["Date"] = df["Sent Time"].dt.date
            sentiment_over_time = df.groupby("Date")["Sentiment Score"].mean()
            st.line_chart(sentiment_over_time)

        except Exception as e:
            st.error(f"An error occurred: {e}")
