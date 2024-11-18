import streamlit as st
import google.generativeai as genai
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Configure the API key securely from Streamlit's secrets
# Make sure to add GOOGLE_API_KEY in secrets.toml (for local) or Streamlit Cloud Secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("RCA and Sentiment Analysis App")
st.write("Analyze email content and generate insights")

# Email content input field
email_content = st.text_area("Enter email content:", height=200)

# Button to analyze email content
if st.button("Analyze"):
    try:
        # Load and configure the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate RCA and sentiment analysis
        response = model.generate_content(email_content)
        
        # Sentiment analysis using NLTK and TextBlob
        sentiment = SentimentIntensityAnalyzer().polarity_scores(email_content)
        sentiment_textblob = TextBlob(email_content).sentiment
        
        # Display results in Streamlit
        st.write("Root Cause Analysis:")
        st.write(response.text)
        st.write("Sentiment Analysis:")
        st.write(f"NLTK: {sentiment}")
        st.write(f"TextBlob: {sentiment_textblob}")
    except Exception as e:
        st.error(f"Error: {e}")
