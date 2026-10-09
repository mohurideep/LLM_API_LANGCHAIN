import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.title("🍽️ Restaurant Name Generator")

API_URL = "http://127.0.0.1:8000/generate"
APP_API_KEY = os.getenv("APP_API_KEY")

cuisine = st.sidebar.selectbox(
    "Pick a cuisine",
    ("Indian", "Bengali", "Italian", "Mexican", "Arabic", "American"),
)

if st.sidebar.button("Generate"):
    try:
        with st.spinner("Thinking of a fancy name..."):
            response = requests.post(
                API_URL,
                json={"cuisine": cuisine},
                headers={"x-api-key": APP_API_KEY},
                timeout=120,
            )
    except requests.exceptions.RequestException:
         st.error("Could not reach the restaurant API. Is the FastAPI server running?")
         st.stop()

    if response.status_code == 200:
        data = response.json()
        st.header(data["restaurant_name"].strip())
        st.write(data["slogan"].strip())

        st.subheader("Menu Items")
        for item in data["menu_items"].split(","):
            st.write("-", item.strip())
        st.caption(f"Credits left: {data['credit_left']}")
    elif response.status_code == 429:
                    st.error("Rate limit exceeded! Please try again later.")
    elif response.status_code == 402:
                    st.error("Payment required! Please check your API key and billing.")
    elif response.status_code == 401:
                    st.error("Unauthorized! Please check your API key.")
    elif response.status_code == 503:
                    st.error("Service unavailable! The AI service may be busy, please try again later.")
    else:
        st.error(f"Unexpected error ({response.status_code}): {response.text}")