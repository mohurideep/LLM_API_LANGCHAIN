import langchain_helper
import streamlit as st

st.title("🍽️ Restaurant Name Generator")
cuisine = st.sidebar.selectbox(
    "Pick a cuisine",
    ("Indian", "Bengali", "Italian", "Mexican", "Arabic", "American"),
)

if st.sidebar.button("Generate"):
    try:
        with st.spinner("Thinking of a fancy name..."):
            result = langchain_helper.generate_restaurant_name_and_items(cuisine)
    except RuntimeError as e:
         st.error(f"{e}. The AI service may be busy, please try again in a minute.")
         st.stop()

    st.header(result["restaurant_name"].strip())
    st.write(result["slogan"].strip())

    st.subheader("Menu Items")
    for item in result["menu_items"].split(","):
            st.write("-", item.strip())