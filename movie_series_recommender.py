# movie_series_recommender.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
import os
import streamlit as st

# === API Key ===
os.environ['GOOGLE_API_KEY'] = "AIzaSyBz584xVGKvl6bzl4eA7Lv0CgoGX9Oy8Wk"

# === LangChain Prompt Template ===
recommend_template = """
🎬 Recommend me a {type} based on this description: "{taste}". 
I prefer something {length} in duration. 
Return 3 existing {typename}s with names and a 1-line reason why.
"""

recommend_prompt = PromptTemplate(
    template=recommend_template.strip(),
    input_variables=["type", "taste", "length", "typename"]
)

# === Gemini Model ===
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
recommend_chain = recommend_prompt | gemini_model

# === Streamlit UI ===
st.set_page_config(page_title="🎥 Movie & Series Recommender", layout="centered")

with st.container():
    st.markdown("<h1 style='text-align: center;'>🎥 Movie & Series Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Find your next watch based on your taste and available time.</p>", unsafe_allow_html=True)
    st.markdown("---")

# === Form Layout ===
with st.form("recommend_form"):
    col1, col2 = st.columns(2)

    with col1:
        type_choice = st.selectbox("🔍 What are you looking for?", ["Movie", "Series"])

    with col2:
        length = st.selectbox("⏱️ Duration preference", ["Short", "Medium", "Long"])

    taste = st.text_area("📝 Describe your taste", placeholder="e.g. Mystery thriller with emotional depth")

    submitted = st.form_submit_button("🎯 Recommend")

# === Output Section ===
if submitted:
    if taste.strip() == "":
        st.warning("⚠️ Please describe your taste to get a recommendation.")
    else:
        with st.spinner("🎬 Finding the perfect match for you..."):
            result = recommend_chain.invoke({
                "type": type_choice,
                "taste": taste,
                "length": length,
                "typename": type_choice.lower()
            })

        st.markdown("---")
        st.subheader("📺 Your Recommendations")
        st.write(result.content)
