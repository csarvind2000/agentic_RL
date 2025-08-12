import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load .env file relative to project root
load_dotenv()  # Make sure this is called here too
client = OpenAI()  # No need to pass api_key if OPENAI_API_KEY is in .env

def generate_llm_recommendations(data):
    st.subheader("📝 LLM-Based Personalized Recommendation")
    selected_index = st.selectbox("Select a Subject Row for Recommendation", data.index)
    selected_row = data.loc[selected_index].to_dict()

    prompt = (
        f"You are a fitness and nutrition expert.\n"
        f"Given the following body metrics:\n"
        f"{selected_row}\n"
        f"Suggest improvements in diet, workout, and lifestyle for fat reduction."
    )

    with st.spinner("Generating recommendations..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a health coach."},
                    {"role": "user", "content": prompt}
                ]
            )
            message = response.choices[0].message.content
            st.markdown("### 📋 Recommendation:")
            st.write(message)
        except Exception as e:
            st.error(f"LLM recommendation failed: {str(e)}")
