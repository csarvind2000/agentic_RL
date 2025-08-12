import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def perform_eda(data):
    st.subheader("📊 Exploratory Data Analysis")

    st.markdown("**Descriptive Statistics**")
    st.write(data.describe().T)

    st.markdown("**Feature Correlation Heatmap**")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
