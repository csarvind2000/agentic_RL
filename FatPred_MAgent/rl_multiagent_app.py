import os
import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rl_agents.q_learning_agent import QLearningAgent
from rl_agents.sarsa_agent import SarsaAgent
from rl_agents.dqn_agent import DQNAgent
from agents.eda_agent import perform_eda
from agents.shap_agent import explain_with_shap
from agents.llm_agent import generate_llm_recommendations
from dotenv import load_dotenv

# Load OpenAI key
load_dotenv()

# Initialize session state if needed
def init_state():
    keys = [
        "agent", "q_table", "model_result", "X_train", "X_test", "y_train", "y_test",
        "data", "shap_done", "best_model_name", "all_rewards"
    ]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

# Agent selector
@st.cache_resource
def get_agent(algorithm, actions):
    if algorithm == "Q-Learning":
        return QLearningAgent(actions)
    elif algorithm == "SARSA":
        return SarsaAgent(actions)
    elif algorithm == "DQN":
        return DQNAgent(actions)
    else:
        raise ValueError("Unknown RL algorithm")

# Model trainer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(),
        "MLP": MLPRegressor(),
        "SVR": SVR()
    }

    param_grid = {
        "RandomForest": {"model__n_estimators": [100]},
        "MLP": {"model__hidden_layer_sizes": [(50,)], "model__alpha": [0.001]},
        "SVR": {"model__C": [1], "model__gamma": ["scale"]}
    }

    best_model = None
    best_score = float("inf")
    best_estimator = None
    best_model_name = ""

    for name, model in models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        grid = GridSearchCV(pipeline, param_grid[name], cv=3)
        grid.fit(X_train, y_train)
        mse = mean_squared_error(y_test, grid.predict(X_test))
        if mse < best_score:
            best_score = mse
            best_model = grid
            best_estimator = grid.best_estimator_
            best_model_name = name

    return best_estimator, {"mse": best_score}, X_train, X_test, y_train, y_test, best_model_name

# Main
init_state()
st.set_page_config(page_title="RL Multi-Agent Body Fat App", layout="wide")
st.title("🧠 RL-Driven Multi-Agent Body Fat Prediction & Recommendation")

# Sidebar: Select RL algorithm
algorithm = st.sidebar.selectbox("Select RL Algorithm", ["Q-Learning", "SARSA", "DQN"])

# Upload CSV
uploaded = st.file_uploader("Upload your CSV file", type="csv")
if uploaded:
    if st.session_state.data is None:
        st.session_state.data = pd.read_csv(uploaded)
    data = st.session_state.data

    actions = ["pca", "no_pca"]
    rl_agent = get_agent(algorithm, actions)

    state = rl_agent.get_state(data)
    action = rl_agent.choose_action(state)
    st.session_state.agent = rl_agent

    rl_col, right_col = st.columns([1, 2])

    with rl_col:
        st.subheader(f"🤖 RL Decision ({algorithm})")
        st.info(f"Selected Action: **{action}**")

        st.markdown("### 🧠 RL Agent Stats")
        st.markdown(f"**Last State:** `{rl_agent.last_state}`")
        st.markdown(f"**Last Action:** `{rl_agent.last_action}`")
        st.markdown(f"**Last Reward:** `{rl_agent.last_reward}`")

        q_df = pd.DataFrame.from_dict(rl_agent.q_table, orient="index").fillna(0)
        q_df.index = [f"State {i+1}" for i in range(len(q_df))]
        st.dataframe(q_df)

    with right_col:
        ml_col, llm_col = st.columns([2, 1])

        with ml_col:
            st.subheader("📄 Uploaded Data")
            st.dataframe(data.head())

            perform_eda(data)

            X = data.drop("BodyFat", axis=1)
            y = data["BodyFat"]
            if action == "pca":
                X = PCA(n_components=min(5, X.shape[1])).fit_transform(X)
                st.warning("PCA was applied. Top feature importance not available.")

            if st.session_state.model_result is None:
                best_model, metrics, X_train, X_test, y_train, y_test, best_model_name = train_models(X, y)
                st.session_state.model_result = best_model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.best_model_name = best_model_name

                reward = -metrics["mse"]
                rl_agent.update_q_table(state, action, reward, state)
                st.session_state.all_rewards = st.session_state.all_rewards or {}
                st.session_state.all_rewards[algorithm] = reward

            if not st.session_state.shap_done:
                explain_with_shap(
                    st.session_state.model_result,
                    st.session_state.X_train,
                    st.session_state.X_test,
                    pca_applied=(action == "pca")
                )
                st.session_state.shap_done = True

            st.markdown(f"### 🧠 Best ML Model: `{st.session_state.best_model_name}`")
            st.markdown("Model was selected based on lowest MSE after PCA/no-PCA action decided by RL agent.")

            if action == "pca":
                st.info("RL chose PCA likely due to high-dimensional features, benefiting from dimensionality reduction.")
            else:
                st.info("RL chose no PCA likely because raw features were more predictive without transformation.")

            # Reward comparison plot
            if st.session_state.all_rewards:
                st.markdown("### 📊 RL Algorithm Reward Comparison")
                reward_df = pd.DataFrame({"Algorithm": list(st.session_state.all_rewards.keys()),
                                          "Reward": list(st.session_state.all_rewards.values())})
                fig, ax = plt.subplots()
                ax.bar(reward_df["Algorithm"], reward_df["Reward"], color="skyblue")
                ax.set_ylabel("Reward (-MSE)")
                ax.set_title("RL Algorithm Reward Comparison")
                st.pyplot(fig)

        with llm_col:
            st.markdown("### 💬 LLM-Based Recommendations")
            subject_index = st.selectbox("Select a subject", range(len(data)), index=0)
            selected_subject = data.iloc[subject_index:subject_index+1]
            generate_llm_recommendations(selected_subject)

    if st.session_state.all_rewards:
        best_rl_method = max(st.session_state.all_rewards.items(), key=lambda x: x[1])[0]
        st.success(f"✅ **Best RL Algorithm**: `{best_rl_method}` with highest reward.")
