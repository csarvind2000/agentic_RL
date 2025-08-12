import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from rl_agents.q_learning_agent import QLearningAgent
from rl_agents.sarsa_agent import SarsaAgent
from rl_agents.dqn_agent import DQNAgent

# Setup
st.set_page_config(page_title="RL Reward Logger", layout="wide")
st.title("📊 RL Agent Reward Comparison per Model")

uploaded = st.file_uploader("Upload your CSV file", type="csv")

if uploaded:
    data = pd.read_csv(uploaded)
    actions = ["pca", "no_pca"]

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

    def get_agent(algorithm, actions):
        if algorithm == "Q-Learning":
            return QLearningAgent(actions)
        elif algorithm == "SARSA":
            return SarsaAgent(actions)
        elif algorithm == "DQN":
            return DQNAgent(actions)
        else:
            raise ValueError("Unknown RL algorithm")

    rl_reward_log = []
    all_rewards_summary = {}

    for algo in ["Q-Learning", "SARSA", "DQN"]:
        agent = get_agent(algo, actions)
        state = agent.get_state(data)
        action = agent.choose_action(state)

        X = data.drop("BodyFat", axis=1)
        y = data["BodyFat"]
        if action == "pca":
            X = PCA(n_components=min(5, X.shape[1])).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name, model in models.items():
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
            grid = GridSearchCV(pipeline, param_grid[model_name], cv=3)
            grid.fit(X_train, y_train)
            mse = mean_squared_error(y_test, grid.predict(X_test))
            reward = -mse

            rl_reward_log.append({
                "RL Algorithm": algo,
                "State": str(state),
                "Action": action,
                "Model": model_name,
                "MSE": round(mse, 3),
                "Reward": round(reward, 3)
            })

            # Track best reward for summary
            key = f"{algo}_{model_name}"
            all_rewards_summary[key] = reward

    reward_df = pd.DataFrame(rl_reward_log)

    st.subheader("🔍 RL Agent Log")
    st.dataframe(reward_df)

    st.subheader("📈 Reward Comparison Plot")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=reward_df, x="Model", y="Reward", hue="RL Algorithm")
    plt.title("RL Agent Reward vs. Model")
    st.pyplot(plt)

    st.markdown("""
    **Reward Calculation Note:**
    > Reward = − MSE. Lower MSE leads to higher reward. This formulation guides the RL agent to prefer models/actions that result in lower prediction error.
    """)

    # Final Best Result Summary
    st.markdown("## ✅ Final Summary")
    best_key = max(all_rewards_summary.items(), key=lambda x: x[1])[0]
    best_rl, best_model = best_key.split("_")
    best_reward = all_rewards_summary[best_key]

    st.success(f"🏆 Best RL Algorithm: `{best_rl}`")
    st.success(f"🧠 Best ML Model: `{best_model}`")
    st.success(f"🎯 Highest Reward: `{round(best_reward, 3)}` (i.e., lowest MSE)")

    st.markdown("""
    **Explanation:**
    The best RL agent was selected based on the maximum reward, which corresponds to the lowest MSE achieved across all models and RL strategies.
    This reflects the most optimal combination of model and preprocessing decision (PCA or no PCA) driven by the agent's policy.
    """)
