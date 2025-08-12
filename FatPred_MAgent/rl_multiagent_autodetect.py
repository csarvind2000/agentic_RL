import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score
from rl_agents.q_learning_agent import QLearningAgent
from rl_agents.sarsa_agent import SarsaAgent
from rl_agents.dqn_agent import DQNAgent

# Cache remembered target columns
TARGET_COLUMN_CACHE_FILE = "target_column_cache.csv"

# Load remembered columns
def load_cached_target(data_hash):
    if os.path.exists(TARGET_COLUMN_CACHE_FILE):
        df = pd.read_csv(TARGET_COLUMN_CACHE_FILE)
        match = df[df['data_hash'] == data_hash]
        if not match.empty:
            return match.iloc[0]['target_column']
    return None

def save_target_cache(data_hash, target_column):
    if os.path.exists(TARGET_COLUMN_CACHE_FILE):
        df = pd.read_csv(TARGET_COLUMN_CACHE_FILE)
        df = df[df['data_hash'] != data_hash]
    else:
        df = pd.DataFrame(columns=['data_hash', 'target_column'])
    df = pd.concat([df, pd.DataFrame([[data_hash, target_column]], columns=['data_hash', 'target_column'])], ignore_index=True)
    df.to_csv(TARGET_COLUMN_CACHE_FILE, index=False)

# Detect task type
def detect_task_type(data: pd.DataFrame, target_col: str):
    target = data[target_col]
    if target.dtype == object or target.dtype.name == "category" or target.nunique() <= 10:
        return "classification"
    return "regression"

def get_models_and_params(task_type):
    if task_type == "regression":
        return {
            "RandomForest": RandomForestRegressor(),
            "MLP": MLPRegressor(),
            "SVR": SVR()
        }, {
            "RandomForest": {"model__n_estimators": [100]},
            "MLP": {"model__hidden_layer_sizes": [(50,)], "model__alpha": [0.001]},
            "SVR": {"model__C": [1], "model__gamma": ["scale"]}
        }
    else:
        return {
            "RandomForest": RandomForestClassifier(),
            "MLP": MLPClassifier(),
            "SVC": SVC()
        }, {
            "RandomForest": {"model__n_estimators": [100]},
            "MLP": {"model__hidden_layer_sizes": [(50,)], "model__alpha": [0.001]},
            "SVC": {"model__C": [1], "model__gamma": ["scale"]}
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

# Streamlit setup
st.set_page_config(page_title="RL Task-Aware Agentic Trainer", layout="wide")
st.title("🤖 RL-Driven Smart Trainer: Task-Adaptive Modeling")

uploaded = st.file_uploader("Upload your CSV file", type="csv")
if uploaded:
    data = pd.read_csv(uploaded)
    st.write("Uploaded Data:", data.head())

    data_hash = str(hash(tuple(data.columns))) + str(data.shape)
    remembered_target = load_cached_target(data_hash)

    suggested_targets = [col for col in data.columns if 'class' in col.lower() or 'label' in col.lower() or 'target' in col.lower() or 'BComp19_FM' in col]
    default_index = data.columns.get_loc(remembered_target) if remembered_target and remembered_target in data.columns else 0

    target_col = st.selectbox("Select Target Column", data.columns, index=default_index)
    save_target_cache(data_hash, target_col)

    task_type = detect_task_type(data, target_col)
    st.info(f"**Detected Task Type:** {task_type.upper()}")

    models, param_grid = get_models_and_params(task_type)
    actions = ["pca", "no_pca"]

    rl_rewards = []

    for algo in ["Q-Learning", "SARSA", "DQN"]:
        agent = get_agent(algo, actions)
        state = agent.get_state(data)
        action = agent.choose_action(state)

        X = data.drop(target_col, axis=1)
        y = data[target_col]

        X = pd.get_dummies(X, drop_first=True)

        # Impute missing values
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        if action == "pca":
            X = PCA(n_components=min(5, X.shape[1])).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for name, model in models.items():
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
            grid = GridSearchCV(pipeline, param_grid[name], cv=3)
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)

            if task_type == "regression":
                metric = mean_squared_error(y_test, y_pred)
                reward = -metric
            else:
                metric = accuracy_score(y_test, y_pred)
                reward = metric

            rl_rewards.append({
                "RL Algorithm": algo,
                "Model": name,
                "Action": action,
                "Task": task_type,
                "Metric": round(metric, 4),
                "Reward": round(reward, 4)
            })

    reward_df = pd.DataFrame(rl_rewards)
    st.subheader("🧠 RL Decision Log")
    st.dataframe(reward_df)

    st.subheader("📊 Reward Comparison")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=reward_df, x="Model", y="Reward", hue="RL Algorithm")
    plt.title("RL Reward vs Model")
    plt.tight_layout()
    st.pyplot(plt)

    best_row = reward_df.loc[reward_df["Reward"].idxmax()]
    st.success(f"✅ Best RL Algorithm: `{best_row['RL Algorithm']}`, Model: `{best_row['Model']}` for `{best_row['Task']}` task with Reward: {best_row['Reward']}")

    st.markdown("""
    **Reward Calculation**  
    - **Regression**: Reward = - MSE (Lower MSE → Higher Reward)  
    - **Classification**: Reward = Accuracy (Higher Accuracy → Higher Reward)
    """)
