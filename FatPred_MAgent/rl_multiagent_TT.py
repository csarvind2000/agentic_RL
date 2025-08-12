import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from scipy.stats import linregress
from rl_agents.q_learning_agent import QLearningAgent
from rl_agents.sarsa_agent import SarsaAgent
from rl_agents.dqn_agent import DQNAgent

# Load environment
load_dotenv()
client = OpenAI()

# Cache remembered target columns
TARGET_COLUMN_CACHE_FILE = "target_column_cache.csv"

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
            "SVC": SVC(probability=True)
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

def generate_llm_recommendations(data):
    st.subheader("📝 LLM-Based Personalized Recommendation of the selected subject")
    selected_index = st.selectbox("Select a Subject Row for Recommendation", data.index)
    selected_row = data.loc[[selected_index]]
    selected_row_display = selected_row.to_string(index=False, header=True)
    
    st.markdown(f"**Selected Subject Data:**\n\n```\n{selected_row_display}\n```")
    
    selected_row_dict = selected_row.to_dict(orient="records")[0]
    prompt = (
        f"You are a fitness and nutrition expert.\n"
        f"Given the following body metrics:\n"
        f"{selected_row_dict}\n"
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


# Streamlit UI setup
st.set_page_config(page_title="RL Task-Aware Agentic Trainer", layout="wide")
st.title("🤖 AutoRL-Medic: A RL-Guided Agentic System for Body Composition Modeling")


col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
        st.write("Data Preview:", data.head())

        data_hash = str(hash(tuple(data.columns))) + str(data.shape)
        remembered_target = load_cached_target(data_hash)
        suggested_targets = [col for col in data.columns if 'class' in col.lower() or 'label' in col.lower() or 'target' in col.lower() or 'BComp19_FM' in col]
        default_index = data.columns.get_loc(remembered_target) if remembered_target and remembered_target in data.columns else (data.columns.get_loc(suggested_targets[0]) if suggested_targets else 0)

        target_col = st.selectbox("Select Target Column", data.columns, index=default_index)
        save_target_cache(data_hash, target_col)

        task_type = detect_task_type(data, target_col)
        st.info(f"Detected Task Type: {task_type.upper()}")

        models, param_grid = get_models_and_params(task_type)
        actions = ["pca", "no_pca"]
        rl_rewards = []

        for algo in ["Q-Learning", "SARSA", "DQN"]:
            agent = get_agent(algo, actions)
            state = agent.get_state(data)
            action = agent.choose_action(state)

            X = data.drop(columns=[target_col])
            y = data[target_col]
            X = pd.get_dummies(X, drop_first=True)
            X = SimpleImputer(strategy="mean").fit_transform(X)

            if action == "pca":
                X = PCA(n_components=min(5, X.shape[1])).fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            for model_name, model in models.items():
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
                grid = GridSearchCV(pipeline, param_grid[model_name], cv=3)
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
                    "Model": model_name,
                    "Action": action,
                    "Task": task_type,
                    "Metric": round(metric, 4),
                    "Reward": round(reward, 4)
                })

        reward_df = pd.DataFrame(rl_rewards)
        st.subheader("🧠 RL Decision Log")
        st.dataframe(reward_df)

        st.subheader("📊 Reward Comparison")
        fig, ax = plt.subplots()
        sns.barplot(data=reward_df, x="Model", y="Reward", hue="RL Algorithm", ax=ax)
        st.pyplot(fig)

        best_row = reward_df.loc[reward_df["Reward"].idxmax()]
        st.success(f"Best RL: {best_row['RL Algorithm']}, Model: {best_row['Model']} with Reward: {best_row['Reward']}")

with col2:
    if uploaded:
        st.subheader("🚀 Inference")
        if st.button("Run Inference"):
            st.info("Running evaluation...")
            X = data.drop(columns=[target_col])
            y = data[target_col]
            X = pd.get_dummies(X, drop_first=True)
            X = SimpleImputer(strategy="mean").fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model_cls = get_models_and_params(task_type)[0][best_row['Model']]
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model_cls)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            st.write(f"Total Data: {len(data)}, Test Samples: {len(y_test)}")
            st.write("Top 5 Test Samples:")
            st.dataframe(pd.DataFrame(X_test[:5], columns=data.drop(columns=[target_col]).columns))
            st.info(f"📌 Using Model: {best_row['Model']} via {best_row['RL Algorithm']}")

            if task_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
                st.write(f"MSE: {mse:.4f}, P-value: {p_value:.4f}")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_title("Regression Plot")
                ax.set_xlabel("True")
                ax.set_ylabel("Pred")
                st.pyplot(fig)
            else:
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                cm = confusion_matrix(y_test, y_pred)
                st.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
                st.pyplot(fig)

            try:
                explainer = shap.Explainer(pipeline.named_steps['model'], X_train)
                shap_values = explainer(X_test)
                st.subheader("🔹 SHAP Value Summary")
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(bbox_inches='tight')
            except Exception as e:
                st.warning(f"SHAP error: {e}")

            # st.subheader("🔢 Personalized Subject Report")
            # subject_index = st.number_input("Enter subject index to generate personalized report (0 to N):", min_value=0, max_value=len(data)-1, step=1)
            # st.dataframe(data.iloc[subject_index].to_frame().T)
 
            generate_llm_recommendations(data)


