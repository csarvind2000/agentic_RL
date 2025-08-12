import streamlit as st
import shap
import matplotlib.pyplot as plt

def explain_with_shap(model, X_train, X_test, feature_names=None, pca_applied=False):
    st.subheader("🧠 SHAP Feature Importance")

    if pca_applied:
        st.info("Feature importance not available due to PCA.")
        return

    try:
        explainer = shap.Explainer(model.named_steps["model"], X_train)
        shap_values = explainer(X_test)

        fig = plt.figure()
        shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, show=False)
        st.pyplot(fig)

        top_features = shap_values.abs.mean(0)
        top_names = sorted(zip(top_features, feature_names), reverse=True)[:5]
        st.markdown("### ⭐ Top 5 Important Features")
        for i, (val, name) in enumerate(top_names, 1):
            st.write(f"{i}. {name} (mean SHAP: {val:.3f})")
    except Exception as e:
        st.warning("SHAP analysis failed.")
        st.error(str(e))
