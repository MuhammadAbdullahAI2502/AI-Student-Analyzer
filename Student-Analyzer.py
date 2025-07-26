import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="AI Student Analyzer", layout="wide")

st.title("🧠 AI Student Analyzer")
st.write("Analyze student performance, detect risks, and find learning patterns.")

# File uploader
uploaded_file = st.file_uploader("Upload Student CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Trend Summary")
    st.write(df.describe())

    # Target column selector
    target = st.selectbox("🎯 Select Target Column (e.g., pass/fail or grade)", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Handle non-numeric data
        X = pd.get_dummies(X)

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        st.subheader("✅ Model Results")
        st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Risk analysis
        st.subheader("⚠️ High-Risk Student Detection")
        risk_df = df.copy()
        risk_df["Prediction"] = model.predict(X)

        st.write(risk_df[risk_df["Prediction"] == y.value_counts().idxmin()])
