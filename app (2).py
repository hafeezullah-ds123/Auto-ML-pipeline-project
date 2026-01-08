import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, make_scorer

st.set_page_config(page_title="AutoML Data Prep", layout="wide")
st.title("🚀 AutoML Data Preparation & Best Model Trainer")

# -------------------- Upload CSV --------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Original Dataset")
    st.dataframe(df.head())

    # -------------------- Target Column --------------------
    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    if target_col:

        # -------------------- Problem Type --------------------
        if df[target_col].dtype in ["int64", "float64"] and df[target_col].nunique() > 10:
            problem_type = "Regression"
        else:
            problem_type = "Classification"

        st.success(f"Detected Problem Type: **{problem_type}**")

        # -------------------- Remove Duplicates --------------------
        before_dup = df.shape[0]
        df = df.drop_duplicates()
        after_dup = df.shape[0]
        st.info(f"Removed {before_dup - after_dup} duplicate rows")
        
        before_drop = df.shape[0]
        df = df.dropna()
        after_drop = df.shape[0]
        st.info(f"Removed {before_drop - after_drop} null value rows")

        # -------------------- Outlier Removal --------------------
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols.remove(target_col) if target_col in numeric_cols else None

        def remove_outliers_iqr(data, columns):
            for col in columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower) & (data[col] <= upper)]
            return data

        before_outliers = df.shape[0]
        df = remove_outliers_iqr(df, numeric_cols)
        after_outliers = df.shape[0]

        st.info(f"Removed {before_outliers - after_outliers} outlier rows")

        # -------------------- Split X & y --------------------
        X = df.drop(columns=[target_col])
        y = df[target_col]

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        st.write("🔢 Numeric Columns:", numeric_cols)
        st.write("🔤 Categorical Columns:", categorical_cols)

        # -------------------- Preprocessing --------------------
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

        # -------------------- Train/Test Split --------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------------------- Models --------------------
        if problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(random_state=42)
            }
            scoring = make_scorer(r2_score)
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest Classifier": RandomForestClassifier(random_state=42)
            }
            scoring = make_scorer(accuracy_score)

        # -------------------- Model Selection --------------------
        st.subheader("📊 Model Evaluation (Cross-Validation)")

        best_score = -np.inf
        best_pipeline = None
        best_model_name = None

        for name, model in models.items():
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring=scoring)
            mean_score = scores.mean()

            st.write(f"**{name}** → CV Score: `{mean_score:.4f}`")

            if mean_score > best_score:
                best_score = mean_score
                best_pipeline = pipe
                best_model_name = name

        # -------------------- Train Best Model --------------------
        st.success(f"🏆 Best Model: **{best_model_name}**")
        best_pipeline.fit(X_train, y_train)

        # -------------------- Download Model --------------------
        joblib.dump(best_pipeline, "best_model.pkl")

        st.download_button(
            label="⬇️ Download Trained Model (.pkl)",
            data=open("best_model.pkl", "rb"),
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )

        st.success("✅ Model trained & ready for deployment!")
