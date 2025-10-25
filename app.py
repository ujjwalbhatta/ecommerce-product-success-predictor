import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Product Success Predictor", layout="wide")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    try:
        model_dir = "models"
        reg_model = joblib.load(os.path.join(model_dir, "best_regression_model.pkl"))
        clf_model = joblib.load(os.path.join(model_dir, "best_classification_model.pkl"))

        feature_names = [
            'log_price', 'log_reviewCount',
            'brand_tier_encoded','brand_tier_risk_score','is_rare_brand',
            'category_risk_score','category_size_encoded',
            'is_risky_category','is_major_category',
            'sentence_count','sentiment_polarity','sentiment_subjectivity',
            'positive_words','negative_words','quality_ratio'
        ]
        metadata = {
            "regression": {"model_name": "LightGBM", "r2": 0.92},
            "classification": {"model_name": "XGBoost", "accuracy": 0.91}
        }
        return reg_model, clf_model, feature_names, metadata
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.stop()

reg_model, clf_model, feature_names, metadata = load_models()

# =========================
# FEATURE PREPARATION
# =========================
def prepare_features(user_inputs, feature_names):
    features = pd.DataFrame([user_inputs])
    # Remove extra columns
    for col in list(features.columns):
        if col not in feature_names:
            features.drop(columns=col, inplace=True)
    # Fill missing features with defaults
    for fname in feature_names:
        if fname not in features.columns: features[fname] = 0
    return features[feature_names]

# =========================
# MAIN UI LAYOUT
# =========================
st.title("🛒 Quick Product Success Checker")
st.markdown(
    "Estimate this product's **expected rating and risk level** using category and brand intelligence."
)

st.sidebar.header("Product Details")

# Price & Popularity
price = st.sidebar.number_input("Price ($)", min_value=1.0, max_value=500.0, value=25.0, help="Product sale price")
review_count = st.sidebar.number_input("Expected Review Count", min_value=0, max_value=5000, value=10, help="How many reviews would you expect for this product?")

# Brand details
brand_type = st.sidebar.selectbox("Brand Type", ["Emerging", "Established", "Premium", "Rare"], help="Reputation and reach of the brand")
brand_trust = st.sidebar.slider("Brand Trust Score", 4.23, 4.45, 4.34, help="Higher = more trusted or premium")
is_rare_brand = st.sidebar.radio("Is this brand rare?", ["No", "Yes"], help="Is this an uncommon or niche brand?")

# Categorical mappings (from your label encoders)
brand_tier_encoded = {"Emerging":0, "Established":1, "Premium":2, "Rare":3}[brand_type]
is_rare_brand_val = 0 if is_rare_brand == "No" else 1

# Categories
category_main = st.sidebar.selectbox("Main Category", ["All Electronics", "Computers", "Home Audio Theater", "Camera Photo", "Office Products", "Cell Phones Accessories", "Amazon Fashion", "Other"], help="General category type")
category_tier = st.sidebar.selectbox("Category Tier", ["Reliable", "Premium", "Moderate", "Risky"], help="Category risk tier (Premium: Apple/Amazon, Reliable: top brands, Moderate/Risky: smaller/niche)")
category_size = st.sidebar.selectbox("Category Size", ["Major", "Large", "Medium", "Niche"], help="How common is this category on the marketplace?")
category_risk_score = st.sidebar.slider("Category Risk Score", 4.16, 4.66, 4.25, help="Higher = lower risk (benchmark avg by category in data)")
is_major_category = 1 if category_size == "Major" else 0
is_risky_category_val = 1 if category_tier == "Risky" else 0

# Category encodings (as per notebook)
category_tier_encoded = {"Moderate":0, "Premium":1, "Reliable":2, "Risky":3}[category_tier]
category_size_encoded = {"Large":0, "Major":1, "Medium":2, "Niche":3}[category_size]
category_encoded = {
    "All Electronics":0, "Computers":1, "Home Audio Theater":2, "Camera Photo":3, "Office Products":4,
    "Cell Phones Accessories":5, "Amazon Fashion":6, "Other":7
}.get(category_main, 7)

# Advanced text/quality options
show_advanced = st.sidebar.checkbox("Show advanced text features")
if show_advanced:
    sentence_count = st.sidebar.number_input("Sentence Count", min_value=1, max_value=20, value=5, help="Number of sentences typically in product description")
    sentiment_polarity = st.sidebar.slider("Sentiment Polarity", 0.0, 1.0, 0.3, help="Automated score, 1 = highly positive")
    sentiment_subjectivity = st.sidebar.slider("Sentiment Subjectivity", 0.0, 1.0, 0.5)
    positive_words = st.sidebar.number_input("Positive word count", min_value=0, max_value=20, value=8)
    negative_words = st.sidebar.number_input("Negative word count", min_value=0, max_value=20, value=2)
    quality_ratio = st.sidebar.slider("Quality Ratio", 0.0, 1.0, 0.8)
else:
    # Use dataset averages [file:1]
    sentence_count, sentiment_polarity, sentiment_subjectivity = 5, 0.288, 0.527
    positive_words, negative_words, quality_ratio = 0.54, 0.19, 0.8

# Brand encoding for full set [file:1]
brand_encoded = 0  # Default; real deployment should map from brand dictionary

inputs = {
    'log_price': np.log(price + 1),
    'log_reviewCount': np.log(review_count + 1),
    'brand_encoded': brand_encoded,
    'brand_tier_encoded': brand_tier_encoded,
    'brand_tier_risk_score': brand_trust,
    'is_rare_brand': is_rare_brand_val,
    'category_encoded': category_encoded,
    'category_risk_score': category_risk_score,
    'category_tier_encoded': category_tier_encoded,
    'category_size_encoded': category_size_encoded,
    'is_risky_category': is_risky_category_val,
    'is_major_category': is_major_category,
    'sentence_count': sentence_count,
    'sentiment_polarity': sentiment_polarity,
    'sentiment_subjectivity': sentiment_subjectivity,
    'positive_words': positive_words,
    'negative_words': negative_words,
    'quality_ratio': quality_ratio
}
features = prepare_features(inputs, feature_names)

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict Performance 🚀"):
    try:
        predicted_rating = float(reg_model.predict(features)[0])
        risk_label = clf_model.predict(features)[0]
        risk_prob = clf_model.predict_proba(features)[0]

        st.subheader("🧠 Prediction Results")
        st.metric("Expected Rating", f"{predicted_rating:.2f} / 5")
        st.metric("Risk Category", risk_label)
        st.progress(float(max(risk_prob)))
        st.caption(f"Confidence: " + " | ".join([f"{lbl}: {prob:.2f}" for lbl, prob in zip(clf_model.classes_, risk_prob)]))
        if risk_label.lower() == "high risk" or predicted_rating < 3.0:
            st.warning("This product might underperform. Try adjusting price, description, or category.")
        else:
            st.success("Product looks promising for launch!")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# =========================
# FOOTER
# =========================
st.divider()
st.caption(f"Model: {metadata['regression']['model_name']} (R²={metadata['regression']['r2']}) | "
           f"Classifier: {metadata['classification']['model_name']} (Accuracy={metadata['classification']['accuracy']})")
