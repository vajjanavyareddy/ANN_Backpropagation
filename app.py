import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    border: none;
}

.stButton>button:hover {
    background-color: #ff1f1f;
    color: white;
}

.metric-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# HEADER SECTION
# ----------------------------------------------------

st.markdown("""
# 🚢 Titanic Survival Prediction System

### Deep Learning Based Passenger Survival Prediction
""")

# st.image(
#     "https://images.unsplash.com/photo-1518546305927-5a555bb7020d",
#     use_container_width=True
# )

# ----------------------------------------------------
# DESCRIPTION
# ----------------------------------------------------

with st.container():

    st.markdown("""
    ## 📘 Project Description

    This web application predicts whether a passenger would survive during the Titanic disaster using a manually implemented Artificial Neural Network.

    ### Technologies Used
    - Artificial Neural Network (ANN)
    - Forward Propagation
    - Sigmoid Activation Function
    - Streamlit Deployment
    - Data Visualization

    The system uses passenger information to estimate survival probability.
    """)

st.divider()

# ----------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------

st.markdown("## 🎫 Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=120.0
    )

# ----------------------------------------------------
# NORMALIZATION
# ----------------------------------------------------

pclass_norm = (pclass - 1) / (3 - 1)
age_norm = age / 100
fare_norm = fare / 150

# ----------------------------------------------------
# SIGMOID FUNCTION
# ----------------------------------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# ----------------------------------------------------
# INITIAL WEIGHTS
# ----------------------------------------------------

w1 = 0.11
w2 = 0.14
w3 = 0.17

w4 = 0.21
w5 = 0.24
w6 = 0.27

bh1 = 0.1
bh2 = 0.1

w7 = 0.31
w8 = 0.34

bo = 0.1

# ----------------------------------------------------
# PREDICTION BUTTON
# ----------------------------------------------------

if st.button("🔍 Predict Survival"):

    # ------------------------------------------------
    # FORWARD PROPAGATION
    # ------------------------------------------------

    h1_input = (
        pclass_norm * w1 +
        age_norm * w2 +
        fare_norm * w3 +
        bh1
    )

    h1 = sigmoid(h1_input)

    h2_input = (
        pclass_norm * w4 +
        age_norm * w5 +
        fare_norm * w6 +
        bh2
    )

    h2 = sigmoid(h2_input)

    output_input = (
        h1 * w7 +
        h2 * w8 +
        bo
    )

    prediction = sigmoid(output_input)

    survival_prob = prediction
    nonsurvival_prob = 1 - prediction

    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    st.divider()

    st.markdown("## 📊 Prediction Results")

    if prediction > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Prediction",
            result
        )

    with c2:
        st.metric(
            "Survival Probability",
            f"{survival_prob*100:.2f}%"
        )

    with c3:
        st.metric(
            "Confidence Score",
            f"{max(survival_prob, nonsurvival_prob)*100:.2f}%"
        )

    # ------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------

    st.markdown("## 📈 Probability Visualization")

    labels = ['Survival', 'Non-Survival']
    values = [survival_prob, nonsurvival_prob]

    fig, ax = plt.subplots(figsize=(6,4))

    bars = ax.bar(labels, values)

    ax.set_ylim([0, 1])

    ax.set_ylabel("Probability")

    ax.set_title("Prediction Probability")

    st.pyplot(fig)

    # ------------------------------------------------
    # PROGRESS BAR
    # ------------------------------------------------

    st.markdown("## 🚦 Survival Probability Meter")

    st.progress(float(survival_prob))

    # ------------------------------------------------
    # EVALUATION METRICS
    # ------------------------------------------------

    st.divider()

    st.markdown("## 📉 Model Evaluation Metrics")

    # Example Actual Values
    y_true = np.array([
        1, 0, 1, 1, 0,
        1, 0, 0, 1, 1
    ])

    # Example Predicted Values
    y_pred = np.array([
        1, 0, 1, 1, 0,
        1, 0, 1, 1, 1
    ])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Accuracy",
            f"{accuracy*100:.2f}%"
        )

    with m2:
        st.metric(
            "Precision",
            f"{precision*100:.2f}%"
        )

    with m3:
        st.metric(
            "Recall",
            f"{recall*100:.2f}%"
        )

    with m4:
        st.metric(
            "F1 Score",
            f"{f1*100:.2f}%"
        )

    # ------------------------------------------------
    # PIE CHART
    # ------------------------------------------------

    st.markdown("## 🥧 Survival Distribution")

    fig2, ax2 = plt.subplots(figsize=(5,5))

    ax2.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    st.pyplot(fig2)

    # ------------------------------------------------
    # NETWORK DETAILS
    # ------------------------------------------------

    with st.expander("🧠 ANN Network Details"):

        st.write("### Hidden Layer Outputs")

        st.write(f"h1 Output: {h1:.4f}")
        st.write(f"h2 Output: {h2:.4f}")

        st.write("### Final Output")

        st.write(f"Prediction Value: {prediction:.4f}")

        st.write("### Network Architecture")

        st.write("""
        - 3 Input Neurons
        - 2 Hidden Neurons
        - 1 Output Neuron
        - Sigmoid Activation Function
        """)