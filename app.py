import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import joblib

# 1. Application configuration
st.set_page_config(
    page_title="CivicSense AI",
    page_icon="🏙️",
    layout="wide"
)

DB_FILE = "civicsense.db"

@st.cache_resource
def load_ml_components():
    model = joblib.load("complaint_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_ml_components()

def predict_category(description):
    complaint_tfidf = vectorizer.transform([description])
    prediction = model.predict(complaint_tfidf)
    return prediction[0]

# 2. Initialize the database
def initialize_database():

    conn = sqlite3.connect(DB_FILE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            category TEXT NOT NULL,
            location TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending'
        )
    """)

    conn.commit()
    conn.close()


# 3. Save a complaint
def save_complaint(category, location, severity, description):

    conn = sqlite3.connect(DB_FILE)

    conn.execute("""
        INSERT INTO complaints
        (created_at, category, location, severity, description)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        category,
        location,
        severity,
        description
    ))

    conn.commit()
    conn.close()


# 4. Retrieve complaints
def get_complaints():

    conn = sqlite3.connect(DB_FILE)

    df = pd.read_sql_query("""
        SELECT * FROM complaints
        ORDER BY id DESC
    """, conn)

    conn.close()

    return df


# 5. Initialize application
initialize_database()

st.title("🏙️ CivicSense AI")
st.subheader("Intelligent Civic Issue Reporting System")

st.write(
    "Report civic problems in your area "
    "and help build better communities."
)

st.divider()


# 6. Complaint submission form
st.header("Submit a Complaint")

with st.form("complaint_form"):
    location = st.text_input(
        "Enter complaint location",
        placeholder="Example: Garia, Kolkata"
    )

    severity = st.selectbox(
        "Select severity",
        ["Low", "Medium", "High"]
    )

    description = st.text_area(
        "Describe the problem",
        placeholder="Explain the civic issue..."
    )

    submitted = st.form_submit_button(
        "Submit Complaint",
        use_container_width=True
    )

    if submitted:
        if not location.strip() or not description.strip():
            st.error("Please enter location and complaint description.")
        else:
            predicted_category = predict_category(description)

            save_complaint(
                predicted_category,
                location,
                severity,
                description
            )

            st.success("Complaint submitted successfully!")

            st.info(f"AI Predicted Category: {predicted_category}")

# 7. Administrator dashboard
st.divider()

st.header("Administrator Dashboard")

complaints = get_complaints()

total = len(complaints)

pending = (
    len(complaints[complaints["status"] == "Pending"])
    if total else 0
)

high_priority = (
    len(complaints[complaints["severity"] == "High"])
    if total else 0
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Complaints", total)
col2.metric("Pending Complaints", pending)
col3.metric("High Severity", high_priority)


# 8. Display complaint records
st.subheader("Recent Complaints")

if not complaints.empty:

    st.dataframe(
        complaints,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No complaints submitted yet.")
