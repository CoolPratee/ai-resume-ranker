import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from huggingface_hub import InferenceClient
import re

# ---------------- LOAD MODELS ---------------- #

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    hf_client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token="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"   # 🔥 PUT YOUR TOKEN HERE
    )

    return embed_model, hf_client

embed_model, hf_client = load_models()

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c92d2);
    color: white;
}
textarea {
    background-color: white !important;
    color: black !important;
}
div[data-testid="stFileUploader"] button {
    background: #0072ff !important;
    color: white !important;
}
.stButton > button, .stDownloadButton > button {
    background: #0072ff;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ---------------- #

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join([p.extract_text() or "" for p in reader.pages])
    else:
        doc = Document(file)
        return " ".join([p.text for p in doc.paragraphs])


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def calculate_score(jd, resume):
    jd_clean = clean_text(jd)
    res_clean = clean_text(resume)

    jd_emb = embed_model.encode(jd_clean, convert_to_tensor=True)
    res_emb = embed_model.encode(res_clean, convert_to_tensor=True)

    semantic_score = util.cos_sim(jd_emb, res_emb).item()

    jd_words = set(jd_clean.split()) - set(ENGLISH_STOP_WORDS)
    res_words = set(res_clean.split()) - set(ENGLISH_STOP_WORDS)

    keyword_score = len(jd_words.intersection(res_words)) / max(len(jd_words), 1)

    return (semantic_score * 0.7 + keyword_score * 0.3) * 100


# ---------------- REAL AI FEEDBACK ---------------- #

def generate_feedback(jd, resume):

    prompt = f"""
You are a senior recruiter.

Carefully analyze the job description and resume.

Job Description:
{jd}

Resume:
{resume}

Provide structured feedback:

1. Positives:
- What matches well with the role

2. Improvements:
- What is missing or weak

Be specific, professional, and human-like.
"""

    try:
        response = hf_client.text_generation(
            prompt,
            max_new_tokens=400,
            temperature=0.7
        )
        return response

    except Exception as e:
        return f"Error generating feedback: {str(e)}"


# ---------------- HEADER ---------------- #

st.title("🚀 AI Resume Ranker")
st.write("Smart ATS with real AI feedback (Mistral LLM)")

# ---------------- INPUT ---------------- #

job_desc = st.text_area("📄 Enter Job Description")

uploaded_files = st.file_uploader(
    "📂 Upload Resumes (PDF / Word)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# ---------------- PROCESS ---------------- #

if st.button("🚀 Rank Resumes"):

    if job_desc and uploaded_files:

        results = []

        for file in uploaded_files:
            raw_text = extract_text(file)

            score = calculate_score(job_desc, raw_text)
            feedback = generate_feedback(job_desc, raw_text)

            results.append({
                "Resume": file.name,
                "ATS Score (%)": round(score, 2),
                "AI Feedback": feedback
            })

        df = pd.DataFrame(results).sort_values(by="ATS Score (%)", ascending=False)

        # ---------------- OUTPUT ---------------- #

        st.subheader("📊 Results Overview")
        st.dataframe(df)

        top = df.iloc[0]
        st.success(f"🏆 Top Candidate: {top['Resume']} ({top['ATS Score (%)']}%)")

        fig = px.bar(df, x="Resume", y="ATS Score (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 🧠 Detailed AI Feedback")

        for _, row in df.iterrows():
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.1);
                        padding:15px;
                        border-radius:10px;
                        margin-bottom:10px;">
            <h4>{row['Resume']} ({row['ATS Score (%)']}%)</h4>
            <p>{row['AI Feedback']}</p>
            </div>
            """, unsafe_allow_html=True)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "results.csv")

    else:
        st.warning("⚠️ Please enter job description and upload resumes.")