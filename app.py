import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# ---------------- LOAD MODEL ---------------- #
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c92d2);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 15px;
    padding: 25px;
    margin-top: 25px;
}
textarea {
    background-color: white !important;
    color: black !important;
}
div[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
    color: white !important;
}
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
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
    text = re.sub(r'[^a-zA-Z0-9\s\.]', '', text)
    return text


# ---------------- SMART FEEDBACK ---------------- #

def generate_feedback(jd, resume):

    jd = clean_text(jd)
    resume = clean_text(resume)

    jd_sentences = [s.strip() for s in jd.split(".") if len(s.strip()) > 20]
    res_sentences = [s.strip() for s in resume.split(".") if len(s.strip()) > 20]

    positives = []
    improvements = []

    for jd_line in jd_sentences:

        jd_emb = model.encode(jd_line, convert_to_tensor=True)

        best_score = 0

        for res_line in res_sentences:
            res_emb = model.encode(res_line, convert_to_tensor=True)
            score = util.cos_sim(jd_emb, res_emb).item()

            if score > best_score:
                best_score = score

        if best_score > 0.65:
            positives.append(f"Strong alignment found for requirement: '{jd_line[:80]}...'")
        else:
            improvements.append(f"The resume does not clearly address: '{jd_line[:80]}...'")

    # Additional intelligent checks
    if "project" in jd and "project" not in resume:
        improvements.append("Relevant projects are not clearly highlighted in the resume.")

    if "experience" in jd and "experience" not in resume:
        improvements.append("Professional experience is not clearly described.")

    if "skills" in jd and "skills" not in resume:
        improvements.append("A structured skills section is missing or not clearly visible.")

    # Limit output
    positives = positives[:4]
    improvements = improvements[:4]

    return " ".join(positives), " ".join(improvements)


# ---------------- SCORE ---------------- #

def calculate_score(jd, resume):

    jd_clean = clean_text(jd)
    res_clean = clean_text(resume)

    jd_emb = model.encode(jd_clean, convert_to_tensor=True)
    res_emb = model.encode(res_clean, convert_to_tensor=True)

    semantic_score = util.cos_sim(jd_emb, res_emb).item()

    jd_words = set(jd_clean.split()) - set(ENGLISH_STOP_WORDS)
    res_words = set(res_clean.split()) - set(ENGLISH_STOP_WORDS)

    keyword_score = len(jd_words.intersection(res_words)) / max(len(jd_words), 1)

    final_score = (semantic_score * 0.7 + keyword_score * 0.3) * 100

    return final_score


# ---------------- HEADER ---------------- #

st.markdown("""
<h1 style='text-align:center;'>📄 AI Resume Ranker</h1>
<p style='text-align:center;'>🚀 Smart AI ATS Analysis</p>
""", unsafe_allow_html=True)

# ---------------- UI ---------------- #

st.markdown('<div class="glass">', unsafe_allow_html=True)

job_desc = st.text_area("📝 Enter Job Description")

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

            positives, improvements = generate_feedback(job_desc, raw_text)

            results.append({
                "Resume": file.name,
                "ATS Score (%)": round(score, 2),
                "Positives": positives,
                "Improvements": improvements
            })

        df = pd.DataFrame(results).sort_values(by="ATS Score (%)", ascending=False)

        st.subheader("📊 Results Overview")
        st.dataframe(df)

        top = df.iloc[0]
        st.markdown(f"""
        <div style="background:#00c6ff;padding:10px;border-radius:10px;">
        🏆 Top Candidate: {top['Resume']} ({top['ATS Score (%)']}%)
        </div>
        """, unsafe_allow_html=True)

        fig = px.bar(df, x="Resume", y="ATS Score (%)")
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download Results", csv)

    else:
        st.warning("⚠️ Please enter job description and upload resumes.")

st.markdown('</div>', unsafe_allow_html=True)