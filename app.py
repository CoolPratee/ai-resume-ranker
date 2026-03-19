import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI CONFIG ---------------- #
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c92d2);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass container */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 15px;
    padding: 25px;
    margin-top: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Headings */
h3, h4 {
    color: #ffffff !important;
    font-weight: 700;
}

/* Inputs */
textarea {
    background-color: rgba(255,255,255,0.95) !important;
    color: black !important;
    border-radius: 10px !important;
}

/* Upload Box */
section[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 12px;
}

/* Browse Button */
button[kind="secondary"] {
    background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Main Button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 230px;
    font-size: 18px;
}

/* Table */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 12px;
}

/* Top card */
.top-card {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ---------------- #

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text

    elif file.name.endswith(".docx"):
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + " "
        return text

    return ""


def preprocess(text):
    return text.lower()


# ---------------- HEADER ---------------- #

st.markdown("""
<h1 style='text-align:center;'>📄 AI Resume Ranker</h1>
<p style='text-align:center;'>🚀 Smart AI system to rank candidates intelligently</p>
""", unsafe_allow_html=True)

# Tech GIF (no human)
st.markdown("""
<div style="text-align:center;">
<img src="https://media.giphy.com/media/QssGEmpkyEOhBCb7e1/giphy.gif" width="250">
</div>
""", unsafe_allow_html=True)

# ---------------- MAIN UI ---------------- #

st.markdown('<div class="glass">', unsafe_allow_html=True)

st.markdown("<h3>📥 Input Details</h3>", unsafe_allow_html=True)

# ✅ CUSTOM LABELS (VISIBLE)
st.markdown("<h4>📝 Enter Job Description</h4>", unsafe_allow_html=True)
job_desc = st.text_area("", label_visibility="collapsed")

st.markdown("<h4>📂 Upload Resumes (PDF / Word)</h4>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# ---------------- PROCESS ---------------- #

if st.button("🚀 Rank Resumes"):
    if job_desc and uploaded_files:

        resumes = []
        names = []

        for file in uploaded_files:
            text = extract_text(file)
            processed = preprocess(text)
            resumes.append(processed)
            names.append(file.name)

        processed_jd = preprocess(job_desc)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_jd] + resumes)

        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Skill Matching
        skills_list = [
            "python", "sql", "excel", "machine learning",
            "data analysis", "power bi", "tableau",
            "aws", "azure", "deep learning"
        ]

        def extract_skills(text):
            return {skill for skill in skills_list if skill in text}

        jd_skills = extract_skills(processed_jd)

        results = []

        for i, resume in enumerate(resumes):
            score = scores[i]
            resume_skills = extract_skills(resume)

            matched = jd_skills.intersection(resume_skills)
            missing = jd_skills - resume_skills

            match_percent = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0

            results.append({
                "Resume": names[i],
                "Score": round(score, 3),
                "Match %": round(match_percent, 2),
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing)
            })

        result = pd.DataFrame(results).sort_values(by="Score", ascending=False)

        # ---------------- OUTPUT ---------------- #

        st.markdown("<h3>📊 Results Overview</h3>", unsafe_allow_html=True)
        st.dataframe(result.style.background_gradient(cmap='viridis'))

        # Top Candidate
        top = result.iloc[0]
        st.markdown(f"""
        <div class="top-card">
        🏆 Top Candidate: {top['Resume']}
        </div>
        """, unsafe_allow_html=True)

        # ---------------- DASHBOARD ---------------- #

        st.markdown("---")
        st.markdown("<h3>📊 Analytics Dashboard</h3>", unsafe_allow_html=True)

        fig1 = px.bar(result, x="Resume", y="Score", color="Score",
                      title="Resume Score Comparison")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(result, x="Resume", y="Match %",
                      color="Match %", title="Skill Match Percentage")
        st.plotly_chart(fig2, use_container_width=True)

        matched_count = len(top["Matched Skills"].split(", ")) if top["Matched Skills"] else 0
        missing_count = len(top["Missing Skills"].split(", ")) if top["Missing Skills"] else 0

        fig3 = px.pie(values=[matched_count, missing_count],
                      names=["Matched", "Missing"],
                      title="Top Candidate Skill Coverage")
        st.plotly_chart(fig3, use_container_width=True)

        # Download
        csv = result.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "resume_ranking.csv")

    else:
        st.warning("⚠️ Please enter job description and upload resumes.")

# Close glass UI
st.markdown('</div>', unsafe_allow_html=True)