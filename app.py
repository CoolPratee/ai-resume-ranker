import streamlit as st
import pandas as pd
import PyPDF2
import spacy
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# 🌈 FINAL POLISHED UI
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c92d2);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
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
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 30px;
    margin-top: 25px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
}

/* Titles */
h1 {
    text-align: center;
    font-size: 3rem;
}

p {
    text-align: center;
    color: #cfd8dc;
}

/* Labels FIX */
label {
    color: #e0e0e0 !important;
    font-size: 15px;
    font-weight: 600;
}

/* Text area */
textarea {
    background-color: rgba(255,255,255,0.95) !important;
    color: black !important;
    border-radius: 12px !important;
}

/* Upload box */
section[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 12px;
}

/* Browse button FIX */
button[kind="secondary"] {
    background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Main button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 230px;
    font-size: 18px;
    margin-top: 15px;
}

.stButton>button:hover {
    transform: translateY(-2px);
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

# Extract PDF text
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Preprocess text
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Config
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

# Header
st.markdown("""
<h1>📄 AI Resume Ranker</h1>
<p>🚀 AI-powered intelligent hiring system</p>
""", unsafe_allow_html=True)

# Tech GIF (no human)
st.markdown("""
<div style="text-align:center;">
<img src="https://media.giphy.com/media/QssGEmpkyEOhBCb7e1/giphy.gif" width="300">
</div>
""", unsafe_allow_html=True)

# Glass start
st.markdown('<div class="glass">', unsafe_allow_html=True)

st.markdown("### 📥 Input Details")

job_desc = st.text_area("📝 Enter Job Description")
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF only)", accept_multiple_files=True)

# Button
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

        st.markdown("### 📊 Results Overview")
        st.dataframe(result.style.background_gradient(cmap='viridis'))

        top = result.iloc[0]
        st.markdown(f"""
        <div class="top-card">
        🏆 Top Candidate: {top['Resume']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📊 Analytics Dashboard")

        fig1 = px.bar(result, x="Resume", y="Score", color="Score",
                      title="Resume Score Comparison", color_continuous_scale="Blues")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(result, x="Resume", y="Match %",
                      color="Match %", title="Skill Match Percentage",
                      color_continuous_scale="Teal")
        st.plotly_chart(fig2, use_container_width=True)

        matched_count = len(top["Matched Skills"].split(", ")) if top["Matched Skills"] else 0
        missing_count = len(top["Missing Skills"].split(", ")) if top["Missing Skills"] else 0

        fig3 = px.pie(values=[matched_count, missing_count],
                      names=["Matched", "Missing"],
                      title="Top Candidate Skill Coverage")
        st.plotly_chart(fig3, use_container_width=True)

        csv = result.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "resume_ranking.csv")

    else:
        st.warning("⚠️ Please enter job description and upload resumes.")

# Glass end
st.markdown('</div>', unsafe_allow_html=True)