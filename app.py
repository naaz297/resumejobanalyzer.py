import streamlit as st
import matplotlib.pyplot as plt
import PyPDF2
import re

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

h1{
    text-align:center;
    color:#003366;
}

h2,h3{
    color:#003366;
}

.stButton>button{
    width:100%;
    background:#4CAF50;
    color:white;
    border-radius:10px;
    height:3em;
    font-size:18px;
}

.stButton>button:hover{
    background:#2E8B57;
    color:white;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)



st.title("🚀 AI Resume Job Analyzer")

st.markdown("""
### Analyze how well your resume matches a Job Description

This application uses:

- TF-IDF
- Cosine Similarity
- NLP Keyword Extraction
- ATS Style Resume Analysis
""")


with st.sidebar:

    st.title("📘 About")

    st.info("""
This Resume Analyzer helps you:

✅ Upload Resume

✅ Paste Job Description

✅ Compare Similarity

✅ Find Missing Skills

✅ Improve ATS Score
""")

    st.markdown("---")

    st.subheader("📌 Steps")

    st.write("""
1. Upload Resume

2. Paste Job Description

3. Click Analyze

4. View ATS Report
""")



def extract_text_from_pdf(uploaded_file):

    try:

        reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        return text

    except Exception as e:

        st.error(e)

        return ""


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text


def remove_stopwords(text):

    stop_words = set(stopwords.words("english"))

    words = word_tokenize(text)

    filtered = []

    for word in words:

        if word not in stop_words:

            filtered.append(word)

    return " ".join(filtered)


def calculate_similarity(resume, job):

    resume = remove_stopwords(clean_text(resume))

    job = remove_stopwords(clean_text(job))

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform([resume, job])

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return (
        round(similarity * 100, 2),
        resume,
        job
    )


def extract_keywords(text, num=15):

    words = word_tokenize(text)

    tagged = pos_tag(words)

    nouns = []

    for word, tag in tagged:

        if tag.startswith("NN"):

            nouns.append(word.lower())

    frequency = Counter(nouns)

    keywords = []

    for word, count in frequency.most_common(num):

        keywords.append(word)

    return keywords



col1, col2 = st.columns(2)

with col1:

    uploaded_resume = st.file_uploader(
        "📄 Upload Resume (PDF)",
        type=["pdf"]
    )

with col2:

    job_description = st.text_area(
        "💼 Paste Job Description",
        height=320,
        placeholder="Paste complete job description here..."
    )

st.markdown("---")



analyze = st.button("🚀 Analyze Resume")

if analyze:

    if uploaded_resume is None:

        st.warning("Please upload your resume.")

        st.stop()

    if job_description.strip() == "":

        st.warning("Please paste a Job Description.")

        st.stop()

    with st.spinner("Analyzing Resume..."):

        resume_text = extract_text_from_pdf(uploaded_resume)

        score, resume_processed, job_processed = calculate_similarity(
            resume_text,
            job_description
        )

        resume_keywords = extract_keywords(resume_processed)

        job_keywords = extract_keywords(job_processed)

        missing_keywords = list(
            set(job_keywords) - set(resume_keywords)
        )

        

        st.success("✅ Analysis Completed Successfully!")

        st.markdown("## 📊 Resume Analysis Dashboard")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                label="🎯 Match Score",
                value=f"{score}%"
            )

        with metric2:
            st.metric(
                label="📄 Resume Keywords",
                value=len(resume_keywords)
            )

        with metric3:
            st.metric(
                label="❌ Missing Skills",
                value=len(missing_keywords)
            )

        st.markdown("---")

       

        st.subheader("📈 ATS Match Score")

        st.progress(score / 100)

        if score >= 85:
            st.success("🌟 Excellent! Your resume is highly compatible with this job.")
        elif score >= 70:
            st.info("👍 Good Match! Adding a few missing skills can improve your chances.")
        elif score >= 50:
            st.warning("⚠️ Average Match. Consider updating your resume.")
        else:
            st.error("❌ Low Match. Add more relevant skills and experience.")

        st.markdown("---")


        st.subheader("🏆 ATS Rating")

        if score >= 90:
            st.success("⭐⭐⭐⭐⭐ Outstanding")
        elif score >= 80:
            st.success("⭐⭐⭐⭐ Excellent")
        elif score >= 70:
            st.info("⭐⭐⭐ Good")
        elif score >= 60:
            st.warning("⭐⭐ Average")
        else:
            st.error("⭐ Needs Improvement")

        st.markdown("---")



        tab1, tab2, tab3 = st.tabs(
            [
                "📄 Resume Keywords",
                "💼 Job Keywords",
                "❌ Missing Skills"
            ]
        )

        with tab1:

            st.write("### Resume Keywords")

            st.write(", ".join(resume_keywords))

        with tab2:

            st.write("### Job Keywords")

            st.write(", ".join(job_keywords))

        with tab3:

            st.write("### Missing Skills")

            if missing_keywords:

                for skill in sorted(missing_keywords):

                    st.error(skill)

            else:

                st.success("🎉 No important keywords are missing.")

        st.markdown("---")

    

        st.subheader("💡 Suggestions")

        if score >= 85:

            st.success("""
✔ Resume is highly optimized.

✔ Keep applying.

✔ Tailor projects according to job.
""")

        elif score >= 70:

            st.info("""
• Add missing technical skills.

• Improve project descriptions.

• Include measurable achievements.

• Match wording with the Job Description.
""")

        else:

            st.warning("""
• Add relevant programming languages.

• Mention frameworks used.

• Include internships/projects.

• Add certifications.

• Use keywords from Job Description.
""")

        st.markdown("---")


        st.subheader("📊 Match Visualization")

        fig, ax = plt.subplots(figsize=(6,4))

        bars = ax.bar(
            ["Resume Match"],
            [score],
            width=0.5
        )

        if score >= 80:
            bars[0].set_color("green")
        elif score >= 60:
            bars[0].set_color("orange")
        else:
            bars[0].set_color("red")

        ax.set_ylim(0,100)

        ax.set_ylabel("Percentage")

        ax.set_title("Resume vs Job Description")

        ax.text(
            0,
            score + 2,
            f"{score:.2f}%",
            ha="center",
            fontsize=12,
            fontweight="bold"
        )

        st.pyplot(fig)

        st.markdown("---")

     

        report = f"""
===============================
        AI RESUME ANALYZER
===============================

Resume Match Score : {score}%

Resume Keywords:
{", ".join(resume_keywords)}

Job Keywords:
{", ".join(job_keywords)}

Missing Skills:
{", ".join(missing_keywords)}

Suggestions:
"""

        if score >= 85:
            report += "Excellent Resume. Ready to Apply."
        elif score >= 70:
            report += "Good Resume. Add a few missing skills."
        else:
            report += "Improve resume using missing keywords."

        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name="resume_analysis_report.txt",
            mime="text/plain"
        )

        st.markdown("---")


        st.markdown(
            """
            <hr>
            <center>

            <h4>🚀 AI Resume Job Analyzer</h4>

            Built with ❤️ using

            <br><br>

            Python • Streamlit • NLP • TF-IDF • Cosine Similarity

            </center>
            """,
            unsafe_allow_html=True
        )
