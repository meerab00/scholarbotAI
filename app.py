
import streamlit as st
from langchain_groq import ChatGroq
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ API KEY missing in .env")
    st.stop()

# =========================
# MODEL (FREE GROQ)
# =========================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🤖 AI Study Assistant")

feature = st.sidebar.selectbox(
    "Choose Feature",
    ["AI Chat", "PDF Summarizer", "Math Solver", "Quiz Generator", "Translator", "Formula Sheet"]
)

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# =========================
# TITLE
# =========================
st.title("🤖 AI Study Assistant (Groq + LangChain)")

# =========================
# PDF PROCESSING
# =========================
pdf_text = ""
chunks = []

if uploaded_file:

    pdf_reader = PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(pdf_text)

    st.sidebar.success("PDF Loaded Successfully")

# =========================
# CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# FORMULA SHEET
# =========================
if feature == "Formula Sheet":

    st.header("📘 Calculus")

    st.latex(r"\int x^n dx = \frac{x^{n+1}}{n+1} + C")
    st.latex(r"\frac{d}{dx}(x^n)=nx^{n-1}")

    st.header("📗 Algebra")

    st.latex(r"(a+b)^2 = a^2 + b^2 + 2ab")

    st.header("📙 Geometry")

    st.latex(r"Area = \pi r^2")

# =========================
# INPUT
# =========================
prompt = st.chat_input("Ask something...")

# =========================
# MAIN LOGIC
# =========================
if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # =====================
        # AI CHAT
        # =====================
        if feature == "AI Chat":

            response = llm.invoke(prompt).content

        # =====================
        # PDF SUMMARIZER (SAFE)
        # =====================
        elif feature == "PDF Summarizer":

            if not uploaded_file or len(chunks) == 0:
                response = "⚠ Please upload a PDF first."

            else:

                summaries = []

                for chunk in chunks[:5]:

                    result = llm.invoke(
                        f"Summarize this in simple English:\n{chunk}"
                    ).content

                    summaries.append(result)

                response = llm.invoke(
                    f"Combine these summaries into one final summary:\n{summaries}"
                ).content

        # =====================
        # MATH SOLVER (LATEX ENABLED)
        # =====================
        elif feature == "Math Solver":

            full_prompt = f"""
Solve step by step.

IMPORTANT:
- Use LaTeX for all formulas
- Show clear steps
- Give final answer

Problem: {prompt}
"""

            response = llm.invoke(full_prompt).content

            st.markdown("### 🧮 Step-by-Step Solution")

        # =====================
        # QUIZ GENERATOR
        # =====================
        elif feature == "Quiz Generator":

            response = llm.invoke(
                f"Generate 10 MCQs with answers:\n{prompt}"
            ).content

        # =====================
        # TRANSLATOR
        # =====================
        elif feature == "Translator":

            response = llm.invoke(
                f"Translate into Urdu:\n{prompt}"
            ).content

        else:
            response = "Feature not available."

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Made with ❤️ using LangChain + Groq + Streamlit")
