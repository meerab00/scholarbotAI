
import streamlit as st
from langchain_groq import ChatGroq
from pypdf import PdfReader
from dotenv import load_dotenv
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🤖",
    layout="wide"
)

# =========================
# LOAD ENV
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# CHECK API KEY
# =========================
if not GROQ_API_KEY:
    st.error("Groq API Key Not Found")
    st.stop()

# =========================
# LOAD MODEL
# =========================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
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
    [
        "AI Chat",
        "PDF Summarizer",
        "Math Solver",
        "Quiz Generator",
        "Translator",
        "Formula Sheet"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# =========================
# TITLE
# =========================
st.title("🤖 AI Study Assistant")
st.write("LangChain + Groq AI Agent")

# =========================
# PDF TEXT EXTRACTION
# =========================
pdf_text = ""

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        text = page.extract_text()

        if text:
            pdf_text += text

    st.sidebar.success("PDF Uploaded Successfully")

# =========================
# CHAT HISTORY
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# FORMULA SHEET
# =========================
if feature == "Formula Sheet":

    st.header("📘 Algebra")

    st.latex(r"(a+b)^2 = a^2 + b^2 + 2ab")
    st.latex(r"(a-b)^2 = a^2 + b^2 - 2ab")

    st.header("📗 Calculus")

    st.latex(r"\frac{d}{dx}(x^n)=nx^{n-1}")
    st.latex(r"\int x^n dx = \frac{x^{n+1}}{n+1}")

    st.header("📙 Geometry")

    st.latex(r"Area = \pi r^2")
    st.latex(r"Circumference = 2\pi r")

# =========================
# USER INPUT
# =========================
prompt = st.chat_input("Ask something...")

# =========================
# MAIN AI LOGIC
# =========================
if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        try:

            # =========================
            # AI CHAT
            # =========================
            if feature == "AI Chat":

                response = llm.invoke(prompt).content

            # =========================
            # PDF SUMMARIZER
            # =========================
            elif feature == "PDF Summarizer":

                if pdf_text == "":
                    response = "Please upload a PDF first."

                else:

                    full_prompt = f"""
                    Summarize this PDF in easy English:

                    {pdf_text}
                    """

                    response = llm.invoke(full_prompt).content

            # =========================
            # MATH SOLVER
            # =========================
            elif feature == "Math Solver":

                full_prompt = f"""
                Solve this math problem step by step:

                {prompt}
                """

                elif feature == "Math Solver":

    full_prompt = f"""
Solve this math problem step by step.

IMPORTANT:
- Use proper LaTeX format for all equations
- Show clear steps
- Give final answer at the end

Problem:
{prompt}
"""

    response = llm.invoke(full_prompt).content

    st.markdown("### 🧮 Solution")
    st.markdown(response)

            # =========================
            # QUIZ GENERATOR
            # =========================
            elif feature == "Quiz Generator":

                full_prompt = f"""
                Generate 10 MCQs with answers about:

                {prompt}
                """

                response = llm.invoke(full_prompt).content

            # =========================
            # TRANSLATOR
            # =========================
            elif feature == "Translator":

                full_prompt = f"""
                Translate this into Urdu:

                {prompt}
                """

                response = llm.invoke(full_prompt).content

            else:
                response = "Feature not available."

        except Exception as e:
            response = f"Error: {str(e)}"

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Made with LangChain + Groq + Streamlit")
