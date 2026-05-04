from flask import Flask, render_template, request, jsonify

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

app = Flask(__name__)
import requests
from bs4 import BeautifulSoup

def scrape_web(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text[:2000]  # limit size

    except:
        return "No web data retrieved."
urls = [
    "https://www.greenplan.gov.sg/"
]
documents = []

# Load PDF
pdf_loader = PyMuPDFLoader(
    "C:/Users/Ei/OneDrive - Temasek Polytechnic/2026_intern/Diploma Descriptions.pdf"
)
documents.extend(pdf_loader.load())

# Load PPTX
pptx_loader = PyMuPDFLoader(
    "C:/Users/Ei/Downloads/Group Project Brief and Template_Apr 2026_Final.pptx"
)
documents.extend(pptx_loader.load())
#SPLIT TEXT -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    template="""
You are an AI assistant helping students generate sustainability-related SCA group project ideas.

Priority order:
1. Student profile
2. Retrieved SCA guideline context

Student Profile:
- Diploma Programme: {diploma}
- Sustainability Concern: {concern}
- SDG Focus Area: {sdg}
- Project Theme: {theme}
- Preferred Presentation Mode: {presentation}

Additional Preferences:
{extra_input}

Retrieved Context (Guidelines + Web Data):
{context}

Task:
Generate 3 project ideas that best match the student profile and comply with the SCA guideline context.

For each project idea, provide:
1. Project Title
2. Short Description
3. Why It Matches the Diploma
4. Why It Aligns with the Sustainability Concern / SDG
5. Suggested Presentation Format
6. Feasibility Level

Rules:
- Prioritize SCA guideline context over web data
- Keep the ideas realistic for student groups.
- Do not suggest ideas that conflict with student preferences.
- Make the 3 ideas clearly different from each other.
- Use clear, student-friendly language.
- Format the output clearly with headings and bullet points.
- Do not ask follow-up questions.
- Provide the final answer directly.
""",
    input_variables=[
        "diploma",
        "concern",
        "sdg",
        "theme",
        "presentation",
        "extra_input",
        "context"
    ]
)
# Gemini LLM
os.environ["GOOGLE_API_KEY"] = "AIzaSyDsOO7qnN85h73cuH4OF9frGOItc5BPTY8"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)
# QA CHAIN
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def chatbot():
    return render_template("chatbot_formv2.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()

        # --- Get inputs safely ---
        diploma = (data.get("diploma") or "").strip()
        concern = (data.get("concern") or "").strip()
        sdg = (data.get("sdg") or "").strip()
        theme = (data.get("theme") or "").strip()
        presentation = data.get("presentation") or []
        extra_input = (data.get("extra_input") or "").strip()

        # --- Validation ---
        if not all([diploma, concern, sdg, theme]) or len(presentation) == 0:
            return jsonify({
                "answer": "Please complete all required fields before generating project ideas."
            })

        # --- Convert list → string ---
        if isinstance(presentation, list):
            presentation_text = ", ".join(presentation)
        else:
            presentation_text = presentation

        # --- Build retrieval query ---
        search_query = f"""
        Diploma Programme: {diploma}
        Sustainability Concern: {concern}
        SDG Focus Area: {sdg}
        Project Theme: {theme}
        Preferred Presentation Mode: {presentation_text}
        Additional Preferences: {extra_input if extra_input else "None"}
        """

        # --- Retrieve context ---
        retrieved_docs = retriever.invoke(search_query)

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        ) if retrieved_docs else "No relevant guideline context found."

        web_context = ""

        for url in urls:
            web_context += scrape_web(url) + "\n\n"

        full_context = context + "\n\n" + web_context

        # --- Format prompt ---
        prompt = prompt_template.format(
            diploma=diploma,
            concern=concern,
            sdg=sdg,
            theme=theme,
            presentation=presentation_text,
            extra_input=extra_input if extra_input else "None",
            context=full_context
        )

        # --- Call LLM ---
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return jsonify({"answer": answer})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "answer": f"Server Error: {str(e)}"
        })
    if __name__ == "__main__":
    # In Jupyter notebooks, use this instead of app.run()
    # This prevents the kernel from exiting
    from werkzeug.serving import run_simple
    
    # If you're using Flask, you can use this approach
    app.run(debug=True, use_reloader=False)
    
    # Alternatively, if the above still causes issues:
    # run_simple('localhost', 5000, app, use_reloader=False, use_debugger=True)
