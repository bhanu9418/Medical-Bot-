from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.prompt import system_prompt


app = Flask(__name__)

# ---------------- ENV ----------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing API keys in .env")

# ---------------- EMBEDDINGS ----------------
embeddings = download_hugging_face_embeddings()

# ---------------- VECTOR STORE ----------------
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ---------------- LLM ----------------
chatModel = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    api_key=GROQ_API_KEY,
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# ---------------- SAFE v1.x RAG (replacement for chains) ----------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": lambda x: format_docs(
            retriever.invoke(x["input"])
        ),
        "input": RunnablePassthrough(),
    }
    | prompt
    | chatModel
)

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")

    if not msg:
        return "No input received"

    print("User:", msg)

    response = rag_chain.invoke({"input": msg})

    # v1.x returns AIMessage
    answer = response.content

    print("Bot:", answer)
    return answer

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)