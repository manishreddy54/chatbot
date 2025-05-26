from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnableMap
from langchain_core.chat_history import InMemoryChatMessageHistory
import os
from flask import send_from_directory



os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- LangChain setup ---
system_template = (
    "You are a friendly, helpful chiropractic clinic assistant. "
    "Greet users warmly and answer their questions clearly and empathetically. "
    "You should only answer questions related to chiropractic care, appointments, clinic hours, and services offered. "
    "If a user asks a question that is not relevant to chiropractic care or the clinic, politely inform them that you can only assist with chiropractic-related inquiries, and offer to connect them with a staff member if appropriate. "
    "Do not attempt to answer questions outside your area of expertise."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{question}")
])

loader = TextLoader("chiropractic_clinic_faq.txt")
documents = loader.load()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()
llm = ChatGroq(
    model_name = "llama3-70b-8192",
    api_key=os.environ.get("GROQ_API_KEY")
)

def get_memory(session_id):
    return InMemoryChatMessageHistory()

chain = (
    RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"],
    })
    | prompt
    | llm
)
chatbot = RunnableWithMessageHistory(
    chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# --- Flask setup ---
app = Flask(__name__)
from flask_cors import CORS
CORS(app)


@app.route('/')
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    session_id = data.get("session_id", "user-session-1")
    response = chatbot.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    return jsonify({"answer": response.content})

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))  # 5000 is fallback for local dev
#     app.run(host="0.0.0.0", port=port)
