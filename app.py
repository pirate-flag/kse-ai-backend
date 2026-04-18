from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# تحميل FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

db = FAISS.load_local(
    "faiss_index_multi",
    embeddings,
    allow_dangerous_deserialization=True
)

@app.route("/")
def home():
    return "Backend is running"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"reply": "ما وصلني سؤال."}), 400

    results = db.similarity_search(question, k=5)

    if not results:
        return jsonify({"reply": "ما لقيت معلومات كافية."})

    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""
أنت مساعد ذكي لجمعية المهندسين الكويتية.

اعتمد فقط على المعلومات الموجودة في السياق.
إذا وجدت الدورة أو معلومات قريبة جدًا من السؤال، فأجب مباشرة.
لا تقل لا توجد بيانات إلا إذا لم يظهر شيء متعلق بالسؤال في السياق.
إذا سأل عن الموقع الرسمي فاذكر: https://www.kse.org.kw
جاوب باختصار وبشكل واضح.

السؤال:
{question}

السياق:
{context}
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )

    return jsonify({"reply": response.output_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)