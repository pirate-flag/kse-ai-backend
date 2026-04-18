from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import re

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# مهم: FakeEmbeddings فقط لتحميل FAISS بدون تحميل موديل ثقيل
embeddings = FakeEmbeddings(size=384)

db = FAISS.load_local(
    "faiss_index_multi",
    embeddings,
    allow_dangerous_deserialization=True
)

@app.route("/")
def home():
    return "Backend is running"

def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def get_documents():
    docs = []
    for _, doc in db.docstore._dict.items():
        docs.append(doc.page_content)
    return docs

def lexical_search(question, k=5):
    question_norm = normalize_text(question)
    q_words = set(question_norm.split())

    scored = []
    for doc in get_documents():
        doc_norm = normalize_text(doc)
        doc_words = set(doc_norm.split())

        overlap = len(q_words & doc_words)
        bonus = 0

        # بونص إذا السؤال بالكامل أو جزء كبير منه موجود بالنص
        if question_norm in doc_norm:
            bonus += 10

        # بونص بسيط لبعض الكلمات المهمة
        for word in q_words:
            if word in doc_norm:
                bonus += 1

        score = overlap + bonus
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored if score > 0][:k]
    return top_docs

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"reply": "ما وصلني سؤال."}), 400

    results = lexical_search(question, k=5)

    if not results:
        return jsonify({"reply": "ما لقيت معلومات كافية."})

    context = "\n\n".join(results)

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