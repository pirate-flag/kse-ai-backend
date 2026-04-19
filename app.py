from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import re
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def parse_course(doc_text):
    data = {}

    parts = doc_text.split(" | ")
    for part in parts:
        if ":" in part:
            key, value = part.split(":", 1)
            data[key.strip()] = value.strip()

    course_name = data.get("اسم الدورة", "")
    start_raw = data.get("يبدأ في", "")
    end_raw = data.get("ينتهي في", "")
    price = data.get("التكلفة للمتدرب", "")
    period = data.get("وقت الدورة", "")
    session = data.get("الفترة", "")

    start_date = None
    end_date = None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            if start_raw and not start_date:
                start_date = datetime.strptime(start_raw, fmt)
        except:
            pass
        try:
            if end_raw and not end_date:
                end_date = datetime.strptime(end_raw, fmt)
        except:
            pass

    return {
        "name": course_name,
        "start_date": start_date,
        "end_date": end_date,
        "price": price,
        "period": period,
        "session": session,
        "raw": doc_text
    }

def get_all_courses():
    courses = []
    for _, doc in db.docstore._dict.items():
        courses.append(parse_course(doc.page_content))
    return courses

def not_expired(course):
    today = datetime.today()
    if course["end_date"]:
        return course["end_date"].date() >= today.date()
    if course["start_date"]:
        return course["start_date"].date() >= today.date()
    return True

def filter_this_month(courses):
    today = datetime.today()
    return [
        c for c in courses
        if c["start_date"] and c["start_date"].year == today.year and c["start_date"].month == today.month
    ]

def filter_next_month(courses):
    today = datetime.today()
    year = today.year
    month = today.month + 1
    if month == 13:
        month = 1
        year += 1

    return [
        c for c in courses
        if c["start_date"] and c["start_date"].year == year and c["start_date"].month == month
    ]

def match_category(question, course_name):
    q = normalize_text(question)
    name = normalize_text(course_name)

    categories = {
        "كهرباء": ["كهرباء", "كهربائية", "الطاقة", "قدرة", "محولات", "مولدات"],
        "مباني": ["مباني", "إنشائية", "خرسانة", "هندسة مدنية", "مدني", "بناء"],
        "سلامة": ["سلامة", "أمن", "مخاطر", "وقاية"],
        "عقود": ["عقود", "مناقصات", "إدارة العقود"],
        "طاقة": ["طاقة", "كفاءة الطاقة", "ترشيد", "استهلاك"],
        "ميكانيكا": ["ميكانيكا", "ميكانيكية", "HVAC", "تكييف", "مضخات"]
    }

    for _, keywords in categories.items():
        for kw in keywords:
            if kw in q and kw in name:
                return True

    return False

def filter_by_question(courses, question):
    q = normalize_text(question)

    # استبعاد المنتهي أولاً
    courses = [c for c in courses if not_expired(c)]

    # هذا الشهر
    if "هذا الشهر" in q or "الشهر الحالي" in q:
        courses = filter_this_month(courses)

    # الشهر القادم
    elif "الشهر القادم" in q or "الشهر الجاي" in q:
        courses = filter_next_month(courses)

    # فلترة بالتخصص/النوع
    filtered_by_cat = [c for c in courses if match_category(question, c["name"])]
    if filtered_by_cat:
        courses = filtered_by_cat

    # إذا كتب اسم دورة مباشرة
    else:
        q_words = set(q.split())
        scored = []
        for c in courses:
            name_norm = normalize_text(c["name"])
            overlap = len(q_words & set(name_norm.split()))
            if q in name_norm:
                overlap += 10
            scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored and scored[0][0] > 0:
            top_score = scored[0][0]
            courses = [c for s, c in scored if s > 0 and s >= max(1, top_score - 1)][:5]

    return courses

def courses_to_context(courses):
    lines = []
    for c in courses[:10]:
        lines.append(
            f"اسم الدورة: {c['name']} | يبدأ في: {c['start_date']} | ينتهي في: {c['end_date']} | "
            f"التكلفة للمتدرب: {c['price']} | وقت الدورة: {c['period']} | الفترة: {c['session']}"
        )
    return "\n\n".join(lines)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"reply": "ما وصلني سؤال."}), 400

    all_courses = get_all_courses()
    matched_courses = filter_by_question(all_courses, question)

    if not matched_courses:
        return jsonify({"reply": "ما لقيت دورات مناسبة حسب طلبك أو كل الدورات المطابقة منتهية."})

    context = courses_to_context(matched_courses)

    prompt = f"""
أنت مساعد ذكي لجمعية المهندسين الكويتية.

تعليمات مهمة:
- اعتمد فقط على السياق.
- لا تعرض أي دورة منتهية.
- إذا كان السؤال عن هذا الشهر أو الشهر القادم، اعرض قائمة الدورات المناسبة فقط.
- إذا كان السؤال عن تخصص مثل كهرباء أو مباني أو طاقة، اعرض الدورات المتعلقة بهذا التخصص فقط.
- إذا كانت النتيجة أكثر من دورة، اعرضها كقائمة مرتبة وواضحة.
- إذا سأل عن الموقع الرسمي فاذكر: https://www.kse.org.kw
- جاوب بالعربية وباختصار ووضوح.

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