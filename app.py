from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
import os
import re
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

app = Flask(__name__, template_folder="templates")
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
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/health")
def health():
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
        if c["start_date"]
        and c["start_date"].year == today.year
        and c["start_date"].month == today.month
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
        if c["start_date"]
        and c["start_date"].year == year
        and c["start_date"].month == month
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
        "ميكانيكا": ["ميكانيكا", "ميكانيكية", "hvac", "تكييف", "مضخات"]
    }

    for _, keywords in categories.items():
        for kw in keywords:
            if kw in q and kw in name:
                return True

    return False

def broad_match(question, course_name):
    q = normalize_text(question)
    name = normalize_text(course_name)

    q_words = set(q.split())
    name_words = set(name.split())

    overlap = len(q_words & name_words)

    if q in name:
        overlap += 10

    for word in q_words:
        if word in name:
            overlap += 1

    return overlap

def filter_by_question(courses, question):
    q = normalize_text(question)
    valid_courses = [c for c in courses if not_expired(c)]

    if "هذا الشهر" in q or "الشهر الحالي" in q:
        month_courses = filter_this_month(valid_courses)

        if len(month_courses) >= 3:
            return month_courses[:5]

        extra_courses = [c for c in valid_courses if c not in month_courses]
        return (month_courses + extra_courses)[:5]

    if "الشهر القادم" in q or "الشهر الجاي" in q:
        next_month_courses = filter_next_month(valid_courses)

        if len(next_month_courses) >= 3:
            return next_month_courses[:5]

        extra_courses = [c for c in valid_courses if c not in next_month_courses]
        return (next_month_courses + extra_courses)[:5]

    filtered_by_cat = [c for c in valid_courses if match_category(question, c["name"])]

    if filtered_by_cat:
        return filtered_by_cat[:5]

    scored = []

    for c in valid_courses:
        score = broad_match(question, c["name"])
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    strong_matches = [c for s, c in scored if s > 0][:5]

    if strong_matches:
        return strong_matches

    return valid_courses[:5]

def format_course_line(course):
    start_str = course["start_date"].strftime("%Y-%m-%d") if course["start_date"] else "غير محدد"
    end_str = course["end_date"].strftime("%Y-%m-%d") if course["end_date"] else "غير محدد"
    price_str = course["price"] if course["price"] else "غير محدد"
    period_str = course["period"] if course["period"] else "غير محدد"
    session_str = course["session"] if course["session"] else "غير محدد"

    return (
        f"اسم الدورة: {course['name']} | "
        f"يبدأ في: {start_str} | "
        f"ينتهي في: {end_str} | "
        f"التكلفة للمتدرب: {price_str} | "
        f"وقت الدورة: {period_str} | "
        f"الفترة: {session_str}"
    )

def courses_to_context(courses):
    return "\n\n".join([format_course_line(c) for c in courses[:10]])

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"reply": "ما وصلني سؤال."}), 400

    all_courses = get_all_courses()
    matched_courses = filter_by_question(all_courses, question)

    if not matched_courses:
        return jsonify({
            "reply": "ما لقيت دورات مناسبة حسب طلبك أو كل الدورات المطابقة منتهية."
        })

    context = courses_to_context(matched_courses)

    prompt = f"""
أنت مساعد ذكي لجمعية المهندسين الكويتية.

تعليمات مهمة جدًا:
- اعتمد فقط على السياق.
- لا تعرض أي دورة منتهية.
- إذا كان السؤال عن هذا الشهر أو الشهر القادم، اعرض قائمة الدورات المناسبة فقط.
- إذا كانت النتائج قليلة، اعرض أقرب دورات متاحة غير منتهية.
- إذا كان السؤال عن تخصص مثل كهرباء أو مباني أو طاقة، اعرض الدورات المتعلقة بهذا التخصص فقط.
- إذا كانت النتيجة أكثر من دورة، اعرضها بشكل مرتب وواضح.
- استخدم هذا التنسيق دائمًا لكل دورة:
📚 اسم الدورة: ...
📅 تاريخ البداية: ...
📅 تاريخ النهاية: ...
🕒 الوقت: ...
💰 السعر: ...

- اترك سطر فارغ بين كل دورة والثانية.
- إذا كانت دورة واحدة فقط، اعرضها بنفس الشكل لكن بدون مقدمة طويلة.
- إذا سأل عن الموقع الرسمي فاذكر: https://www.kse.org.kw
- جاوب بالعربية وباختصار ووضوح.
- لا تستخدم markdown مثل ** أو ##.

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