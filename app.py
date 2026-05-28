from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
import threading
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os

# ── Heavy libs loaded lazily (only when first used) ─────────────
# This means Flask starts in <1 second even if DeepFace/TF is slow
_cv2 = None
_DeepFace = None
_np = None
_Image = None

def get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _cv2_mod
        _cv2 = _cv2_mod
    return _cv2

def get_DeepFace():
    global _DeepFace
    if _DeepFace is None:
        from deepface import DeepFace as _DF
        _DeepFace = _DF
    return _DeepFace

def get_np():
    global _np
    if _np is None:
        import numpy as _np_mod
        _np = _np_mod
    return _np

def get_Image():
    global _Image
    if _Image is None:
        from PIL import Image as _Im
        _Image = _Im
    return _Image

# ── App setup ────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "faceask_iilm_2026"

DB_PATH = "faceask.db"

UPLOAD_FOLDER = "uploads"
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_PDF_EXT   = {"pdf"}
ALLOWED_EXT       = ALLOWED_IMAGE_EXT | ALLOWED_PDF_EXT
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── DB helpers ───────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT    UNIQUE NOT NULL,
        password TEXT    NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS upload_history (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        username   TEXT    NOT NULL,
        filename   TEXT    NOT NULL,
        filetype   TEXT    NOT NULL,
        emotion    TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        hashed = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def save_upload_history(username, filename, filetype, emotion):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO upload_history (username, filename, filetype, emotion) VALUES (?, ?, ?, ?)",
            (username, filename, filetype, emotion)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_upload_history(username, limit=10):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT filename, filetype, emotion, created_at FROM upload_history "
            "WHERE username = ? ORDER BY created_at DESC LIMIT ?",
            (username, limit)
        )
        rows = c.fetchall()
        conn.close()
        return [{"filename": r[0], "filetype": r[1], "emotion": r[2], "created_at": r[3]} for r in rows]
    except Exception:
        return []

# ── Suggestions ──────────────────────────────────────────────────
SUGGESTIONS = {
    "happy":    {"message": "You're Happy! Keep the energy going!",                "activities": ["Play chess or cricket!", "Call a close friend!", "Listen to upbeat music!", "Try a new hobby!", "Go for a workout!"],                  "therapist": False},
    "sad":      {"message": "You seem Sad. It's okay - let's help you feel better!", "activities": ["Watch a feel-good movie!", "Write in a journal!", "Go for a nature walk!", "Listen to calm music!", "Talk to a friend!"],           "therapist": True},
    "angry":    {"message": "You look Angry. Let's calm down together!",            "activities": ["Deep breathing: 4s in 4s out!", "Do some exercise!", "Listen to relaxing music!", "Write your feelings!", "Take a cold shower!"],     "therapist": True},
    "fear":     {"message": "You seem Fearful. You are safe!",                      "activities": ["5-4-3-2-1 grounding technique!", "Talk to someone you trust!", "Watch a comforting show!", "Do light yoga!", "Write your fears down!"],"therapist": True},
    "surprise": {"message": "You look Surprised! Something exciting?",              "activities": ["Learn something new!", "Share with a friend!", "Try a brain puzzle!", "Explore a new place!", "Start a creative project!"],           "therapist": False},
    "disgust":  {"message": "You seem Disgusted. Let's shift your focus!",          "activities": ["Get fresh air outside!", "Listen to your playlist!", "Organise your space!", "Watch something funny!", "Do what you enjoy!"],         "therapist": True},
    "neutral":  {"message": "You look Neutral. Let's add some spark!",              "activities": ["Try something new today!", "Go for a walk!", "Read an article!", "Play an online game!", "Call a friend!"],                          "therapist": False},
}

# ── Live camera ──────────────────────────────────────────────────
current_emotion = "neutral"
emotion_lock    = threading.Lock()

def generate_frames():
    global current_emotion
    cv2 = get_cv2()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    frame_count = 0
    COLOR_MAP = {
        "happy": (0, 255, 100), "sad": (255, 100, 50), "angry": (0, 0, 255),
        "fear": (200, 50, 200), "surprise": (0, 200, 255), "disgust": (50, 200, 50), "neutral": (200, 200, 200),
    }
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 480))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        frame_count += 1
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if frame_count % 10 == 0:
                try:
                    DeepFace = get_DeepFace()
                    result   = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False, silent=True)
                    detected = result[0]["dominant_emotion"].lower()
                    with emotion_lock:
                        current_emotion = detected
                except Exception:
                    pass
            with emotion_lock:
                emo = current_emotion
            color = COLOR_MAP.get(emo, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, emo.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

# ── Upload helpers ───────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def pil_to_bgr(pil_img):
    cv2 = get_cv2()
    np  = get_np()
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def detect_emotion_from_array(img_bgr):
    DeepFace = get_DeepFace()
    result   = DeepFace.analyze(img_bgr, actions=["emotion"], enforce_detection=False, silent=True)
    dominant = result[0]["dominant_emotion"].lower()
    all_emo  = {k.lower(): round(float(v), 2) for k, v in result[0]["emotion"].items()}
    return dominant, all_emo

def annotate_and_save(img_bgr, dominant, save_path):
    cv2 = get_cv2()
    COLOR_MAP = {
        "happy": (0, 255, 100), "sad": (255, 100, 50), "angry": (0, 0, 255),
        "fear": (200, 50, 200), "surprise": (0, 200, 255), "disgust": (50, 200, 50),
        "neutral": (200, 200, 200), "no_face": (128, 128, 128),
    }
    color = COLOR_MAP.get(dominant, (0, 255, 0))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    out   = img_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 3)
        cv2.putText(out, dominant.upper(), (x, max(y - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imwrite(save_path, out)

def detect_from_image_file(filepath):
    cv2 = get_cv2()
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        Image = get_Image()
        img_bgr = pil_to_bgr(Image.open(filepath))

    dominant, all_emo = detect_emotion_from_array(img_bgr)

    ann_name = "ann_" + os.path.basename(filepath)
    ann_path = os.path.join(app.config["UPLOAD_FOLDER"], ann_name)
    annotate_and_save(img_bgr, dominant, ann_path)

    return {
        "emotion":      dominant,
        "all_emotions": all_emo,
        "annotated_url": url_for("uploaded_file", filename=ann_name),
    }

def detect_from_pdf_file(filepath):
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise RuntimeError("pdf2image not installed. Run: pip install pdf2image  and install poppler.")

    pages = convert_from_path(filepath, dpi=150)
    if not pages:
        raise RuntimeError("Could not extract pages from PDF.")

    page_results = []
    for i, pil_page in enumerate(pages, 1):
        img_bgr = pil_to_bgr(pil_page)
        try:
            dominant, all_emo = detect_emotion_from_array(img_bgr)
        except Exception:
            dominant, all_emo = "no_face", {}

        ann_name = f"pdf_p{i}_{os.path.basename(filepath)}.jpg"
        ann_path = os.path.join(app.config["UPLOAD_FOLDER"], ann_name)
        annotate_and_save(img_bgr, dominant, ann_path)

        page_results.append({
            "page": i,
            "emotion": dominant,
            "all_emotions": all_emo,
            "annotated_url": url_for("uploaded_file", filename=ann_name),
        })

    valid   = [r["emotion"] for r in page_results if r["emotion"] != "no_face"]
    overall = max(set(valid), key=valid.count) if valid else "neutral"
    return {"overall_emotion": overall, "pages": page_results}

# ── Routes ───────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("dashboard"))
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            error = "Incorrect username or password."
    return render_template("index.html", page="login", error=error)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user" in session:
        return redirect(url_for("dashboard"))
    error = success = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm",  "").strip()
        if not username or not password:
            error = "Username and password are required."
        elif len(password) < 4:
            error = "Password must be at least 4 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif create_user(username, password):
            success = "Account created! Please login."
        else:
            error = "Username already exists. Try another."
    return render_template("index.html", page="signup", error=error, success=success)

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", page="dashboard", username=session["user"])

@app.route("/video_feed")
def video_feed():
    if "user" not in session:
        return redirect(url_for("login"))
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/emotion_data")
def emotion_data():
    if "user" not in session:
        return jsonify({"emotion": "neutral"})
    with emotion_lock:
        emo = current_emotion
    s = SUGGESTIONS.get(emo, SUGGESTIONS["neutral"])
    return jsonify({"emotion": emo, "message": s["message"], "activities": s["activities"], "therapist": s["therapist"]})

@app.route("/api/upload_emotion", methods=["POST"])
def api_upload_emotion():
    if "user" not in session:
        return jsonify({"error": "Unauthorised"}), 401
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported file type. Use: {', '.join(sorted(ALLOWED_EXT))}"}), 400

    filename = secure_filename(f.filename)
    ext      = filename.rsplit(".", 1)[1].lower()
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    try:
        if ext in ALLOWED_IMAGE_EXT:
            result   = detect_from_image_file(filepath)
            emotion  = result["emotion"]
            filetype = "image"
        else:
            result   = detect_from_pdf_file(filepath)
            emotion  = result["overall_emotion"]
            filetype = "pdf"

        s = SUGGESTIONS.get(emotion, SUGGESTIONS["neutral"])
        result.update({"emotion": emotion, "filetype": filetype,
                        "message": s["message"], "activities": s["activities"], "therapist": s["therapist"]})
        save_upload_history(session["user"], filename, filetype, emotion)
        return jsonify(result)

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/api/upload_history")
def api_upload_history():
    if "user" not in session:
        return jsonify([]), 401
    return jsonify(get_upload_history(session["user"]))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    init_db()
    print("✅ FaceAsk started — open http://localhost:5000")
    print("   (DeepFace/camera loads only when you use those features)")
    app.run(debug=False)   # debug=False prevents double startup
