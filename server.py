import os
import uuid
import re
import shutil

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from agents_v3 import AzureOpenAIClient, GuardrailAgent, OrchestratorAgent, VoiceOrchestratorAgent
import db


HIST_TAG_START = "<!--HISTORICAL_DOC_TEXT-->"
HIST_TAG_END = "<!--END_HISTORICAL_DOC_TEXT-->"


def strip_hidden_doc_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        re.escape(HIST_TAG_START) + r".*?" + re.escape(HIST_TAG_END),
        "",
        text,
        flags=re.DOTALL,
    ).strip()


# ensure uploads dir exists
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__)

# CORS (en prod mejor limitar por FRONTEND_ORIGIN)
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/api/*": {"origins": frontend_origin}, r"/health/*": {"origins": frontend_origin}})


# -------------------------
# Health / Admin endpoints
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/health/db")
def health_db():
    """
    No toca tablas, solo comprueba que la conexión auth/red funciona.
    """
    try:
        import psycopg2
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            return jsonify({"ok": False, "error": "DATABASE_URL missing"}), 500
        conn = psycopg2.connect(dsn)
        with conn.cursor() as cur:
            cur.execute("SELECT current_user, current_database();")
            user, dbname = cur.fetchone()
        conn.close()
        return jsonify({"ok": True, "current_user": user, "database": dbname})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/admin/init-db")
def admin_init_db():
    """
    Inicializa tablas bajo demanda para evitar que Gunicorn muera al arrancar.
    Protegido con token simple.
    """
    token = os.getenv("INIT_DB_TOKEN", "")
    sent = request.headers.get("x-init-token", "")
    if token and sent != token:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    try:
        db.init_db()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def save_uploads(files, session_id: str) -> list:
    """Save uploaded files into uploads/{session_id}/ and return stored paths."""
    out_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(out_dir, exist_ok=True)
    stored_paths = []
    for f in files:
        orig = f.filename or "upload"
        suffix = os.path.splitext(orig)[1]
        stored_name = f"{uuid.uuid4().hex}{suffix}"
        path = os.path.join(out_dir, stored_name)
        with open(path, "wb") as out:
            out.write(f.read())
        stored_paths.append(path)
    return stored_paths


@app.route("/api/query", methods=["POST"])
def handle_query():
    # Expects form-data: 'query' and files under 'files'
    query = request.form.get("query", "")
    voice_mode = request.form.get("voice_mode", "false").lower() == "true"
    uploaded = request.files.getlist("files") or []
    session_id = request.form.get("session_id") or uuid.uuid4().hex

    # ensure session exists
    db.create_session(session_id)

    filenames = [f.filename for f in uploaded]

    try:
        client = AzureOpenAIClient()
    except Exception as e:
        return jsonify({"ok": False, "error": f"Azure client setup error: {str(e)}"}), 500

    # Guardrail only on first message
    guard = GuardrailAgent(client=client)
    recent_one = db.get_recent_messages(session_id, limit=1)
    is_first_message = len(recent_one) == 0

    if is_first_message:
        valid, reason = guard.validate(query, filenames)
        if not valid:
            return jsonify({"ok": False, "rejected": True, "reason": reason}), 400

        qnorm = (query or "").strip().lower()
        if qnorm in ("financiera", "legal"):
            # persist intent
            db.add_message(session_id, "user", query)
            db.add_message(session_id, "system", f"intent:{qnorm}")
            confirm = f"Intento registrado: '{qnorm}'. Ahora puedes enviar tu consulta o adjuntar archivos."
            return jsonify({"ok": True, "reply": confirm, "session_id": session_id, "voice_mode": voice_mode})

    filepaths = save_uploads(uploaded, session_id)

    try:
        orch = VoiceOrchestratorAgent(client) if voice_mode else OrchestratorAgent(client)

        # fetch recent messages for context
        recent = db.get_recent_messages(session_id, limit=50)
        session_history = [{"role": r["role"], "content": r["content"]} for r in recent]

        reply_with_tags = orch.respond(query, filepaths, session_history=session_history)

        # IMPORTANT: return clean reply to user, but persist full reply for memory
        reply_clean = strip_hidden_doc_tags(reply_with_tags)

    except Exception as e:
        return jsonify({"ok": False, "error": f"Processing error: {str(e)}"}), 500

    # persist messages
    db.add_message(session_id, "user", query)
    db.add_message(session_id, "assistant", reply_with_tags)

    return jsonify({"ok": True, "reply": reply_clean, "session_id": session_id, "voice_mode": voice_mode})


@app.route("/api/sessions", methods=["POST"])
def create_session():
    sid = db.create_session()
    welcome = "Bienvenido. Por favor indica junto a tu mensaje si tu consulta será 'financiera' o 'legal'."
    return jsonify({"ok": True, "session_id": sid, "welcome": welcome})


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    lst = db.list_sessions()
    return jsonify({"ok": True, "sessions": lst})


@app.route("/api/sessions", methods=["DELETE"])
def delete_all_sessions():
    db.delete_all_sessions()

    if os.path.isdir(UPLOADS_DIR):
        try:
            shutil.rmtree(UPLOADS_DIR)
        except Exception:
            pass
        os.makedirs(UPLOADS_DIR, exist_ok=True)

    return jsonify({"ok": True})


@app.route("/api/sessions/<session_id>/clear", methods=["POST"])
def clear_session(session_id):
    if not db.session_exists(session_id):
        return jsonify({"ok": False, "error": "session not found"}), 404
    db.clear_session(session_id)
    return jsonify({"ok": True})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    if not db.session_exists(session_id):
        return jsonify({"ok": False, "error": "session not found"}), 404
    db.delete_session(session_id)

    # remove uploaded files for this session, if any
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    if os.path.isdir(session_dir):
        try:
            shutil.rmtree(session_dir)
        except Exception:
            pass

    return jsonify({"ok": True})


@app.route("/api/sessions/<session_id>/messages", methods=["GET"])
def get_session_messages(session_id):
    if not db.session_exists(session_id):
        return jsonify({"ok": False, "error": "session not found"}), 404
    limit = int(request.args.get("limit", 200))
    msgs = db.get_recent_messages(session_id, limit=limit)

    # Optionally hide tags from the returned messages to frontend
    for m in msgs:
        if m.get("role") == "assistant":
            m["content"] = strip_hidden_doc_tags(m.get("content", ""))

    return jsonify({"ok": True, "messages": msgs})


@app.route("/api/tts", methods=["POST"])
def tts():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "text required"}), 400

    if len(text) > 4000:
        text = text[:4000]

    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        return jsonify({"ok": False, "error": "AZURE_SPEECH_KEY/REGION missing"}), 500

    try:
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        voice = os.getenv("AZURE_SPEECH_VOICE", "es-ES-AlvaroNeural")
        speech_config.speech_synthesis_voice_name = voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
        )

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            return Response(audio_data, mimetype="audio/mpeg", headers={"Cache-Control": "no-store"})

        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation = speechsdk.CancellationDetails(result)
            return jsonify({"ok": False, "error": cancellation.error_details or "canceled"}), 500

        return jsonify({"ok": False, "error": "tts_failed"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5100))
    app.run(host="0.0.0.0", port=port, debug=True)
