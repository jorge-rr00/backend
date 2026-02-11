import os
import re
import operator
import time
from typing import List, Optional, Dict, TypedDict, Annotated

# LangChain / LangGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
import pathlib
import subprocess

# Azure & Tools
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from PyPDF2 import PdfReader
import docx
import requests

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

_paddle_ocr = None


# ---------------------------
# Hidden document persistence
# ---------------------------
HIST_TAG_START = "<!--HISTORICAL_DOC_TEXT-->"
HIST_TAG_END = "<!--END_HISTORICAL_DOC_TEXT-->"
MAX_DOC_CHARS = 20000  # to keep requests efficient & cheap (Azure + tokens)


def strip_hidden_doc_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        re.escape(HIST_TAG_START) + r".*?" + re.escape(HIST_TAG_END),
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def extract_hidden_doc_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(
        re.escape(HIST_TAG_START) + r"(.*?)" + re.escape(HIST_TAG_END),
        text,
        flags=re.DOTALL,
    )
    return (m.group(1) if m else "").strip()


def _truncate_doc(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # keep the tail (often contains signatures/parties), but you can switch to head if you prefer
    return text[-max_chars:]


def _ocr_image_with_azure(path: str) -> str:
    endpoint = os.getenv("AZURE_VISION_ENDPOINT", "").strip()
    key = os.getenv("AZURE_VISION_KEY", "").strip()
    if not endpoint or not key:
        return ""

    url = endpoint.rstrip("/") + "/vision/v3.2/read/analyze?language=es"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream",
    }

    with open(path, "rb") as f:
        resp = requests.post(url, headers=headers, data=f)
    if resp.status_code not in (200, 202):
        return ""

    op_url = resp.headers.get("Operation-Location")
    if not op_url:
        return ""

    for _ in range(10):
        time.sleep(0.7)
        r = requests.get(op_url, headers={"Ocp-Apim-Subscription-Key": key})
        data = r.json()
        status = (data.get("status") or "").lower()
        if status == "succeeded":
            lines = []
            for page in data.get("analyzeResult", {}).get("readResults", []):
                for line in page.get("lines", []):
                    txt = line.get("text", "")
                    if txt:
                        lines.append(txt)
            return "\n".join(lines)
        if status == "failed":
            break

    return ""


def _ocr_image_with_paddle(path: str) -> str:
    global _paddle_ocr
    if PaddleOCR is None:
        return ""

    if _paddle_ocr is None:
        try:
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="es")
        except Exception:
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")

    result = _paddle_ocr.ocr(path, cls=True)
    lines = []
    for page in result or []:
        for item in page or []:
            text = item[1][0] if item and len(item) > 1 else ""
            if text:
                lines.append(text)
    return "\n".join(lines)


def _get_last_user_text(messages: List[BaseMessage]) -> str:
    # Robust: find last HumanMessage (avoid list index errors)
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _log_state(prefix: str, state: Dict):
    msg_n = len(state.get("messages", []) or [])
    doc_n = len(state.get("extracted_text", "") or "")
    files_n = len(state.get("file_paths", []) or [])
    dom = state.get("domain", "") or ""
    has_final = bool(state.get("final_response"))
    print(f"{prefix} messages={msg_n} files={files_n} extracted_chars={doc_n} domain={dom} final={has_final}")


# ---------------------------
# LangGraph State
# ---------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    file_paths: List[str]
    extracted_text: str
    domain: str
    specialist_analysis: str
    final_response: str
    voice_mode: bool


# ---------------------------
# Config (as you have it)
# ---------------------------
# agents_v3.py

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    if not v:
        raise ValueError(f"Missing environment variable: {name}")
    return v

AZURE_CONFIG = {
    "api_key": _env("AZURE_OPENAI_API_KEY"),
    "endpoint": _env("AZURE_OPENAI_ENDPOINT"),
    "deployment": _env("AZURE_OPENAI_DEPLOYMENT"),
    "search_endpoint": _env("AZURE_SEARCH_ENDPOINT"),
    "search_key": _env("AZURE_SEARCH_API_KEY"),
}



class LangGraphAssistant:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_CONFIG["deployment"],
            api_version="2024-05-01-preview",
            azure_endpoint=AZURE_CONFIG["endpoint"],
            api_key=AZURE_CONFIG["api_key"],
            temperature=1,  # required by your constraint
        )

        # RAG clients
        self.legal_search = SearchClient(
            AZURE_CONFIG["search_endpoint"],
            "multimodal-rag-1770652413829",
            AzureKeyCredential(AZURE_CONFIG["search_key"]),
        )
        self.financial_search = SearchClient(
            AZURE_CONFIG["search_endpoint"],
            "multimodal-rag-1770651232495",
            AzureKeyCredential(AZURE_CONFIG["search_key"]),
        )

        self.app = self._build_workflow()

    # ---------------------------
    # Node: Tool (file text extraction)
    # ---------------------------
    def tool_node(self, state: AgentState):
        _log_state("[Node: Tool]", state)

        # If no new files, keep previous extracted_text (from memory)
        if not state.get("file_paths"):
            return {"extracted_text": state.get("extracted_text", "") or ""}

        print("[Node: Tool] Processing new files...")
        text_dump = ""
        for p in state["file_paths"]:
            ext = p.split(".")[-1].lower()
            try:
                if ext == "pdf":
                    reader = PdfReader(p)
                    text_dump += "".join([(page.extract_text() or "") for page in reader.pages])
                elif ext in ("jpg", "jpeg", "png"):
                    if pytesseract and Image:
                        text_dump += pytesseract.image_to_string(Image.open(p))
                    else:
                        ocr_text = _ocr_image_with_paddle(p)
                        if not ocr_text:
                            ocr_text = _ocr_image_with_azure(p)
                        if ocr_text:
                            text_dump += ocr_text
                        else:
                            print("[Node: Tool] OCR unavailable for images. Configure AZURE_VISION_* or install tesseract/paddleocr.")
                elif ext == "docx":
                    d = docx.Document(p)
                    text_dump += "\n".join([para.text for para in d.paragraphs])
                else:
                    print(f"[Node: Tool] Unsupported extension or OCR libs missing: {ext}")
            except Exception as e:
                print(f"[Node: Tool] Error processing {p}: {e}")

        # Combine previous (memory) + new
        combined_text = (state.get("extracted_text", "") or "") + "\n" + (text_dump or "")
        combined_text = _truncate_doc(combined_text.strip(), MAX_DOC_CHARS)
        print(f"[Node: Tool] Extracted chars (after combine+truncate): {len(combined_text)}")
        return {"extracted_text": combined_text}

    # ---------------------------
    # Node: Orchestrator (answer from doc OR route)
    # ---------------------------
    def orchestrator_node(self, state: AgentState):
        _log_state("[Node: Orchestrator]", state)

        user_query = _get_last_user_text(state.get("messages", []))
        if not user_query:
            # Safety fallback; should not happen if server + OrchestratorAgent respond() are correct
            return {"final_response": "No he recibido la pregunta del usuario. Reintenta enviándola."}

        doc_context = _truncate_doc(state.get("extracted_text", "") or "", MAX_DOC_CHARS)

        sys_prompt = (
            "You are the Senior Orchestrator.\n"
            "Your top priority is to answer the user's question using the USER UPLOADED DOCUMENT content, if present.\n"
            "Rules:\n"
            "1) If the answer is explicitly contained in the extracted DOCUMENT text, answer directly and stop. Be precise.\n"
            "2) If the answer is NOT in the document and you need general legal or financial knowledge, respond with exactly "
            "'DOMAIN:LEGAL' or 'DOMAIN:FINANCIAL'.\n"
            "3) Do not say 'according to the document'. Provide the answer professionally.\n"
            "4) Respond in Spanish (Castellano).\n"
            "\nDOCUMENT CONTENT:\n"
            f"{doc_context}"
        )

        llm_messages = [SystemMessage(content=sys_prompt)] + (state.get("messages", []) or [])
        res = self.llm.invoke(llm_messages)
        upper = (res.content or "").upper().strip()

        # Route only if the model explicitly emits DOMAIN:...
        if "DOMAIN:" in upper:
            domain = "legal" if "LEGAL" in upper else "financial"
            print(f"[Node: Orchestrator] Routing -> {domain}")
            return {"domain": domain, "final_response": ""}

        print("[Node: Orchestrator] Answered directly.")
        return {"final_response": (res.content or "").strip()}

    # ---------------------------
    # Node: Specialist (RAG + doc)
    # ---------------------------
    def specialist_node(self, state: AgentState):
        _log_state("[Node: Specialist]", state)

        domain = (state.get("domain") or "").strip().lower() or "legal"
        query = _get_last_user_text(state.get("messages", []))

        client = self.legal_search if domain == "legal" else self.financial_search

        print(f"[Node: Specialist] RAG lookup domain={domain} top=3")
        try:
            results = client.search(search_text=query, top=3)
            rag_data = "\n".join([r.get("content", "") for r in results if r.get("content")])
        except Exception as e:
            print(f"[Node: Specialist] RAG error: {e}")
            rag_data = ""

        doc_text = _truncate_doc(state.get("extracted_text", "") or "", MAX_DOC_CHARS)

        prompt = (
            f"You are a domain specialist ({domain}). Use BOTH the RAG context and the USER DOCUMENT to answer.\n"
            "Respond in Spanish (Castellano). Be correct and concise.\n"
            "\nUSER DOCUMENT:\n"
            f"{doc_text}\n"
            "\nRAG CONTEXT:\n"
            f"{rag_data}\n"
        )

        llm_messages = [SystemMessage(content=prompt)] + (state.get("messages", []) or [])
        res = self.llm.invoke(llm_messages)
        return {"specialist_analysis": (res.content or "").strip()}

    # ---------------------------
    # Node: Final redactor (only if specialist was used)
    # ---------------------------
    def final_redactor_node(self, state: AgentState):
        _log_state("[Node: Final Redactor]", state)

        if state.get("final_response"):
            return {}

        analysis = (state.get("specialist_analysis") or "").strip()
        if not analysis:
            return {"final_response": "No tengo suficiente información para responder con seguridad."}

        sys_msg = SystemMessage(
            content=(
                "You are the Orchestrator.\n"
                "Rewrite the specialist analysis into natural Spanish (Castellano), user-facing.\n"
                "Be brief, direct, and do not include internal reasoning.\n"
            )
        )
        res = self.llm.invoke([sys_msg, HumanMessage(content=analysis)])
        return {"final_response": (res.content or "").strip()}

    # ---------------------------
    # Graph builder + Mermaid
    # ---------------------------
    def _build_workflow(self):
        builder = StateGraph(AgentState)
        builder.add_node("tool", self.tool_node)
        builder.add_node("orchestrator", self.orchestrator_node)
        builder.add_node("specialist", self.specialist_node)
        builder.add_node("final_redactor", self.final_redactor_node)

        builder.set_entry_point("tool")
        builder.add_edge("tool", "orchestrator")
        builder.add_conditional_edges(
            "orchestrator",
            lambda x: "end" if (x.get("final_response") or "").strip() else "specialist",
            {"end": END, "specialist": "specialist"},
        )
        builder.add_edge("specialist", "final_redactor")
        builder.add_edge("final_redactor", END)

        graph = builder.compile()

        return graph

# ---------------------------
# Public Agents
# ---------------------------
class OrchestratorAgent:
    def __init__(self, client=None):
        # client kept for compatibility with server.py
        self.engine = LangGraphAssistant()

    def respond(self, user_query: str, filepaths: List[str], session_history: List[dict] = None) -> str:
        messages: List[BaseMessage] = []
        historical_doc_text = ""

        # 1) Rebuild messages and retrieve doc memory from DB history
        if session_history:
            for m in session_history:
                raw_content = (m.get("content") or "")
                role = (m.get("role") or "").strip().lower()

                # extract doc memory if present in any assistant message
                extracted = extract_hidden_doc_text(raw_content)
                if extracted:
                    historical_doc_text = extracted

                # strip tags before giving to LLM
                clean_content = strip_hidden_doc_tags(raw_content)
                if not clean_content:
                    continue

                if role == "user":
                    messages.append(HumanMessage(content=clean_content))
                elif role == "assistant":
                    messages.append(AIMessage(content=clean_content))
                elif role == "system":
                    messages.append(SystemMessage(content=clean_content))
                else:
                    # fallback
                    messages.append(AIMessage(content=clean_content))

        # 2) IMPORTANT: append current user message (prevents empty state["messages"])
        messages.append(HumanMessage(content=(user_query or "").strip()))

        # 3) Run graph
        result = self.engine.app.invoke(
            {
                "messages": messages,
                "file_paths": filepaths or [],
                "extracted_text": _truncate_doc(historical_doc_text, MAX_DOC_CHARS),
                "domain": "",
                "specialist_analysis": "",
                "final_response": "",
                "voice_mode": False,
            }
        )

        final_ans = (result.get("final_response") or "").strip()

        # 4) Persist document text invisibly for next turns
        extracted_text = _truncate_doc((result.get("extracted_text") or "").strip(), MAX_DOC_CHARS)
        if extracted_text:
            final_ans = final_ans + "\n" + HIST_TAG_START + extracted_text + HIST_TAG_END
        elif filepaths:
            final_ans = (
                final_ans
                + "\n\nNota: No pude extraer texto de los archivos adjuntos. "
                + "Si es una imagen, activa OCR (tesseract o AZURE_VISION_*), o sube PDF/DOCX."
            )

        return final_ans


class VoiceOrchestratorAgent(OrchestratorAgent):
    def respond(self, user_query: str, filepaths: List[str], session_history: List[dict] = None) -> str:
        # same flow; you can later add voice-specific shortening if desired
        return super().respond(user_query, filepaths, session_history)


class GuardrailAgent:
    def __init__(self, client=None):
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_CONFIG["deployment"],
            api_version="2024-05-01-preview",
            azure_endpoint=AZURE_CONFIG["endpoint"],
            api_key=AZURE_CONFIG["api_key"],
            temperature=1,
        )

    def validate(self, q, f):
        # If files are attached, allow (you already have file-based flow)
        if f:
            return True, None

        sys = (
            "You are a guardrail classifier.\n"
            "If the user request is related to legal or financial topics, respond ONLY with: ACCEPT\n"
            "Otherwise respond ONLY with: REJECT\n"
        )
        res = self.llm.invoke([SystemMessage(content=sys), HumanMessage(content=q or "")])
        ok = "ACCEPT" in (res.content or "").upper()
        return (ok, None if ok else "Por favor, cíñete a temas legales o financieros.")


class AzureOpenAIClient:
    # kept for compatibility with server.py
    def __init__(self, d=None):
        pass
