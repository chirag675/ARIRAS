import sys
import os
from pathlib import Path

# ── Ensure project root is always on the path (Windows + venv safe) ──────────
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

import streamlit as st
import json
import datetime
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

from core.vectorstore import build_vectorstore, load_vectorstore
from agents.rag_agent import ask_regulation
from agents.gap_detector import detect_gaps
from agents.policy_builder import generate_policy_guidance, build_excel
from langchain_community.document_loaders import PyPDFLoader
# PATCH 1 — new import
from core.edge_handler import detect_reg_conflicts

st.set_page_config(
    page_title="GovernIQ",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* =========================
   LIGHT THEME (DEFAULT)
========================= */
:root {
    --bg:      #F7F8FA;
    --surface: #FFFFFF;
    --surface2:#F0F4F8;
    --border:  #D1D5DB;
    --accent:  #4F7FFF;
    --accent2: #00E5A0;
    --danger:  #FF5C5C;
    --warn:    #FFB347;
    --text:    #1A1D23;
    --muted:   #6B7280;
    --mono:    'DM Mono', monospace;
    --sans:    'DM Sans', sans-serif;
    --serif:   'DM Serif Display', serif;
}

/* =========================
   DARK THEME
========================= */
@media (prefers-color-scheme: dark) {
    :root {
        --bg:      #0B0F17;
        --surface: #111827;
        --surface2:#0F172A;
        --border:  #1F2937;
        --text:    #E5E7EB;
        --muted:   #9CA3AF;

        --accent:  #4F7FFF;
        --accent2: #00E5A0;
        --danger:  #FF5C5C;
        --warn:    #FFB347;
    }
}
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3rem 4rem 3rem !important; max-width: 1200px; }

[data-theme="dark"] {
    --bg: #0B0F17;
    --surface:#111827;
    --surface2:#0F172A;
    --border:#1F2937;
    --text:#E5E7EB;
    --muted:#9CA3AF;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    transform: none !important;
    visibility: visible !important;
    width: 260px !important;
    min-width: 260px !important;
    position: relative !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important; margin-left: 0 !important;
    width: 260px !important; min-width: 260px !important;
}
[data-testid="stSidebarCollapseButton"] { display: none !important; }
section[data-testid="stSidebarContent"] { visibility: visible !important; }

.governiq-header {
    display: flex; align-items: center; gap: 20px;
    padding: 28px 32px;
    background: linear-gradient(135deg, #E3EAFD 0%, #F0F4F8 100%);
    border: 1px solid var(--border); border-radius: 16px; margin-bottom: 20px;
}
.governiq-logo {
    font-family: var(--serif); font-size: 40px; letter-spacing: -1px;
    background: linear-gradient(135deg, #4F7FFF, #00E5A0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.governiq-tagline { font-size: 13px; color: var(--muted); font-weight: 300; letter-spacing: 0.6px; margin-top: 4px; }

.upload-zone { background: #FFFFFF; border: 1px solid #D1D5DB; border-radius: 14px; padding: 24px 28px; margin-bottom: 20px; }
.upload-zone-ready { background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 14px; padding: 24px 28px; margin-bottom: 20px; }

.next-banner {
    display: flex; align-items: flex-start; gap: 12px;
    background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #4F7FFF;
    border-radius: 10px; padding: 16px 20px; margin-top: 24px; font-size: 13px; color: #1E40AF; line-height: 1.6;
}
.next-banner-success {
    background: #F0FDF4; border: 1px solid #86EFAC; border-left: 4px solid #00A870;
    border-radius: 10px; padding: 14px 20px; margin-bottom: 16px; font-size: 13px; color: #065F46; line-height: 1.6;
}

.badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 500; font-family: var(--mono); letter-spacing: 0.3px; }
.badge-blue  { background: rgba(79,127,255,0.15); color: #4F7FFF; border: 1px solid rgba(79,127,255,0.3); }
.badge-green { background: rgba(0,229,160,0.12);  color: #00A870; border: 1px solid rgba(0,229,160,0.25); }
.badge-warn  { background: rgba(255,179,71,0.12); color: #CC8800; border: 1px solid rgba(255,179,71,0.25); }

[data-testid="stTabs"] [role="tablist"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 5px; gap: 4px; }
[data-testid="stTabs"] button[role="tab"] { font-family: var(--sans); font-size: 13px; font-weight: 500; color: var(--muted); border-radius: 8px; padding: 10px 24px; border: none; transition: all 0.2s; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: var(--accent) !important; color: white !important; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: 28px; }

.card-title { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 16px; }
.section-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted); margin-bottom: 8px; margin-top: 4px; }

.result-box {
    background: var(--surface2); border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 10px; padding: 20px 22px; font-family: var(--mono); font-size: 13px;
    line-height: 1.8; white-space: pre-wrap; margin-top: 16px; color: var(--text);
}
.result-box.success { border-left-color: var(--accent2); }
.result-box.danger  { border-left-color: var(--danger); }
.result-box.warn    { border-left-color: var(--warn); }

.audit-entry { display: flex; gap: 14px; align-items: flex-start; padding: 14px 0; border-bottom: 1px solid var(--border); font-size: 13px; line-height: 1.5; }
.audit-time { font-family: var(--mono); font-size: 11px; color: var(--muted); min-width: 80px; padding-top: 2px; }
.audit-text { color: var(--text); line-height: 1.6; }

.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
.metric-tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 22px 20px; text-align: center; }
.metric-value { font-family: var(--serif); font-size: 36px; line-height: 1.1; margin-bottom: 6px; }
.metric-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; }

.stProgress > div > div { background: var(--accent) !important; }
[data-testid="stFileUploader"] { background: var(--surface2) !important; border: 1px dashed var(--border) !important; border-radius: 10px !important; }
.stButton > button { font-family: var(--sans); font-weight: 500; font-size: 14px; background: var(--accent); color: white; border: none; border-radius: 8px; padding: 11px 26px; transition: all 0.2s; }
.stButton > button:hover { background: #6B95FF; transform: translateY(-1px); }
.stTextArea textarea, .stTextInput input { background: var(--surface2) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--sans) !important; border-radius: 8px !important; font-size: 14px !important; line-height: 1.6 !important; }
.stTextArea textarea { padding: 12px 14px !important; }

.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
.dot-green { background: #00A870; box-shadow: 0 0 6px #00E5A0; }
.dot-red   { background: var(--danger); box-shadow: 0 0 6px var(--danger); }
.dot-grey  { background: var(--muted); }

hr { border-color: var(--border) !important; margin: 24px 0 !important; }
[data-testid="stSelectbox"] > div > div { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 10px !important; margin-bottom: 12px !important; }
[data-testid="stExpander"] summary { padding: 14px 18px !important; font-weight: 500 !important; font-size: 14px !important; }
[data-testid="stExpander"] > div > div { padding: 4px 18px 18px 18px !important; }
.stAlert { border-radius: 10px !important; padding: 14px 18px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
document.documentElement.setAttribute('data-theme', 'dark');
</script>
""", unsafe_allow_html=True)

# ── Persistent storage ────────────────────────────────────────────────────────
_PERSIST_FILE = os.path.join(_project_root, "data", "GovernIQ_state.json")

def _load_persisted():
    try:
        os.makedirs(os.path.dirname(_PERSIST_FILE), exist_ok=True)
        if os.path.exists(_PERSIST_FILE):
            with open(_PERSIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_persisted():
    try:
        os.makedirs(os.path.dirname(_PERSIST_FILE), exist_ok=True)
        data = {
            "audit_log":   st.session_state.audit_log,
            "gap_history": st.session_state.gap_history,
        }
        with open(_PERSIST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# ── Wipe vectorstore once per session ─────────────────────────────────────────
import shutil as _shutil
if "vectorstore_wiped" not in st.session_state:
    _chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    if os.path.exists(_chroma_dir):
        try:
            _shutil.rmtree(_chroma_dir, ignore_errors=True)
        except Exception:
            pass
    st.session_state.vectorstore_wiped = True

_persisted = _load_persisted()

if "audit_log"              not in st.session_state: st.session_state.audit_log           = _persisted.get("audit_log", [])
if "vectorstore_ready"      not in st.session_state: st.session_state.vectorstore_ready   = False
if "uploaded_regs"          not in st.session_state: st.session_state.uploaded_regs       = []
if "gap_results"            not in st.session_state: st.session_state.gap_results         = None
if "gap_history"            not in st.session_state: st.session_state.gap_history         = _persisted.get("gap_history", [])
if "gap_policy_name"        not in st.session_state: st.session_state.gap_policy_name     = ""
if "gap_reg_name"           not in st.session_state: st.session_state.gap_reg_name        = ""
if "rag_answer"             not in st.session_state: st.session_state.rag_answer          = None
if "policy_guidance"        not in st.session_state: st.session_state.policy_guidance     = None
if "policy_excel_bytes"     not in st.session_state: st.session_state.policy_excel_bytes  = None
if "pb_step"                not in st.session_state: st.session_state.pb_step             = 1
if "pb_company_name"        not in st.session_state: st.session_state.pb_company_name     = ""
if "pb_company_description" not in st.session_state: st.session_state.pb_company_description = ""
if "pb_information_flows"   not in st.session_state: st.session_state.pb_information_flows   = ""
if "pb_key_stakeholders"    not in st.session_state: st.session_state.pb_key_stakeholders    = ""
if "_last_profile_file"     not in st.session_state: st.session_state._last_profile_file     = ""


# PATCH 2 — UI helper functions (inserted before log_audit)

def _confidence_badge(edge: dict):
    """Renders a confidence badge with score + warning banners from _edge dict."""
    if not edge:
        return
    conf     = edge.get("confidence", "")
    score    = edge.get("confidence_score", 0)
    warnings = edge.get("warnings", [])
    faith    = edge.get("faithfulness", {})
    grounding= edge.get("grounding", {})

    COLORS = {
        "HIGH":         ("#00A870", "#D1FAE5", "#065F46"),
        "MEDIUM":       ("#CC8800", "#FEF3C7", "#92400E"),
        "LOW":          ("#CC3333", "#FEE2E2", "#7F1D1D"),
        "INSUFFICIENT": ("#6B7280", "#F3F4F6", "#374151"),
    }
    accent, bg, text = COLORS.get(conf, COLORS["MEDIUM"])

    # Badge row
    verdict  = faith.get("verdict", "")
    sim_str  = f"{grounding.get('max_sim', 0):.2f}"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;"
        f"background:{bg};border:1px solid {accent};border-radius:8px;"
        f"padding:10px 16px;margin:12px 0;flex-wrap:wrap;'>"
        f"<span style='font-size:12px;font-weight:700;color:{accent};"
        f"font-family:var(--mono);letter-spacing:0.5px;'>CONFIDENCE: {conf}</span>"
        f"<span style='font-size:12px;color:{text};font-family:var(--mono);'>{score}/100</span>"
        f"<span style='font-size:11px;color:{text};'>·  Semantic similarity: {sim_str}</span>"
        f"<span style='font-size:11px;color:{text};'>·  Faithfulness: {verdict}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Unsupported claims — most visible signal of hallucination
    claims = faith.get("unsupported_claims", [])
    if claims:
        st.markdown(
            f"<div style='background:#FEE2E2;border:1px solid #FECACA;"
            f"border-left:3px solid #CC3333;border-radius:8px;"
            f"padding:10px 14px;font-size:12px;color:#7F1D1D;"
            f"margin-bottom:8px;line-height:1.7;'>"
            f"<b>🚫 Unsupported claims detected by faithfulness checker:</b><br>"
            + "<br>".join(f"• {c}" for c in claims[:4]) +
            f"</div>",
            unsafe_allow_html=True,
        )

    # Generic warnings
    for w in warnings:
        st.markdown(
            f"<div style='background:#FEF9EE;border:1px solid #FDE68A;"
            f"border-left:3px solid #CC8800;border-radius:8px;"
            f"padding:10px 14px;font-size:12px;color:#92400E;"
            f"margin-bottom:6px;line-height:1.6;'>⚠️ {w}</div>",
            unsafe_allow_html=True,
        )


def _preflight_banners(preflight: dict):
    """Renders pre-flight error/warning banners + document stats."""
    if not preflight:
        return
    for err in preflight.get("errors", []):
        st.markdown(
            f"<div style='background:#FEE2E2;border:1px solid #FECACA;"
            f"border-left:4px solid #CC3333;border-radius:8px;"
            f"padding:12px 16px;font-size:12px;color:#7F1D1D;"
            f"margin-bottom:8px;line-height:1.6;'>🚫 <b>Blocked:</b> {err}</div>",
            unsafe_allow_html=True,
        )
    for w in preflight.get("warnings", []):
        st.markdown(
            f"<div style='background:#FEF9EE;border:1px solid #FDE68A;"
            f"border-left:4px solid #CC8800;border-radius:8px;"
            f"padding:12px 16px;font-size:12px;color:#92400E;"
            f"margin-bottom:8px;line-height:1.6;'>⚠️ {w}</div>",
            unsafe_allow_html=True,
        )
    rel   = preflight.get("compliance_relevance")
    div   = preflight.get("reg_chunk_diversity")
    found = preflight.get("reg_chunks_found")
    if rel is not None:
        st.markdown(
            f"<div style='font-size:11px;color:#9CA3AF;margin-bottom:10px;'>"
            f"Policy compliance relevance: <b>{rel}</b> / 1.00  ·  "
            f"Regulation chunks: <b>{found}</b>  ·  "
            f"Chunk diversity: <b>{div}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _conflict_panel(uploaded_regs: list):
    """Renders a cross-regulation conflict panel when 2+ regs are loaded."""
    data = detect_reg_conflicts(uploaded_regs)
    if not data["conflicts_found"]:
        return
    st.markdown(
        f"<div style='background:#FEF3C7;border:1px solid #FDE68A;"
        f"border-left:4px solid #CC8800;border-radius:10px;"
        f"padding:14px 18px;margin-bottom:20px;'>"
        f"<div style='font-size:12px;font-weight:700;color:#92400E;"
        f"text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;'>"
        f"⚡ Cross-Regulation Conflicts Detected</div>"
        f"<div style='font-size:12px;color:#78350F;'>{data['summary']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    for c in data["conflicts"]:
        color = "#CC3333" if c["severity"] == "high" else "#CC8800"
        regs  = ", ".join(c["regulations_involved"])
        sims  = ""
        if c.get("sim_score_a") is not None:
            sims = (f"  ·  Semantic match scores: "
                    f"{c['sim_score_a']:.2f} / {c['sim_score_b']:.2f}")
        with st.expander(f"⚡  {c['topic']}  [{c['severity'].upper()}]", expanded=True):
            st.markdown(
                f"<div style='font-size:12px;color:{color};"
                f"font-weight:600;margin-bottom:6px;'>"
                f"Regulations: {regs}{sims}</div>"
                f"<div style='font-size:13px;color:#1A1D23;"
                f"line-height:1.7;'>{c['description']}</div>",
                unsafe_allow_html=True,
            )


def log_audit(action: str, detail: str = ""):
    st.session_state.audit_log.append({
        "time":   datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S"),
        "action": action,
        "detail": detail,
    })
    _save_persisted()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 16px 0;'>
        <div style='font-family:"DM Serif Display",serif;font-size:24px;
                    background:linear-gradient(135deg,#4F7FFF,#00E5A0);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            GovernIQ
        </div>
        <div style='font-size:11px;color:#6B7280;margin-top:2px;'>
            Multi-Agent Regulatory Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    sidebar_view = st.radio(
        "Navigation",
        ["Dashboard", "Regulatory Insights", "Activity Trail"],
        key="sidebar_view",
    )
    st.markdown("---")

    vs_ready = st.session_state.vectorstore_ready
    st.markdown("<div class='section-label'>System status</div>", unsafe_allow_html=True)
    st.markdown(
        f"<span class='status-dot {'dot-green' if vs_ready else 'dot-red'}'></span>"
        f"<span style='font-size:13px;'>{'Vector store ready' if vs_ready else 'No Compliance Data Loaded'}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span class='status-dot dot-grey'></span>"
        f"<span style='font-size:13px;'>{len(st.session_state.uploaded_regs)} regulation(s) loaded</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("<div class='section-label'>Session stats</div>", unsafe_allow_html=True)
    st.markdown(
        f"<span style='font-size:13px;color:#6B7280;'>Audit entries: "
        f"<b style='color:#1A1D23;'>{len(st.session_state.audit_log)}</b></span>",
        unsafe_allow_html=True,
    )
    if st.session_state.uploaded_regs:
        st.markdown("---")
        st.markdown("<div class='section-label'>Active Loaded Regulatory Frameworks</div>", unsafe_allow_html=True)
        for r in st.session_state.uploaded_regs:
            st.markdown(f"<span class='badge badge-blue'>{r}</span><br>", unsafe_allow_html=True)
    st.markdown("---")
    


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='governiq-header'>
    <div>
        <div class='governiq-logo'>GovernIQ</div>
        <div class='governiq-tagline'>Multi-Agent Regulatory Intelligence Platform</div>
        <div style='font-size:13px;color:#1A1D23;margin-top:8px;font-weight:600;letter-spacing:0.3px;'>
            Built in India for global enterprises —
            GovernIQ transforms regulations into actionable intelligence using
            autonomous AI agents, semantic compliance reasoning,
            policy gap detection, and governance analytics.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW: Reporting / Dashboard
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.sidebar_view == "Regulatory Insights":

    st.markdown("<div class='card-title'>Reporting &amp; Compliance Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:13px;color:#6B7280;'>"
        "Real-time governance intelligence, compliance risk analytics, and regulatory gap visibility powered by GovernIQ's multi-agent AI engine."
        "</span>", unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    gap_results = st.session_state.gap_results or {}
    gap_history = st.session_state.gap_history
    gaps        = gap_results.get("gaps", [])
    met         = gap_results.get("met", [])
    comp_score  = gap_results.get("compliance_score", 0)
    gap_count   = len(gaps)
    met_count   = len(met)
    sev         = {"high": 0, "medium": 0, "low": 0}
    for g in gaps:
        sev[g.get("severity", "medium")] += 1

    obligation_labels = [g.get("obligation", f"Obligation {i+1}")[:40] for i, g in enumerate(gaps)]
    sev_values        = [{"high": 3, "medium": 2, "low": 1}[g.get("severity", "medium")] for g in gaps]
    score_color       = "#00A870" if comp_score >= 70 else "#CC8800" if comp_score >= 40 else "#CC3333"
    plotly_paper      = "#F7F8FA"
    plotly_bg         = "#FFFFFF"
    plotly_font       = dict(family="DM Sans, sans-serif", color="#1A1D23", size=13)
    plotly_grid       = "#E5E7EB"

    st.markdown(f"""
    <div class='metric-grid'>
        <div class='metric-tile'><div class='metric-value' style='color:{score_color};'>{comp_score}%</div><div class='metric-label'>Compliance score</div></div>
        <div class='metric-tile'><div class='metric-value' style='color:#CC3333;'>{gap_count}</div><div class='metric-label'> Total Gaps detected</div></div>
        <div class='metric-tile'><div class='metric-value' style='color:#00A870;'>{met_count}</div><div class='metric-label'>Obligations met</div></div>
        <div class='metric-tile'><div class='metric-value' style='color:#4F7FFF;'>{len(st.session_state.uploaded_regs)}</div><div class='metric-label'>Regulations loaded</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=comp_score,
            title={"text": "Overall compliance score", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": plotly_grid},
                "bar":  {"color": score_color, "thickness": 0.28},
                "bgcolor": plotly_bg, "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "#FEE2E2"},
                    {"range": [40, 70], "color": "#FEF3C7"},
                    {"range": [70,100], "color": "#D1FAE5"},
                ],
                "threshold": {"line": {"color": score_color, "width": 3}, "thickness": 0.8, "value": comp_score},
            },
            number={"suffix": "%", "font": {"size": 36, "color": score_color}},
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor=plotly_paper, font=plotly_font)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            names=["High", "Medium", "Low"],
            values=[sev["high"], sev["medium"], sev["low"]],
            color=["High", "Medium", "Low"],
            color_discrete_map={"High": "#FF5C5C", "Medium": "#FFB347", "Low": "#00E5A0"},
            title="Gap severity breakdown", hole=0.45,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color=plotly_bg, width=2)))
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor=plotly_paper, plot_bgcolor=plotly_bg, font=plotly_font, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if obligation_labels:
            sev_color_map = {3: "#FF5C5C", 2: "#FFB347", 1: "#00E5A0"}
            fig_bar = go.Figure(go.Bar(
                x=sev_values, y=obligation_labels, orientation="h",
                marker=dict(color=[sev_color_map[v] for v in sev_values], line=dict(width=0)),
                text=["HIGH" if v == 3 else "MED" if v == 2 else "LOW" for v in sev_values],
                textposition="inside", insidetextanchor="middle", textfont=dict(color="white", size=11),
            ))
            fig_bar.update_layout(
                title="Top violated obligations", height=300, margin=dict(l=10, r=20, t=50, b=20),
                paper_bgcolor=plotly_paper, plot_bgcolor=plotly_bg, font=plotly_font,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 3.5]),
                yaxis=dict(showgrid=False, tickfont=dict(size=11), automargin=True),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.markdown("""
            <div style='background:#F0F4F8;border:1px dashed #D1D5DB;border-radius:10px;
                        padding:60px 20px;text-align:center;color:#6B7280;font-size:13px;'>
                Run a gap analysis to see obligation breakdown.
            </div>""", unsafe_allow_html=True)

    with col4:
        hist_labels = [h["label"] for h in gap_history] if gap_history else ["Run 1","Run 2","Run 3","Run 4","Run 5"]
        hist_scores = [h["score"] for h in gap_history] if gap_history else [0,0,0,0,0]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=hist_labels, y=hist_scores, mode="lines+markers",
            line=dict(color="#4F7FFF", width=2.5, shape="spline"),
            marker=dict(color="#4F7FFF", size=8, line=dict(color="white", width=2)),
            fill="tozeroy", fillcolor="rgba(79,127,255,0.08)",
        ))
        fig_line.update_layout(
            title="Compliance score over time", height=300, margin=dict(l=10, r=20, t=50, b=20),
            paper_bgcolor=plotly_paper, plot_bgcolor=plotly_bg, font=plotly_font,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor=plotly_grid, zeroline=False, range=[0,105], ticksuffix="%"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<div class='section-label' style='margin-top:8px;'>Regulation coverage summary</div>", unsafe_allow_html=True)
    regs = st.session_state.uploaded_regs or []
    if regs and gap_results:
        table_rows = ""
        for reg in regs:
            color = "#00A870" if comp_score >= 70 else "#CC8800" if comp_score >= 40 else "#CC3333"
            table_rows += f"""
            <tr style='border-bottom:1px solid #E5E7EB;'>
                <td style='padding:10px 12px;font-size:13px;font-weight:500;'>{reg}</td>
                <td style='padding:10px 12px;font-size:13px;'>
                    <div style='display:flex;align-items:center;gap:10px;'>
                        <div style='flex:1;background:#E5E7EB;border-radius:4px;height:8px;'>
                            <div style='width:{comp_score}%;background:{color};border-radius:4px;height:8px;'></div>
                        </div>
                        <span style='color:{color};font-weight:600;min-width:36px;'>{comp_score}%</span>
                    </div>
                </td>
                <td style='padding:10px 12px;font-size:13px;color:#CC3333;text-align:center;'>{gap_count}</td>
                <td style='padding:10px 12px;font-size:13px;color:#00A870;text-align:center;'>{met_count}</td>
            </tr>"""
        st.markdown(f"""
        <div style='background:#FFFFFF;border:1px solid #D1D5DB;border-radius:10px;overflow:hidden;'>
            <table style='width:100%;border-collapse:collapse;'>
                <thead><tr style='background:#F0F4F8;border-bottom:1px solid #D1D5DB;'>
                    <th style='padding:10px 12px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#6B7280;text-align:left;'>Regulation</th>
                    <th style='padding:10px 12px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#6B7280;text-align:left;'>Coverage</th>
                    <th style='padding:10px 12px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#6B7280;text-align:center;'>Gaps</th>
                    <th style='padding:10px 12px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#6B7280;text-align:center;'>Met</th>
                </tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#F0F4F8;border:1px dashed #D1D5DB;border-radius:10px;
                    padding:30px 20px;text-align:center;color:#6B7280;font-size:13px;'>
            Load a regulation and run gap analysis to populate this table.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "compliance_score": comp_score,
        "gaps_detected": gap_count,
        "obligations_met": met_count,
        "regulations": regs,
        "gap_details": gaps,
        "audit_log": st.session_state.audit_log,
    }
    st.download_button(
        label="⬇  Export full compliance report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="GovernIQ_compliance_report.json",
        mime="application/json",
    )


# ═════════════════════════════════════════════════════════════════════════════
# VIEW: Audit Trail
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.sidebar_view == "Activity Trail":

    st.markdown("<div class='card-title'>Full Audit Trail</div>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:13px;color:#6B7280;'>"
        "Every GovernIQ action logged with timestamp and decision rationale — "
        "exportable for regulatory traceability."
        "</span>", unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑  Clear log", key="btn_clear"):
        st.session_state.audit_log = []
        st.session_state.gap_history = []
        st.session_state.gap_results = None
        _save_persisted()
        st.rerun()

    if st.session_state.audit_log:
        agent_runs   = sum(1 for e in st.session_state.audit_log if any(k in e["action"] for k in ["Gap","Breach","RAG","query"]))
        docs_indexed = sum(1 for e in st.session_state.audit_log if "indexed" in e["action"].lower())

        st.markdown(f"""
        <div class='metric-grid'>
            <div class='metric-tile'><div class='metric-value' style='color:#4F7FFF;'>{len(st.session_state.audit_log)}</div><div class='metric-label'>Total entries</div></div>
            <div class='metric-tile'><div class='metric-value' style='color:#00A870;'>{agent_runs}</div><div class='metric-label'>Agent runs</div></div>
            <div class='metric-tile'><div class='metric-value' style='color:#CC8800;'>{docs_indexed}</div><div class='metric-label'>Docs indexed</div></div>
            <div class='metric-tile'><div class='metric-value' style='color:#CC3333;'>0</div><div class='metric-label'>Errors</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='background:#FFFFFF;border:1px solid #D1D5DB;border-radius:12px;padding:16px 20px;'>", unsafe_allow_html=True)
        for entry in reversed(st.session_state.audit_log):
            st.markdown(f"""
            <div class='audit-entry'>
                <span class='audit-time'>{entry['time']}</span>
                <span class='status-dot dot-green'></span>
                <span class='audit-text'>
                    <b>{entry['action']}</b>
                    {"&nbsp;&mdash;&nbsp;" + entry['detail'] if entry['detail'] else ""}
                </span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            label="⬇  Export audit log (JSON)",
            data=json.dumps(st.session_state.audit_log, indent=2),
            file_name="GovernIQ_audit_trail.json",
            mime="application/json",
        )
    else:
        st.markdown("""
        <div style='background:#F0F4F8;border:1px dashed #D1D5DB;border-radius:10px;
                    padding:60px 20px;text-align:center;color:#6B7280;font-size:13px;'>
            No actions logged yet. Upload a regulation in the Main App to start tracking.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW: Main App
# ═════════════════════════════════════════════════════════════════════════════
else:

    vs_ready = st.session_state.vectorstore_ready

    # ── How it works — 3 steps ────────────────────────────────────────────────
    step1 = "<div style='flex:1;background:#F8FAFF;border:1px solid #BFDBFE;border-radius:10px 0 0 10px;padding:12px 16px;'><div style='font-size:18px;margin-bottom:4px;'>①</div><div style='font-size:12px;font-weight:600;color:#1E40AF;margin-bottom:3px;'>Upload your regulation</div><div style='font-size:11px;color:#6B7280;line-height:1.5;'>Indian or global - DPDP Act, SEBI, RBI, GDPR, SOX, HIPAA, Companies Act, or any law worldwide.</div></div>"
    step2 = "<div style='flex:1;background:#F8FFF8;border-top:1px solid #86EFAC;border-bottom:1px solid #86EFAC;border-left:none;border-right:none;padding:12px 16px;'><div style='font-size:18px;margin-bottom:4px;'>②</div><div style='font-size:12px;font-weight:600;color:#065F46;margin-bottom:3px;'>Index it in seconds</div><div style='font-size:11px;color:#6B7280;line-height:1.5;'>GovernIQ reads and embeds the entire document into a searchable knowledge base in less than 30 seconds.</div></div>"
    step3 = "<div style='flex:1;background:#FEFBF0;border:1px solid #FDE68A;border-radius:0 10px 10px 0;padding:12px 16px;'><div style='font-size:18px;margin-bottom:4px;'>③</div><div style='font-size:12px;font-weight:600;color:#92400E;margin-bottom:3px;'>Get instant compliance intelligence</div><div style='font-size:11px;color:#6B7280;line-height:1.5;'> - Ask questions. <div> - Build a regulation-specific policy from scratch. <div> - Detect gaps in your existing policies.</div></div>"

    st.markdown(
        "<div style='margin-bottom:16px;'>"
        "<div style='font-size:12px;font-weight:600;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>📂 Regulation Knowledge Base</div>"
        "<div style='display:flex;gap:0;align-items:stretch;'>"
        + step1 + step2 + step3 +
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Upload zone ───────────────────────────────────────────────────────────
    if not vs_ready:
        st.markdown(
            "<div style='font-size:13px;color:#6B7280;margin-bottom:12px;line-height:1.6;'>"
            "Upload regulatory frameworks, compliance policies, or governance documents to initialize GovernIQ’s intelligence engine."
            "</div>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload regulation PDF", type=["pdf"], key="reg_upload",
            label_visibility="collapsed", accept_multiple_files=False,
        )
        if uploaded_file:
            if st.button("⚙️  Process & Index", key="btn_index"):
                with st.spinner("Indexing regulation..."):
                    try:
                        os.makedirs("data/uploads", exist_ok=True)
                        file_path = f"data/uploads/{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.read())
                        build_vectorstore(file_path)
                        st.session_state.vectorstore_ready = True
                        if uploaded_file.name not in st.session_state.uploaded_regs:
                            st.session_state.uploaded_regs.append(uploaded_file.name)
                        log_audit("Regulation indexed", uploaded_file.name)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.markdown(
            "<div style='font-size:12px;font-weight:600;color:#6B7280;text-transform:uppercase;"
            "letter-spacing:1px;margin-bottom:10px;'>📂 Manage Regulations</div>",
            unsafe_allow_html=True,
        )
        for reg in list(st.session_state.uploaded_regs):
            col_name, col_btn = st.columns([5, 1])
            with col_name:
                st.markdown(
                    f"<div style='background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;"
                    f"padding:10px 16px;font-size:13px;color:#065F46;margin-bottom:6px;'>"
                    f"✅ <b>{reg}</b> &nbsp;·&nbsp;<span style='font-size:12px;color:#6B7280;'>Indexed and ready</span></div>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("✕ Remove", key=f"remove_{reg}"):
                    st.session_state.uploaded_regs.remove(reg)
                    if not st.session_state.uploaded_regs:
                        import shutil
                        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
                        if os.path.exists(chroma_dir): shutil.rmtree(chroma_dir)
                        st.session_state.vectorstore_ready = False
                        log_audit("Knowledge base cleared", reg)
                    else:
                        import shutil
                        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
                        if os.path.exists(chroma_dir): shutil.rmtree(chroma_dir)
                        rebuilt = 0
                        for remaining in st.session_state.uploaded_regs:
                            fpath = f"data/uploads/{remaining}"
                            if os.path.exists(fpath):
                                build_vectorstore(fpath)
                                rebuilt += 1
                        if rebuilt == 0: st.session_state.vectorstore_ready = False
                        log_audit("Regulation removed — knowledge base rebuilt", reg)
                    _save_persisted()
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("➕  Upload another regulation", expanded=False):
            st.markdown("<div style='font-size:12px;color:#6B7280;margin-bottom:8px;'>Add another regulation PDF to the knowledge base.</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload regulation PDF", type=["pdf"], key="reg_upload_extra", label_visibility="collapsed")
            if uploaded_file:
                if st.button("⚙️ Process & Index", key="btn_index_extra"):
                    with st.spinner("Indexing regulation..."):
                        try:
                            os.makedirs("data/uploads", exist_ok=True)
                            file_path = f"data/uploads/{uploaded_file.name}"
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.read())
                            build_vectorstore(file_path)
                            if uploaded_file.name not in st.session_state.uploaded_regs:
                                st.session_state.uploaded_regs.append(uploaded_file.name)
                            log_audit("Regulation indexed", uploaded_file.name)
                            _save_persisted()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📄  Regulation Q&A", "📝  Policy Builder", "🔍  Gap Detector"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Regulation Q&A
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("<div class='card-title'>Regulation Q&amp;A</div>", unsafe_allow_html=True)
        st.markdown(
            "<span style='font-size:13px;color:#9CA3AF;line-height:1.9;'>"
            "Interact directly with GovernIQ’s regulatory intelligence engine. "
            "Ask questions across uploaded laws, circulars, frameworks, and compliance documents "
            "to receive AI-generated answers with clause-level traceability, semantic grounding, "
            "and governance context."
            "<br><br>"
            "Run unlimited regulatory queries in real time — from obligations and penalties "
            "to reporting requirements, consent mandates, and operational risk exposure."
            "</span>", unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if not vs_ready:
            st.markdown("""
            <div style='background:#FEF9EE;border:1px solid #FDE68A;border-radius:10px;
                        padding:18px 22px;font-size:13px;color:#92400E;'>
                ⚠️ Upload and index a regulation PDF in the panel above to start asking questions.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='section-label'>Your question</div>", unsafe_allow_html=True)
            question = st.text_area(
                "Ask question",
                placeholder="e.g. What are the reporting obligations?\ne.g. What is the deadline for breach notification?\ne.g. What penalties apply for non-compliance?",
                height=130, key="reg_question", label_visibility="collapsed",
            )

            col_btn, col_hint = st.columns([1, 3])
            with col_btn:
                get_answer = st.button("🔎  Get Answer", key="btn_rag", use_container_width=True)
            with col_hint:
                st.markdown(
                    "<div style='font-size:12px;color:#9CA3AF;padding-top:10px;'>"
                    "💡 To ask a new question, clear the box above and type again.</div>",
                    unsafe_allow_html=True,
                )

            if get_answer:
                if not question.strip():
                    st.warning("⚠️ Please enter a question.")
                else:
                    with st.spinner("Searching regulation..."):
                        try:
                            answer = ask_regulation(question)
                            st.session_state.rag_answer = answer
                            log_audit("RAG query", question[:60])
                        except Exception as e:
                            st.error(f"Agent error: {e}")

            if st.session_state.rag_answer:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Answer</div>", unsafe_allow_html=True)
                ans = st.session_state.rag_answer
                st.markdown(f"<div class='result-box success'>{ans.get('answer', 'No answer returned.')}</div>", unsafe_allow_html=True)
                # PATCH 3 — confidence badge after answer
                _confidence_badge(ans.get("_edge", {}))

                if ans.get("sources"):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<div class='section-label'>Source clauses</div>", unsafe_allow_html=True)
                    for src in ans["sources"]:
                        st.markdown(f"""
                        <div style='background:#F0F4F8;border:1px solid #D1D5DB;border-radius:8px;
                                    padding:12px 16px;margin-bottom:10px;font-size:12px;color:#6B7280;
                                    font-family:var(--mono);line-height:1.7;'>{src}</div>
                        """, unsafe_allow_html=True)

                st.markdown("""
                <div class='next-banner' style='margin-top:28px;'>
                    💡 <div><b>Ready to take the next step?</b><br>
                    → Go to <b>Policy Builder</b> to get guidance on what your company needs to include<br>
                    → Or go to <b>Gap Detector</b> to check your existing policy against this regulation
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Policy Builder
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("<div class='card-title'>📝 Policy Builder</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;
                    padding:14px 18px;font-size:13px;color:#1E40AF;line-height:1.7;margin-bottom:20px;'>
            🧠 <b>How this works</b> - Answer a few questions about your company.
            GovernIQ analyses your answers together with the loaded regulation and tells you
            exactly what your policy needs to include, with sample clauses ready to adapt.
            Works for <b>any regulation</b>.
        </div>
        """, unsafe_allow_html=True)

        if not vs_ready:
            st.markdown("""
            <div style='background:#FEF9EE;border:1px solid #FDE68A;border-radius:8px;
                        padding:12px 16px;font-size:13px;color:#92400E;margin-bottom:16px;'>
                💡 <b>Tip:</b> Upload a regulation in the panel above for targeted guidance.
                Without it, GovernIQ will use general compliance best practices.
            </div>""", unsafe_allow_html=True)

        step_now  = st.session_state.pb_step
        steps     = ["Company Profile", "Current State & Generate"]
        step_html = "<div style='display:flex;gap:8px;align-items:center;margin-bottom:24px;flex-wrap:wrap;'>"
        for i, s in enumerate(steps, start=1):
            if i < step_now:   bg, tc, label = "#00A870", "#FFFFFF", f"✓ {s}"
            elif i == step_now: bg, tc, label = "#4F7FFF",  "#FFFFFF", s
            else:               bg, tc, label = "#E5E7EB",  "#6B7280", s
            step_html += f"<div style='background:{bg};color:{tc};padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600;'>{label}</div>"
            if i < len(steps): step_html += "<div style='color:#D1D5DB;font-size:16px;'>→</div>"
        step_html += "</div>"
        st.markdown(step_html, unsafe_allow_html=True)

        # ── Section 1 — Company Profile ───────────────────────────────────────
        with st.expander("🏢  Section 1 — Company Profile", expanded=(step_now == 1)):

            st.markdown(
                "<div style='font-size:13px;color:#6B7280;margin-bottom:16px;line-height:1.6;'>"
                "Upload a company document <b>or</b> fill in the fields manually — or do both. "
                "If you upload a file, GovernIQ will pre-fill the fields below for you to review and edit freely."
                "</div>", unsafe_allow_html=True,
            )

            # ── File uploader ─────────────────────────────────────────────────
            st.markdown(
                "<div style='font-size:11px;font-weight:600;text-transform:uppercase;"
                "letter-spacing:1px;color:#4F7FFF;margin-bottom:6px;'>"
                "📄 Upload company document (recommended)</div>",
                unsafe_allow_html=True,
            )
            pb_profile_file = st.file_uploader(
                "Upload company profile document", type=["pdf", "txt"],
                key="pb_profile_file", label_visibility="collapsed",
            )

            # Auto-extract and pre-fill on new upload
            if pb_profile_file:
                if st.session_state.get("_last_profile_file") != pb_profile_file.name:
                    try:
                        os.makedirs("data/uploads", exist_ok=True)
                        profile_path = f"data/uploads/profile_{pb_profile_file.name}"
                        with open(profile_path, "wb") as f:
                            f.write(pb_profile_file.read())
                        if pb_profile_file.name.endswith(".pdf"):
                            loader = PyPDFLoader(profile_path)
                            pages  = loader.load()
                            profile_text = "\n".join([p.page_content for p in pages])[:5000]
                        else:
                            with open(profile_path, "r", encoding="utf-8", errors="ignore") as f:
                                profile_text = f.read()[:5000]
                        st.session_state.pb_company_description = profile_text
                        st.session_state.pb_information_flows   = "Extracted from uploaded document — please review and edit."
                        st.session_state.pb_key_stakeholders    = "Extracted from uploaded document — please review and edit."
                        st.session_state._last_profile_file     = pb_profile_file.name
                        log_audit("Company profile uploaded", pb_profile_file.name)
                    except Exception as e:
                        st.error(f"Error reading document: {e}")

                st.markdown("""
                <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:6px;
                            padding:8px 12px;font-size:12px;color:#1E40AF;margin:8px 0 4px 0;'>
                    ✅ Document uploaded — fields below have been pre-filled. Review and edit as needed.
                </div>
                """, unsafe_allow_html=True)

            # ── OR divider — always visible ───────────────────────────────────
            st.markdown("""
            <div style='display:flex;align-items:center;gap:12px;margin:20px 0 16px 0;'>
                <div style='flex:1;height:1px;background:#D1D5DB;'></div>
                <div style='font-size:13px;font-weight:700;color:#1A1D23;
                            padding:8px 24px;border:1px solid #D1D5DB;border-radius:20px;
                            background:#F7F8FA;letter-spacing:0.5px;'>
                    OR fill in manually
                </div>
                <div style='flex:1;height:1px;background:#D1D5DB;'></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Company name ──────────────────────────────────────────────────
            pb_company_name = st.text_input(
                "Company name (required)",
                placeholder="e.g. Finova Technologies Pvt Ltd",
                key="pb_company_name",
            )

            # ── Manual fields — always visible, always editable ───────────────
            pb_company_description = st.text_area(
                "What does your company primarily do?",
                placeholder=(
                    "e.g. We are a fintech startup providing short-term loans to salaried employees. "
                    "We operate a mobile app where users apply for credit, submit documents, "
                    "and receive disbursements directly to their bank accounts."
                ),
                height=110, key="pb_company_description",
            )
            pb_information_flows = st.text_area(
                "What kind of information flows through your business?",
                placeholder="e.g. We collect customer names, phone numbers, bank account details, salary slips, and Aadhaar numbers for KYC.",
                height=90, key="pb_information_flows",
            )
            pb_key_stakeholders = st.text_area(
                "Who are the key people or organisations involved?",
                placeholder="e.g. Our customers, banking partner (HDFC), cloud provider (AWS), credit bureaus (CIBIL), and our investors.",
                height=80, key="pb_key_stakeholders",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Next →", key="pb_next1"):
                if not st.session_state.get("pb_company_name", "").strip():
                    st.warning("⚠️ Please enter your company name.")
                elif not st.session_state.get("pb_company_description", "").strip():
                    st.warning("⚠️ Either upload a company document or fill in the description field.")
                else:
                    log_audit("Company profile saved", st.session_state.get("pb_company_name", ""))
                    st.session_state.pb_step = 2
                    st.rerun()

        # ── Section 2 — Current State + Generate ─────────────────────────────
        with st.expander("📋  Section 2 — Where You Are Today", expanded=(step_now == 2)):
            st.markdown(
                "<div style='font-size:13px;color:#6B7280;margin-bottom:16px;'>"
                "You can be as honest as you can. "
                "GovernIQ needs to understand your starting point to give useful guidance."
                "</div>", unsafe_allow_html=True,
            )
            pb_compliance_concerns = st.text_area(
                "What compliance areas are you most worried about or unsure of?",
                placeholder=(
                    "e.g. We are not sure if we need to register with any regulator. "
                    "We don't know if our consent process is legally valid. "
                    "We are worried about our data storage practices."
                ),
                height=110, key="pb_compliance_concerns",
            )
            pb_existing_policies = st.text_area(
                "What policies or documentation do you already have?",
                placeholder=(
                    "e.g. We have a basic privacy policy on our website but it hasn't been updated "
                    "in 2 years. We have employee NDAs but no formal data handling SOP."
                ),
                height=90, key="pb_existing_policies",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            col_b, col_g = st.columns([1, 4])
            with col_b:
                if st.button("← Back", key="pb_back3"):
                    st.session_state.pb_step = 1
                    st.rerun()
            with col_g:
                if st.button("🚀  Generate My Policy Guidance", key="btn_gen_policy", use_container_width=True):
                    if not pb_compliance_concerns.strip():
                        st.warning("⚠️ Please tell us your compliance concerns.")
                    else:
                        with st.spinner("GovernIQ is analysing your business and building guidance..."):
                            try:
                                reg_context = ""
                                if st.session_state.vectorstore_ready:
                                    vs        = load_vectorstore()
                                    retriever = vs.as_retriever(search_kwargs={"k": 6})
                                    docs      = retriever.invoke("obligations requirements compliance reporting disclosure penalties")
                                    reg_context = "\n\n".join([d.page_content for d in docs])[:3500]

                                guidance = generate_policy_guidance(
                                    company_name        = st.session_state.get("pb_company_name", ""),
                                    company_description = st.session_state.get("pb_company_description", ""),
                                    information_flows   = st.session_state.get("pb_information_flows", ""),
                                    key_stakeholders    = st.session_state.get("pb_key_stakeholders", ""),
                                    compliance_concerns = pb_compliance_concerns,
                                    existing_policies   = pb_existing_policies,
                                    regulation_context  = reg_context,
                                )
                                excel_bytes = build_excel(guidance)
                                st.session_state.policy_guidance    = guidance
                                st.session_state.policy_excel_bytes = excel_bytes
                                log_audit("Policy guidance generated", st.session_state.get("pb_company_name", ""))
                                st.success("✅ Guidance report ready!")
                            except Exception as e:
                                st.error(f"Error: {e}")

            if st.session_state.policy_guidance:
                g     = st.session_state.policy_guidance
                score = g.get("readiness_score", 0)
                sc    = "#00A870" if score >= 70 else "#CC8800" if score >= 40 else "#CC3333"

                st.markdown("<br>", unsafe_allow_html=True)
                col_score, col_dl = st.columns([1, 1])
                with col_score:
                    st.markdown(f"""
                    <div style='background:#F0F4F8;border:1px solid #D1D5DB;border-radius:10px;padding:20px;text-align:center;'>
                        <div style='font-family:"DM Serif Display",serif;font-size:52px;color:{sc};line-height:1;'>{score}</div>
                        <div style='font-size:12px;color:#6B7280;margin-top:4px;'>Readiness score / 100</div>
                        <div style='font-size:11px;color:#6B7280;margin-top:6px;font-style:italic;'>Regulation: {g.get("regulation_used","General")}</div>
                    </div>""", unsafe_allow_html=True)

                with col_dl:
                    st.markdown(
                        "<div style='padding-top:16px;'>"
                        "<div class='section-label'>Your report is ready</div>"
                        "<div style='font-size:13px;color:#6B7280;margin-bottom:12px;'>"
                        "Download the Excel — it has your full guidance with sample clauses, regulation references, and priority actions."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    if st.session_state.policy_excel_bytes:
                        fname = f"GovernIQ_Policy_Guidance_{st.session_state.get('pb_company_name','Company').replace(' ','_')}.xlsx"
                        st.download_button(
                            label="⬇  Download Policy Guidance (Excel)",
                            data=st.session_state.policy_excel_bytes,
                            file_name=fname,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                if g.get("summary"):
                    st.markdown(f"""
                    <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
                                padding:14px 16px;font-size:13px;color:#1E40AF;margin-top:16px;line-height:1.7;'>
                        📌 <b>GovernIQ Summary:</b> {g.get("summary")}
                    </div>""", unsafe_allow_html=True)

                sections = g.get("sections", [])
                if sections:
                    st.markdown(f"<div class='section-label' style='margin-top:20px;'>Guidance sections ({len(sections)})</div>", unsafe_allow_html=True)
                    for sec in sections:
                        with st.expander(f"📋  {sec.get('section_name','')}"):
                            st.markdown(f"<div style='font-size:12px;color:#6B7280;margin-bottom:8px;'><b>Regulation reference:</b> {sec.get('regulation_reference','—')}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:13px;margin-bottom:10px;'><b>What to include:</b><br>{sec.get('what_to_include','')}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='result-box success'><b>Sample clause:</b><br><br>{sec.get('sample_clause','')}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='background:#FEF3C7;border:1px solid #FDE68A;border-radius:6px;padding:10px 14px;font-size:12px;color:#92400E;margin-top:8px;'>⚠️ <b>Why it matters:</b> {sec.get('why_it_matters','')}</div>", unsafe_allow_html=True)

                actions = g.get("priority_actions", [])
                if actions:
                    st.markdown("<div class='section-label' style='margin-top:20px;'>Top priority actions</div>", unsafe_allow_html=True)
                    for i, action in enumerate(actions, start=1):
                        st.markdown(f"<div style='background:#D1FAE5;border:1px solid #6EE7B7;border-radius:6px;padding:10px 14px;font-size:13px;color:#065F46;margin-bottom:6px;'><b>{i}.</b> {action}</div>", unsafe_allow_html=True)

                risks = g.get("risk_areas", [])
                if risks:
                    st.markdown("<div class='section-label' style='margin-top:16px;'>Key risk areas</div>", unsafe_allow_html=True)
                    for risk in risks:
                        st.markdown(f"<div style='background:#FEE2E2;border:1px solid #FECACA;border-radius:6px;padding:10px 14px;font-size:13px;color:#991B1B;margin-bottom:6px;'>⚠️ {risk}</div>", unsafe_allow_html=True)

                st.markdown("""
                <div class='next-banner' style='margin-top:24px;'>
                    💡 <div><b>Have an existing policy document?</b><br>
                    → Go to <b>Gap Detector</b> to check it against the loaded regulation
                    and get a compliance score with detailed gap analysis.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Gap Detector
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("<div class='card-title'>🔍 Policy Gap Detector</div>", unsafe_allow_html=True)
        st.markdown(
            "<span style='font-size:13px;color:#6B7280;'>"
            "Upload your company's existing policy document. GovernIQ maps every obligation "
            "from the loaded regulation against your policy and flags exactly what's missing."
            "</span>", unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if not vs_ready:
            st.markdown("""
            <div style='background:#FEF9EE;border:1px solid #FDE68A;border-radius:10px;
                        padding:16px 20px;font-size:13px;color:#92400E;'>
                ⚠️ Upload and index a regulation PDF in the panel above first.
                Gap Detector needs a regulation to check your policy against.
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-label'>Upload your company policy</div>", unsafe_allow_html=True)
        policy_file = st.file_uploader("Policy PDF or TXT", type=["pdf", "txt"], key="policy_upload", label_visibility="collapsed")

        st.markdown("<div class='section-label' style='margin-top:16px;'>Regulation to check against</div>", unsafe_allow_html=True)

        raw_regs = st.session_state.uploaded_regs
        if len(raw_regs) > 1:
            reg_options = ["All loaded regulations"] + raw_regs
        elif raw_regs:
            reg_options = raw_regs
        else:
            reg_options = ["No regulations loaded"]

        selected_reg = st.selectbox("Regulation", options=reg_options, key="reg_select", label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        # PATCH 4A — conflict panel just before the run button
        _conflict_panel(st.session_state.uploaded_regs)

        if st.button("🔍  Run Gap Analysis", key="btn_gap", use_container_width=False):
            if not vs_ready:
                st.warning("⚠️ Index a regulation first using the panel above.")
            elif not policy_file:
                st.warning("⚠️ Upload your company policy file.")
            else:
                with st.spinner("Mapping obligations and detecting gaps..."):
                    try:
                        reg_name = ", ".join(raw_regs) if selected_reg == "All loaded regulations" else selected_reg
                        result = detect_gaps(policy_file=policy_file, regulation_name=reg_name)
                        st.session_state.gap_results     = result
                        st.session_state.gap_policy_name = policy_file.name
                        st.session_state.gap_reg_name    = reg_name
                        st.session_state.gap_history.append({
                            "label": f"Run {len(st.session_state.gap_history)+1}",
                            "score": result.get("compliance_score", 0),
                            "gaps":  len(result.get("gaps", [])),
                        })
                        log_audit("Gap detection run", f"Policy vs {reg_name}")
                        _save_persisted()
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"Agent error: {e}")

        if st.session_state.gap_results:
            st.markdown("<hr>", unsafe_allow_html=True)
            res   = st.session_state.gap_results
            score = res.get("compliance_score", 0)
            # PATCH 4B — preflight banners + blocked edge-case check
            _preflight_banners(res.get("_preflight", {}))

            if res.get("_edge_case", {}).get("blocked"):
                st.markdown(
                    f"<div style='background:#FEE2E2;border:1px solid #FECACA;"
                    f"border-radius:10px;padding:24px;text-align:center;"
                    f"font-size:13px;color:#7F1D1D;'>"
                    f"🚫 <b>Analysis blocked.</b> "
                    f"{res['_edge_case'].get('reason','')} "
                    f"Fix the issues above and re-run."
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.stop()

            sc    = "#00A870" if score >= 70 else "#CC8800" if score >= 40 else "#CC3333"

            st.markdown("<div class='section-label'>Gap analysis results</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:#F0F4F8;border:1px solid #D1D5DB;border-radius:10px;
                        padding:24px 20px;margin-bottom:20px;text-align:center;'>
                <div style='font-family:"DM Serif Display",serif;font-size:56px;color:{sc};line-height:1;'>{score}%</div>
                <div style='font-size:13px;color:#6B7280;margin-top:6px;'>Compliance coverage score</div>
            </div>""", unsafe_allow_html=True)
            st.progress(score / 100)

            gaps = res.get("gaps", [])
            if gaps:
                st.markdown(f"<div class='section-label' style='margin-top:24px;'>Gaps found ({len(gaps)})</div>", unsafe_allow_html=True)
                for g in gaps:
                    sev_str      = g.get("severity", "medium")
                    color_map    = {"high": "danger", "medium": "warn", "low": "success"}
                    what_missing = g.get("what_is_missing", g.get("detail", ""))
                    rationale    = g.get("rationale", "")
                    rec_action   = g.get("recommended_action", "")
                    reg_source   = g.get("regulation_source", "")
                    st.markdown(f"""
                    <div class='result-box {color_map.get(sev_str,"warn")}' style='margin-bottom:10px;'>
<b>{g.get("obligation","")}</b>  ·  Severity: {sev_str.upper()}
{"" if not reg_source   else chr(10) + "📌 Regulation says: " + reg_source}
{"" if not what_missing else chr(10) + "❌ What's missing: "  + what_missing}
{"" if not rationale    else chr(10) + "💡 Why it matters: "  + rationale}
{"" if not rec_action   else chr(10) + "✅ Action: "           + rec_action}
                    </div>""", unsafe_allow_html=True)

            met = res.get("met", [])
            if met:
                st.markdown(f"<div class='section-label' style='margin-top:24px;'>Obligations met ({len(met)})</div>", unsafe_allow_html=True)
                for m in met:
                    if isinstance(m, dict):
                        label = m.get("obligation", "")
                        note  = m.get("note", "")
                        st.markdown(
                            f"<span style='color:#00A870;font-size:13px;'>✓</span> "
                            f"<span style='font-size:13px;'><b>{label}</b>{'  —  ' + note if note else ''}</span><br>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<span style='color:#00A870;font-size:13px;'>✓</span> "
                            f"<span style='font-size:13px;'>{m}</span><br>",
                            unsafe_allow_html=True,
                        )

            st.markdown("""
            <div class='next-banner-success' style='margin-top:28px;'>
                ✅ <div><b>Analysis complete.</b> View your full compliance dashboard →
                go to <b>Reporting / Dashboard</b> in the sidebar for charts, trends, and the full exportable report.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Export your gap report</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:13px;color:#6B7280;margin-bottom:10px;'>"
                "Download a formatted Excel report with 3 sheets — Gap Analysis (with rationale), Obligations Met, and Summary."
                "</div>", unsafe_allow_html=True,
            )
            try:
                from agents.gap_detector import build_gap_excel
                excel_bytes = build_gap_excel(
                    result=st.session_state.gap_results,
                    regulation_name=st.session_state.get("gap_reg_name", ""),
                    policy_name=st.session_state.get("gap_policy_name", ""),
                )
                pname = st.session_state.get("gap_policy_name","Policy").replace(" ","_").replace(".","_")
                st.download_button(
                    label="⬇  Download Gap Analysis Report (Excel)",
                    data=excel_bytes,
                    file_name=f"GovernIQ_Gap_Analysis_{pname}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=False,
                )
            except Exception as e:
                st.error(f"Excel export error: {e}")
