"""
Student Performance Predictor - Streamlit app (neon dashboard UI)
Flow: typed input values -> saved ML model -> predicted final score -> risk level + tips.
Run:  python -m streamlit run app.py
"""
import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor",
                   page_icon="🎓", layout="wide")

MODEL_PATH = Path("models/best_model.pkl")
INFO_PATH = Path("models/model_info.json")
DATA_PATH = Path("data/student_performance.csv")

# ------------------------------------------------------------------ configurable rules
# Risk bands use the PREDICTED score. They are application rules (not ML classes) and
# were chosen from the training data: High risk ~ lowest 10% of final scores (<60),
# Low risk ~ at or above the 'GOOD' mark (>=70).
RISK_CONFIG = {
    "high_below": 60.0,        # predicted score below this  -> High Risk
    "medium_below": 70.0,      # predicted score below this  -> Medium Risk, else Low Risk
    "low_attendance": 75.0,    # attendance below this raises the risk one level
    "backlog_escalate": 2,     # this many backlogs (or more) raises the risk one level
}
WEAK_BELOW = {"attendance": 75, "internal_marks": 60, "assignment_score": 60,
              "continuous_assessment": 60, "current_cgpa": 60}
LABELS = {"attendance": "Attendance", "internal_marks": "Internal Marks",
          "assignment_score": "Assignment Score", "continuous_assessment": "Continuous Assessment",
          "current_cgpa": "Current CGPA", "backlogs": "Backlogs"}
TIPS = {
    "attendance": "Attendance is {v:g}% - aim for 85% or more (many colleges require at least 75%).",
    "internal_marks": "Internal marks are {v:g} - revise class-test topics and unit notes to lift them.",
    "assignment_score": "Assignment score is {v:g} - submit on time and focus on quality of answers.",
    "continuous_assessment": "Continuous assessment is {v:g}% - prepare better for quizzes and regular evaluations.",
    "current_cgpa": "Current CGPA is {v:g}% - strengthen weak subjects and revise the basics.",
}
# Input fields in the exact on-screen / Enter-key order. This is also the order of the
# model features in train_model.py (the model itself is still fed via info["features"]).
# (widget key, model feature, label, placeholder, type, min, max)
FIELDS = [
    ("in_attendance", "attendance", "📅 Attendance Percentage (%)", "e.g. 82", "float", 0.0, 100.0),
    ("in_internal", "internal_marks", "📝 Internal Assessment Marks (0-100)", "e.g. 75", "float", 0.0, 100.0),
    ("in_assignment", "assignment_score", "📋 Assignment Score (0-100)", "e.g. 80", "float", 0.0, 100.0),
    ("in_cont", "continuous_assessment", "📊 Continuous Assessment Marks (%)", "e.g. 78", "float", 0.0, 100.0),
    ("in_cgpa", "current_cgpa", "🎓 Current CGPA (%)", "e.g. 76", "float", 0.0, 100.0),
    ("in_backlogs", "backlogs", "📚 Number of Backlogs (0-10)", "e.g. 0", "int", 0, 10),
]

# Runs once in the browser. The fields sit in a grid of 2 columns x 3 rows (order 1..6 left-to-right,
# top-to-bottom): Arrow keys move focus to the neighbouring field (and stay put when there is none),
# Enter goes to the next field in order (Shift+Enter = previous, Enter on the last field focuses the
# Predict button). Arrows never change values; Tab / Shift+Tab are left untouched.
NAV_JS = """
<script>
(function () {
  var KEYS = %s, COLS = 2, w, doc;
  try { w = window.parent; doc = w.document; } catch (e) { return; }
  if (w.__sppNav) { doc.removeEventListener("keydown", w.__sppNav, true); }
  function field(i) { return doc.querySelector(".st-key-" + KEYS[i] + " input"); }
  function off() {                      // belt and braces: keep browser autofill off on every field
    for (var i = 0; i < KEYS.length; i++) {
      var f = field(i);
      if (f && f.getAttribute("autocomplete") !== "off") { f.setAttribute("autocomplete", "off"); }
    }
  }
  off();
  if (w.__sppObs) { w.__sppObs.disconnect(); }
  if (w.MutationObserver) { w.__sppObs = new w.MutationObserver(off); w.__sppObs.observe(doc.body, {childList: true, subtree: true}); }
  function go(j) {
    var n = (j >= 0 && j < KEYS.length) ? field(j) : null;
    if (n) { n.focus(); n.select(); return true; }
    return false;
  }
  function handler(e) {
    if (e.isComposing) { return; }
    var idx = -1;
    for (var i = 0; i < KEYS.length; i++) { if (field(i) === e.target) { idx = i; break; } }
    if (idx < 0) { return; }
    var plain = !e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey;
    var col = idx %% COLS;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!e.ctrlKey && !e.altKey && !e.metaKey) { go(idx + (e.key === "ArrowDown" ? COLS : -COLS)); }
      return;
    }
    if (e.key === "ArrowRight" && plain) {
      if (col < COLS - 1 && idx + 1 < KEYS.length) { e.preventDefault(); go(idx + 1); }
      return;
    }
    if (e.key === "ArrowLeft" && plain) {
      if (col > 0) { e.preventDefault(); go(idx - 1); }
      return;
    }
    if (e.key !== "Enter") { return; }
    e.preventDefault();
    if (e.shiftKey) { go(idx - 1); return; }
    if (idx + 1 >= KEYS.length) {
      var b = doc.querySelector(".st-key-predict_btn button");
      if (b) { b.focus(); }
      return;
    }
    go(idx + 1);
  }
  w.__sppNav = handler;
  doc.addEventListener("keydown", handler, true);
})();
</script>
""" % json.dumps([f[0] for f in FIELDS])

LEVEL_COLORS = {"Low Risk": "#22d3a0", "Medium Risk": "#fbbf24", "High Risk": "#fb5b6b"}


def H(s: str) -> str:
    """Flatten an HTML snippet to one line (keeps Streamlit's markdown from breaking it)."""
    return "".join(line.strip() for line in s.strip().splitlines())


# ------------------------------------------------------------------ styling
CSS = """
<style>
.stApp{background:radial-gradient(900px 500px at 85% -5%,rgba(110,80,255,.32),transparent 60%),
 radial-gradient(800px 500px at -5% 105%,rgba(30,120,255,.22),transparent 60%),
 linear-gradient(180deg,#050b21 0%,#0a1440 100%);}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.6rem;padding-bottom:2rem;max-width:1400px;}
html,body,.stApp,[data-testid="stMarkdownContainer"] p{color:#e8ecff;}

/* sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#070f2e,#0b1747);border-right:1px solid rgba(120,140,255,.18);}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:8px;}
[data-testid="stSidebar"] div[role="radiogroup"] label{width:100%;padding:12px 14px;border-radius:12px;
 border:1px solid transparent;transition:all .2s;}
[data-testid="stSidebar"] div[role="radiogroup"] label>div:first-child{display:none;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:rgba(124,92,255,.15);}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
 background:linear-gradient(90deg,rgba(124,92,255,.45),rgba(46,168,255,.30));
 border-color:rgba(140,160,255,.55);box-shadow:0 0 18px rgba(110,90,255,.35);}
[data-testid="stSidebar"] div[role="radiogroup"] label p{font-weight:600;font-size:1.02rem;color:#fff;}
.brand{display:flex;align-items:center;gap:10px;padding:6px 4px 18px;font-weight:700;color:#fff;font-size:1.05rem;}
.brand .logo{width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;
 background:linear-gradient(135deg,#7c5cff,#2ea8ff);box-shadow:0 0 16px rgba(110,90,255,.6);font-size:1.2rem;}

/* hero */
.hero h1{font-size:2.7rem;font-weight:800;margin:0;color:#fff;letter-spacing:-.5px;}
.grad{background:linear-gradient(90deg,#2ea8ff,#8a63ff);-webkit-background-clip:text;background-clip:text;
 -webkit-text-fill-color:transparent;}
.hero .bar{width:190px;height:4px;border-radius:4px;margin:10px 0 18px;background:linear-gradient(90deg,#7c5cff,transparent);}
.chain{display:flex;align-items:center;gap:8px;margin-bottom:22px;}
.chip{width:56px;height:56px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;
 background:rgba(255,255,255,.06);border:1px solid rgba(130,150,255,.35);box-shadow:0 0 14px rgba(90,100,255,.25);color:#9fb4ff;font-weight:700;}
.dots{color:#5b6bd6;letter-spacing:3px;}

/* glass cards */
.st-key-form_card,.glass{background:linear-gradient(145deg,rgba(255,255,255,.08),rgba(255,255,255,.03));
 border:1px solid rgba(130,150,255,.35);border-radius:22px;padding:22px 24px;
 box-shadow:0 10px 40px rgba(0,0,10,.45),inset 0 0 30px rgba(110,90,255,.08);backdrop-filter:blur(10px);}
.glass{margin-bottom:16px;}
.card-title{display:flex;align-items:center;gap:12px;font-size:1.35rem;font-weight:700;color:#fff;margin-bottom:6px;}
.avatar{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;
 background:linear-gradient(135deg,#4f8bf9,#7c5cff);box-shadow:0 0 14px rgba(100,110,255,.6);}
.sub{color:#9fb0e8;font-size:.92rem;margin-bottom:10px;}

/* inputs: plain typed text boxes (no spinners, arrows do nothing) */
[data-testid="stWidgetLabel"] p{color:#d6defc !important;font-weight:600;}
[data-testid="stTextInput"] div[data-baseweb="input"]{background:rgba(255,255,255,.06) !important;
 border:1px solid rgba(120,140,255,.35);border-radius:12px;}
[data-testid="stTextInput"] div[data-baseweb="base-input"]{background:transparent !important;}
[data-testid="stTextInput"] input{color:#fff !important;font-size:1.05rem;}
[data-testid="stTextInput"] input::placeholder{color:#7f8fc9 !important;}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within{border-color:#8a63ff;box-shadow:0 0 0 2px rgba(138,99,255,.35);}
[data-testid="InputInstructions"]{display:none;}
.ferr{color:#ff8a98;font-size:.8rem;margin:-8px 0 8px;}
.st-key-nav_js{position:absolute;width:0;height:0;overflow:hidden;margin:0;padding:0;}

/* predict button */
.st-key-predict_btn button{width:100%;height:3.6rem;border:0;border-radius:14px;font-size:1.2rem;font-weight:800;
 letter-spacing:.5px;color:#fff;background:linear-gradient(90deg,#7c5cff 0%,#4f8bf9 55%,#2ea8ff 100%);
 box-shadow:0 8px 28px rgba(90,100,255,.55);transition:transform .15s,box-shadow .15s;}
.st-key-predict_btn button:hover{transform:translateY(-2px);box-shadow:0 12px 34px rgba(90,110,255,.75);color:#fff;}
.st-key-predict_btn button:focus-visible{outline:3px solid #fff;box-shadow:0 0 0 6px rgba(138,99,255,.8);}
.st-key-predict_btn button p{font-size:1.2rem;font-weight:800;color:#fff;}

/* insights */
.ins h3{margin:0 0 12px;font-size:1.2rem;color:#fff;}
.row{display:flex;gap:14px;align-items:center;}
.pill{display:inline-block;padding:6px 14px;border-radius:999px;font-weight:800;font-size:.95rem;color:#07102b;}
.lvl{color:#9fb0e8;font-size:.85rem;margin-top:6px;}
.sec{margin-top:14px;font-weight:700;color:#fff;font-size:.98rem;}
.ul{margin:6px 0 0;padding-left:18px;color:#cfd8ff;font-size:.9rem;line-height:1.55;}
.bar-row{display:flex;align-items:center;gap:8px;font-size:.8rem;color:#b9c6f5;margin-top:5px;}
.bar-row .lab{width:125px;}
.bar-row .trk{flex:1;height:7px;border-radius:7px;background:rgba(255,255,255,.08);overflow:hidden;}
.bar-row .fill{height:100%;background:linear-gradient(90deg,#7c5cff,#2ea8ff);}
.note{color:#8fa0d8;font-size:.78rem;margin-top:12px;line-height:1.4;}
.warn{background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.45);color:#fde9a6;border-radius:10px;
 padding:8px 12px;font-size:.85rem;margin:6px 0;}
.cards3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
.tbl{width:100%;border-collapse:collapse;font-size:.9rem;color:#cfd8ff;}
.tbl td,.tbl th{padding:6px 8px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ model + data helpers
@st.cache_resource
def load_model():
    """Load the saved model once (no retraining on every click)."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_info():
    with open(INFO_PATH) as f:
        return json.load(f)


@st.cache_data
def load_data():
    """Training data (only used for range warnings); optional."""
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None


def performance_level(score: float) -> str:
    """Application-level rule (NOT an ML class): score -> label."""
    if score >= 85:
        return "EXCELLENT"
    if score >= 70:
        return "GOOD"
    if score >= 50:
        return "AVERAGE"
    return "NEEDS IMPROVEMENT"


def risk_assessment(score: float, v: dict):
    """Risk level from the predicted score, raised one level by clear risk factors.
    The predicted score itself is never modified."""
    cfg = RISK_CONFIG
    order = ["Low Risk", "Medium Risk", "High Risk"]
    base = 2 if score < cfg["high_below"] else 1 if score < cfg["medium_below"] else 0
    reasons = []
    if v["attendance"] < cfg["low_attendance"]:
        reasons.append(f"attendance is below {cfg['low_attendance']:g}%")
    if v["backlogs"] >= cfg["backlog_escalate"]:
        reasons.append(f"{v['backlogs']} backlogs")
    final = min(base + (1 if reasons else 0), 2)
    note = ""
    if final > base:
        note = f"Raised from {order[base]} because " + " and ".join(reasons) + "."
    return order[final], note


def weak_areas(v: dict):
    return [k for k, t in WEAK_BELOW.items() if v[k] < t]


def personalised_tips(v: dict):
    tips = [TIPS[k].format(v=v[k]) for k in weak_areas(v)]
    if v["backlogs"] > 0:
        tips.append(f"Clear your {v['backlogs']} backlog(s) - they pull the predicted score down.")
    return tips or ["All inputs are in a healthy range - keep up the consistent effort."]


def key_insights(v: dict, info: dict):
    pct = {k: v[k] for k in WEAK_BELOW}
    best, worst = max(pct, key=pct.get), min(pct, key=pct.get)
    out = [f"Strongest area: {LABELS[best]} ({pct[best]:g}).",
           f"Area needing most attention: {LABELS[worst]} ({pct[worst]:g})."]
    out.append("No backlogs recorded." if v["backlogs"] == 0 else f"{v['backlogs']} backlog(s) recorded.")
    top = max(info["importance"], key=info["importance"].get)
    out.append(f"The model weighs {LABELS.get(top, top)} most heavily overall.")
    return out


def range_warnings(v: dict):
    """Warn when a typed value lies outside what the model saw in training."""
    df = load_data()
    if df is None:
        return []
    msgs = []
    for k in v:
        lo, hi = df[k].min(), df[k].max()
        if v[k] < lo or v[k] > hi:
            msgs.append(f"{LABELS[k]} = {v[k]:g} is outside the training range ({lo:g} to {hi:g}); "
                        "the estimate may be less reliable.")
    return msgs


def li(items):
    return "<ul class='ul'>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def parse_field(raw, kind, lo, hi):
    """Typed text -> (value, error). Empty text gives (None, None); invalid text gives (None, message)."""
    text = (raw or "").strip().replace("%", "").replace(",", ".")
    if not text:
        return None, None
    try:
        val = float(text)
    except ValueError:
        return None, "Enter a number (digits only)."
    if not math.isfinite(val):
        return None, "Enter a valid number."
    if kind == "int":
        if not val.is_integer():
            return None, "Enter a whole number."
        val = int(val)
    if val < lo or val > hi:
        return None, f"Must be between {lo:g} and {hi:g}."
    return val, None


# ------------------------------------------------------------------ illustration (inline SVG)
def illustration() -> str:
    nodes = [(20, [30, 60, 90]), (60, [20, 50, 80, 110]), (100, [45, 75])]
    lines, dots = "", ""
    for (x1, ys1), (x2, ys2) in zip(nodes, nodes[1:]):
        for a in ys1:
            for b in ys2:
                lines += f'<line x1="{x1}" y1="{a}" x2="{x2}" y2="{b}" stroke="#4f8bf9" stroke-opacity=".45"/>'
    for x, ys in nodes:
        for y in ys:
            dots += f'<circle cx="{x}" cy="{y}" r="5" fill="#0b1233" stroke="#7c5cff" stroke-width="2"/>'
    code = "".join(
        f'<rect x="{188 + ind}" y="{48 + i * 11}" width="{w}" height="5" rx="2.5" fill="{c}"/>'
        for i, (ind, w, c) in enumerate([(0, 60, "#2ea8ff"), (14, 90, "#8a63ff"), (14, 70, "#22d3a0"),
                                         (28, 80, "#2ea8ff"), (14, 50, "#fbbf24"), (0, 40, "#8a63ff"),
                                         (14, 95, "#2ea8ff"), (28, 60, "#22d3a0"), (0, 70, "#8a63ff")]))
    bars = "".join(f'<rect x="{200 + i * 14}" y="{232 - h}" width="9" height="{h}" rx="2" fill="#2ea8ff"/>'
                   for i, h in enumerate([14, 24, 18, 32]))
    return H(f"""
<svg viewBox="0 0 400 250" width="100%" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="gv" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7c5cff"/><stop offset="1" stop-color="#2ea8ff"/></linearGradient></defs>
<circle cx="205" cy="120" r="112" fill="#5b4bff" opacity=".13"/>
<g transform="translate(14,6)">{lines}{dots}</g>
<rect x="170" y="30" width="204" height="122" rx="9" fill="#0b1233" stroke="url(#gv)" stroke-width="2.5"/>
<rect x="180" y="40" width="184" height="102" rx="4" fill="#0e1850"/>
{code}
<polygon points="152,152 392,152 408,168 136,168" fill="#18246b" stroke="#4f8bf9" stroke-opacity=".6"/>
<rect x="30" y="200" width="108" height="16" rx="3" fill="#6d4cff"/>
<rect x="36" y="184" width="98" height="16" rx="3" fill="#2e7dff"/>
<rect x="32" y="168" width="102" height="16" rx="3" fill="#8a63ff"/>
<polygon points="83,134 122,149 83,164 44,149" fill="url(#gv)"/>
<path d="M64 157v9q19 9 38 0v-9" fill="#3b3ad6"/>
<line x1="118" y1="151" x2="118" y2="164" stroke="#fbbf24" stroke-width="2"/><circle cx="118" cy="166" r="3" fill="#fbbf24"/>
<rect x="186" y="178" width="84" height="62" rx="9" fill="#0e1850" stroke="#4f8bf9" stroke-opacity=".6"/>{bars}
<rect x="284" y="178" width="96" height="62" rx="9" fill="#0e1850" stroke="#8a63ff" stroke-opacity=".7"/>
<polyline points="294,222 316,206 336,214 358,192 372,196" fill="none" stroke="#22d3a0" stroke-width="3" stroke-linecap="round"/>
<circle cx="372" cy="196" r="4" fill="#22d3a0"/>
</svg>""")


# ------------------------------------------------------------------ page pieces
def hero():
    st.markdown(H("""
<div class="hero"><h1>🎓 Student <span class="grad">Performance</span> Predictor</h1><div class="bar"></div>
<div class="chain"><div class="chip">🎓</div><span class="dots">· · ·</span><div class="chip">&lt;/&gt;</div>
<span class="dots">· · ·</span><div class="chip">🧠</div><span class="dots">· · ·</span><div class="chip">🗄️</div></div></div>"""),
        unsafe_allow_html=True)


def insights_card(result):
    """Right-hand 'Prediction Insights' card. result is None until Predict is clicked."""
    if result is None:
        body = H("""
<div class="row"><div class="avatar" style="opacity:.6">📊</div>
<div class="sub" style="margin:0">Your predicted score, risk level, key insights and personalised
suggestions will appear here after you click <b>PREDICT PERFORMANCE</b>.</div></div>""")
    else:
        score, risk, note, level, v, info = (result[k] for k in
                                             ("score", "risk", "note", "level", "values", "info"))
        color = LEVEL_COLORS[risk]
        dash = score / 100 * 326.7
        imp = sorted(info["importance"].items(), key=lambda kv: -kv[1])[:4]
        bars = "".join(f'<div class="bar-row"><span class="lab">{LABELS.get(k, k)}</span>'
                       f'<span class="trk"><span class="fill" style="width:{min(p, 100)}%;display:block"></span></span>'
                       f'<span>{p:g}%</span></div>' for k, p in imp)
        body = H(f"""
<div class="row"><svg width="124" height="124" viewBox="0 0 130 130">
<defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7c5cff"/><stop offset="1" stop-color="#2ea8ff"/></linearGradient></defs>
<circle cx="65" cy="65" r="52" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="10"/>
<circle cx="65" cy="65" r="52" fill="none" stroke="url(#rg)" stroke-width="10" stroke-linecap="round" stroke-dasharray="{dash:.1f} 326.7" transform="rotate(-90 65 65)"/>
<text x="65" y="70" text-anchor="middle" font-size="30" font-weight="800" fill="#ffffff">{score}</text>
<text x="65" y="90" text-anchor="middle" font-size="12" fill="#9fb0e8">out of 100</text></svg>
<div><div class="sub" style="margin:0">Academic risk level</div>
<span class="pill" style="background:{color};box-shadow:0 0 16px {color}88">{risk}</span>
<div class="lvl">Performance level: <b style="color:#fff">{level}</b></div></div></div>
{f'<div class="note" style="margin-top:8px">{note}</div>' if note else ''}
<div class="sec">💡 Key performance insights</div>{li(key_insights(v, info))}
<div class="sec">🧠 What the model weighs most</div>{bars}
<div class="sec">🎯 Personalised improvement suggestions</div>{li(personalised_tips(v))}
<div class="note">Risk levels are estimates from a model trained on synthetic data - they are not a
definitive judgment about any student.</div>""")
    st.markdown(H(f'<div class="glass ins"><h3>📈 Prediction Insights</h3>{body}</div>'),
                unsafe_allow_html=True)


def page_prediction(model, info):
    hero()
    form_col, side_col = st.columns([1.55, 1], gap="large")
    result = None
    with form_col:
        with st.container(key="form_card"):
            st.markdown(H('<div class="card-title"><div class="avatar">👤</div>Predict Student Performance</div>'
                          '<div class="sub">Type each value. Enter or the arrow keys move between fields.</div>'),
                        unsafe_allow_html=True)
            parsed, errors, empty = {}, {}, []
            for row in range(0, len(FIELDS), 2):            # two fields per row keeps the tab/Enter order 1..6
                for col, (key, feat, label, hint, kind, lo, hi) in zip(st.columns(2, gap="large"), FIELDS[row:row + 2]):
                    with col:
                        raw = st.text_input(label, key=key, placeholder=hint, max_chars=8, autocomplete="off")
                        val, err = parse_field(raw, kind, lo, hi)
                        if err:
                            errors[feat] = err
                            st.markdown(f'<div class="ferr">⚠ {err}</div>', unsafe_allow_html=True)
                        elif val is None:
                            empty.append(label.split(" ", 1)[1].split(" (")[0])
                        else:
                            parsed[feat] = val
            clicked = st.button("🎓 PREDICT PERFORMANCE", key="predict_btn")
            with st.container(key="nav_js"):
                st.components.v1.html(NAV_JS, height=0)

            if clicked:
                if errors:
                    st.error("Please correct the highlighted field(s) before predicting.")
                elif empty:
                    st.error("Please enter a value for: " + ", ".join(empty) + ".")
                else:
                    values = dict(parsed)
                    try:
                        # unchanged: columns in the exact order the model was trained on
                        X_new = pd.DataFrame([values])[info["features"]]
                        score = round(min(max(float(model.predict(X_new)[0]), 0), 100), 1)
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                    else:
                        risk, note = risk_assessment(score, values)
                        result = {"score": score, "risk": risk, "note": note, "values": values,
                                  "level": performance_level(score), "info": info}
                        for m in range_warnings(values):
                            st.markdown(f'<div class="warn">⚠️ {m}</div>', unsafe_allow_html=True)
    with side_col:
        st.markdown(f'<div class="glass" style="padding:10px 14px">{illustration()}</div>',
                    unsafe_allow_html=True)
        insights_card(result)


def page_home():
    hero()
    st.markdown(H("""
<div class="glass"><div class="card-title">Welcome</div>
<div class="sub">This project estimates a student's final score (0-100) from six academic inputs using a
machine-learning regression model, then maps the score to a performance level and an academic risk level.</div>
<div class="cards3">
<div class="glass" style="margin:0"><b>1 &middot; Enter</b><div class="sub">Type attendance, marks, CGPA and backlogs.</div></div>
<div class="glass" style="margin:0"><b>2 &middot; Predict</b><div class="sub">The saved model estimates the final score.</div></div>
<div class="glass" style="margin:0"><b>3 &middot; Improve</b><div class="sub">Get a risk level and tips based on your inputs.</div></div>
</div></div>"""), unsafe_allow_html=True)
    st.info("Open **Prediction** in the sidebar to try it.")


# ------------------------------------------------------------------ main
if not MODEL_PATH.exists() or not INFO_PATH.exists():
    st.error("Model not found. Run `python generate_dataset.py` and `python train_model.py` first.")
    st.stop()

try:
    model = load_model()
    info = load_info()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}. Re-run `python train_model.py`.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🎓</div>Student Performance Predictor</div>',
                unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏠 Home", "🔮 Prediction"], index=1,
                    label_visibility="collapsed")
    st.markdown(f'<div class="note">Model: {info["best_model"]}<br>Individual prediction only.</div>',
                unsafe_allow_html=True)

if page.endswith("Home"):
    page_home()
else:
    page_prediction(model, info)
