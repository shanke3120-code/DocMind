# DocMind: AI Document Assistant (RAG with Streamlit + LangChain + FAISS + Groq)
# Run with:  streamlit run app.py
# Extra package needed for the "open source page" popup:  pip install pymupdf
import streamlit as st
# Page settings: must be the first Streamlit command
st.set_page_config(
    page_title="DocMind | AI Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
import base64, html, inspect, json, math, os, random, re, tempfile, time, uuid  # light imports only
from datetime import datetime
from pathlib import Path
from typing import List

# ---------- Settings you can change ----------
OWNER = {
    "name": "Shashank Kumar Jha",
    "initials": "SJ",
    "role": "B.Tech CSE (Data Science)",
    "badge": "Open to Data Science and AI/ML internships",
    "github": "https://github.com/shanke3120-code",
    "linkedin": "https://www.linkedin.com/in/shashank-jha-520689325/",
    "email": "mailto:shanke3120@gmail.com",
}

PRIMARY_MODEL = "openai/gpt-oss-120b"   # main model
BACKUP_MODEL = "openai/gpt-oss-20b"     # used automatically if the main one fails (e.g. daily limit)
SAVE_CHATS = False   # True keeps chats in chats.json. Local use only: on a shared server it would mix users' chats.
PIN_PROFILE = True   # keeps the profile card at the bottom of the sidebar
MIN_SPLASH = 2.5   # the splash stays at least this many seconds, even when loading is instant
# Newer Streamlit uses width="stretch"; older versions use use_container_width=True
STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}

# ---------- Avatars and icons (SVG drawn in code, turned into data URIs) ----------
AI_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
          '<stop offset="0" stop-color="#a79bff"/><stop offset="1" stop-color="#5eead4"/></linearGradient></defs>'
          '<circle cx="14" cy="14" r="14" fill="url(#g)"/><rect x="7" y="9" width="14" height="10" rx="5" fill="#0b0d14"/>'
          '<circle cx="11.2" cy="14" r="1.6" fill="#5eead4"/><circle cx="16.8" cy="14" r="1.6" fill="#5eead4"/></svg>')
USER_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28"><defs><linearGradient id="u" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#f472b6"/><stop offset="1" stop-color="#8b7bff"/></linearGradient></defs>'
            '<circle cx="14" cy="14" r="14" fill="url(#u)"/><circle cx="14" cy="11" r="4.2" fill="#0b0d14"/>'
            '<path d="M5.5 23c1.3-4.6 4.8-6.4 8.5-6.4s7.2 1.8 8.5 6.4a14 14 0 0 1-17 0z" fill="#0b0d14"/></svg>')
GITHUB_PATH = ("M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04"
               "-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 "
               "1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 "
               "1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 "
               "2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92"
               ".42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627"
               "-5.373-12-12-12")
LINKEDIN_PATH = ("M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561"
                 "h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 "
                 "0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452z"
                 "M22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 "
                 "23.2 0 22.222 0h.003z")
GITHUB_SVG = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="#e8eaf6" d="{GITHUB_PATH}"/></svg>'
LINKEDIN_SVG = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="#e8eaf6" d="{LINKEDIN_PATH}"/></svg>'
MAIL_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#e8eaf6" stroke-width="1.8" '
            'stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M3.5 7.5 12 13l8.5-5.5"/></svg>')

def svg_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
AI_URI, USER_URI = svg_uri(AI_SVG), svg_uri(USER_SVG)

# ---------- Styling (dark theme, glowing cards, modern fonts) ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Sora:wght@500;600;700&display=swap');
html, body, .stMarkdown, button, input, textarea {font-family:'DM Sans',sans-serif !important}
h1, h2, h3, .sora {font-family:'Sora',sans-serif !important}
html, body {background:#0b0d14}  /* the dark base color now lives on the page itself */
/* make Streamlit's own layers see-through so our canvas shows behind them */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:transparent !important; color:#e8eaf6}
[data-testid="stBottom"] > div {background:transparent !important}  /* the area around the chat input */

[data-testid="stHeader"] {background:transparent}
footer, [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {display:none !important}
[data-testid="stSidebar"] {background:#0b0d14; border-right:1px solid #252b46}
[data-testid="stSidebar"] .stButton button {justify-content:flex-start; text-align:left}
.hero-title {font-family:'Sora',sans-serif; font-weight:700; letter-spacing:-1.2px; font-size:clamp(28px,3.2vw,43px);
             line-height:1.17; margin:0; background:linear-gradient(90deg,#a78bfa,#8b9cff 48%,#5eead4);
             -webkit-background-clip:text; background-clip:text; color:transparent}
.subhead {color:#9ba6ca; font-size:15px; margin:8px 0 14px}
.brand-row {display:flex; align-items:center; gap:10px; margin:0 0 14px}
.brand-name {font-family:'Sora',sans-serif; font-size:21px; font-weight:700}
.brand-sub {font-size:11px; color:#9ba6ca; margin-top:-3px}
.section-label {color:#9ba6ca; font-size:10px; font-weight:700; letter-spacing:1.15px; text-transform:uppercase; margin:16px 0 8px}
.dm-card {background:#111420; border:1px solid #252b46; border-radius:15px; padding:13px 14px; margin:0 0 9px}
.dm-card:hover {border-color:#9b83ff; box-shadow:0 8px 24px rgba(110,90,220,.10)}
.doc-row {display:flex; justify-content:space-between; gap:8px; font-size:12px; padding:3px 0}
.doc-name {overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:78%}
.stats {display:flex; gap:8px; margin-top:10px; padding-top:10px; border-top:1px solid #252b46}
.stats div {flex:1; text-align:center}
.stats b {display:block; font-family:'Sora',sans-serif; font-size:20px}
.stats span, .muted, .small-note {color:#9ba6ca; font-size:11px}
.orb {width:70px; height:70px; margin:0 auto; border-radius:50%; background:linear-gradient(135deg,#a79bff,#5eead4); display:flex;
      align-items:center; justify-content:center; animation:breathe-s 3s ease-in-out infinite}
.orb img {width:48px; height:48px}
@keyframes breathe-s {0%,100% {transform:scale(1); box-shadow:0 0 26px rgba(139,123,255,.25)}
                      50% {transform:scale(1.07); box-shadow:0 0 48px rgba(94,234,212,.28)}}
[data-testid="stChatMessage"] {background:#111420; border:1px solid #252b46; border-radius:16px; padding:14px 16px; margin-bottom:13px}
[data-testid="stChatInput"] {border:1px solid #9b83ff; border-radius:16px; box-shadow:0 0 24px rgba(139,123,255,.12)}
[data-testid="stFileUploaderDropzone"] {border:1px dashed #9b83ff; border-radius:15px; background:#111420}
.stButton > button, .stDownloadButton > button {border-radius:11px; border:1px solid #252b46; background:#111420; color:#e8eaf6; transition:all .18s ease}
.stButton > button:hover, .stDownloadButton > button:hover {border-color:#9b83ff; box-shadow:0 0 17px rgba(139,123,255,.2)}
[data-testid="stBaseButton-primary"] {background:rgba(155,131,255,.18) !important; border:1px solid #9b83ff !important; color:#e8eaf6 !important}
.st-key-newchat button {background:linear-gradient(90deg,#8b7bff,#5eead4) !important; color:#0b0d14 !important; border:0 !important; font-weight:700; justify-content:center !important}
.think {display:inline-flex; align-items:center; gap:6px; padding:8px 13px; border-radius:999px; font-size:12px; color:#9b83ff;
        background:rgba(139,123,255,.10); border:1px solid rgba(139,123,255,.3)}
.dot {display:inline-block; width:6px; height:6px; border-radius:50%; background:#9b83ff; animation:pulse 1.1s infinite}
.dot:nth-child(2) {animation-delay:.2s} .dot:nth-child(3) {animation-delay:.4s}
@keyframes pulse {0%,100% {opacity:.3} 50% {opacity:1}}
.gap {height:28px}
/* hides the 1px helper iframe that runs our background script */
.st-key-bgjs {position:absolute; height:0; margin:0; overflow:hidden}

/* Theme switch (fixed, top right) */
.st-key-themebox {position:fixed; top:12px; right:140px; left:auto; width:fit-content !important; z-index:1000001; padding:2px 14px 0; border-radius:999px;
                  background:#111420; border:1px solid #252b46; box-shadow:0 0 18px rgba(139,123,255,.18)}
/* Home card: the whole card is clickable (the button's ::after stretches over it) */
.st-key-herocard {position:relative; padding:20px 22px; border-radius:18px; background:rgba(17,20,32,.55); backdrop-filter:blur(6px); border:1px solid #252b46;
                  min-height:150px; transition:all .2s ease}
.st-key-herocard:hover {border-color:#9b83ff; box-shadow:0 0 30px rgba(139,123,255,.28)}
.st-key-herobtn, .st-key-herobtn div, .st-key-orbbtn, .st-key-orbbtn div {position:static !important}
.st-key-herobtn button {background:linear-gradient(90deg,#8b7bff,#5eead4) !important; color:#0b0d14 !important; border:0 !important;
                        font-weight:700; font-size:17px; padding:.65rem 1.6rem; border-radius:14px;
                        animation:ctapulse 2.6s ease-in-out infinite}
.st-key-herobtn button::after {content:""; position:absolute; inset:0; border-radius:18px}
@keyframes ctapulse {0%,100% {box-shadow:0 0 20px rgba(139,123,255,.35)} 50% {box-shadow:0 0 44px rgba(94,234,212,.55)}}
.st-key-orbcard {position:relative; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:25px;
                 min-height:150px; border-radius:20px; border:1px solid #252b46; transition:all .2s ease;
                 background:radial-gradient(circle at 65% 40%,rgba(112,94,230,.16),transparent 42%),rgba(17,20,32,.55); backdrop-filter:blur(6px)}
.st-key-orbcard:hover {border-color:#5eead4; box-shadow:0 0 30px rgba(94,234,212,.25)}
.st-key-orbcard:hover .orb {animation-duration:1.2s}
.st-key-orbcard [data-testid="stMarkdownContainer"] {display:flex; justify-content:center; width:100%}
.st-key-orbbtn button {border-radius:999px; font-size:13px; border:1px solid #5eead4; color:#5eead4; background:rgba(94,234,212,.08)}
.st-key-orbbtn button::after {content:""; position:absolute; inset:0; border-radius:20px}
@keyframes uploadpulse {0%,100% {box-shadow:0 0 0 0 rgba(155,131,255,0)} 50% {box-shadow:0 0 0 8px rgba(155,131,255,.45)}}
.pulse-upload {animation:uploadpulse .8s ease-in-out 3}
.bubble {display:flex; gap:10px; align-items:flex-start; margin:12px 0; animation:fadein .4s ease}
.bubble img {width:34px; height:34px; flex:none}
.bubble .said {padding:12px 14px; border-radius:4px 16px 16px 16px; background:#151a2a; border:1px solid #252b46; font-size:14px; line-height:1.5}
@keyframes fadein {from {opacity:0; transform:translateY(6px)} to {opacity:1; transform:none}}
/* Source chips: real buttons that open the PDF page */
[class*="st-key-srcrow_"] {flex-wrap:wrap; gap:8px !important}
[class*="st-key-srcrow_"] > div {width:auto !important}
[class*="st-key-src_"] button {min-height:0; padding:5px 12px; border-radius:999px; font-size:12px; color:#5eead4;
                               background:rgba(94,234,212,.08); border:1px solid rgba(94,234,212,.35)}
[class*="st-key-src_"] button:hover {background:rgba(94,234,212,.16); border-color:#5eead4; box-shadow:0 0 16px rgba(94,234,212,.35)}
[class*="st-key-src_"] button:disabled {opacity:.4; text-decoration:line-through}
.small-warn {color:#f9a8d4; font-size:11px}

.source-passage {margin:0 0 12px; padding:0;}
.source-passage-title {margin:0 0 3px; padding:0; font-size:15px; line-height:1.3; font-weight:700; color:#e8eaf6;}
.source-passage-text {margin:0; padding:0; color:#9ba6ca; font-size:12px; line-height:1.45; overflow-wrap:anywhere;}
.source-passage:last-child {margin-bottom:0;}
.quiz-nav [data-testid="stRadio"] > div {gap:8px;}
.quiz-nav [data-testid="stRadio"] label {border:1px solid #252b46; border-radius:10px; padding:5px 12px; background:#111420;}
/* Quiz */
.quiz-title {font-family:'Sora',sans-serif; font-size:22px; font-weight:700; margin:14px 0 6px}
button[data-baseweb="tab"] {border-radius:12px 12px 0 0; transition:background .18s ease}
button[data-baseweb="tab"]:hover {background:rgba(139,123,255,.12)}
button[data-baseweb="tab"] p {font-size:17px !important; font-weight:600}
[data-baseweb="tab-list"] {gap:8px}
[data-baseweb="tab-highlight"] {background:linear-gradient(90deg,#8b7bff,#5eead4) !important; height:3px !important}
.qcard {padding:24px 26px; border-radius:20px; margin:14px 0 18px; border:1px solid transparent;
        background:linear-gradient(#111420,#111420) padding-box, linear-gradient(135deg,#8b7bff,#5eead4) border-box;
        box-shadow:0 0 34px rgba(139,123,255,.18)}
.qbadge {display:inline-block; padding:4px 13px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.8px;
         color:#0b0d14; background:linear-gradient(90deg,#a79bff,#5eead4)}
.qtext {font-family:'Sora',sans-serif; font-size:24px; line-height:1.38; margin-top:14px; font-weight:600}
.qprog {height:6px; border-radius:999px; background:#1d2133; margin-top:18px; overflow:hidden}
.qprog i {display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,#8b7bff,#5eead4)}
[class*="st-key-qz"] [role="radiogroup"] {gap:12px}
[class*="st-key-qz"] label[data-baseweb="radio"] {width:100%; margin:0; padding:15px 20px; border-radius:14px; background:#111420;
                                                  border:1px solid #252b46; cursor:pointer; opacity:1 !important; transition:all .18s ease}
[class*="st-key-qz"] label[data-baseweb="radio"]:hover {transform:translateY(-2px); border-color:#9b83ff; box-shadow:0 0 24px rgba(139,123,255,.32)}
[class*="st-key-qz"] label[data-baseweb="radio"]:has(input:checked) {border:1px solid transparent;
        background:linear-gradient(#171b2e,#171b2e) padding-box, linear-gradient(135deg,#8b7bff,#5eead4) border-box;
        box-shadow:0 0 28px rgba(139,123,255,.38)}
[class*="st-key-qz"] label[data-baseweb="radio"] p {font-size:18px !important; line-height:1.45}
.fb {padding:16px 18px; border-radius:16px; margin-top:16px; border:1px solid; font-size:16px; line-height:1.55}
.fb b {font-family:'Sora',sans-serif; font-size:17px}
.fb .why {margin-top:6px; color:#e8eaf6; opacity:.88}
.fb.good {color:#5eead4; border-color:rgba(94,234,212,.55); background:rgba(94,234,212,.08); box-shadow:0 0 24px rgba(94,234,212,.16)}
.fb.bad {color:#f9a8d4; border-color:rgba(244,114,182,.55); background:rgba(244,114,182,.08); box-shadow:0 0 24px rgba(244,114,182,.16)}
.fb.skip {color:#c9c2ff; border-color:rgba(155,131,255,.55); background:rgba(155,131,255,.08)}
.ring-wrap {display:flex; align-items:center; gap:26px; flex-wrap:wrap; padding:22px 26px; border-radius:20px; background:#111420;
            border:1px solid #252b46; box-shadow:0 0 34px rgba(139,123,255,.16)}
.ring {width:140px; height:140px; border-radius:50%; display:flex; align-items:center; justify-content:center;
       background:conic-gradient(from 0deg,#8b7bff,#5eead4 calc(var(--p) * 1%),#252b46 0)}
.ring-in {width:108px; height:108px; border-radius:50%; background:#111420; display:flex; flex-direction:column;
          align-items:center; justify-content:center}
.ring-in b {font-family:'Sora',sans-serif; font-size:28px}
.ring-msg {font-family:'Sora',sans-serif; font-size:22px; font-weight:700}
.pill {display:inline-block; padding:4px 11px; margin:8px 6px 0 0; border-radius:999px; font-size:12px; font-weight:600; border:1px solid}
.pill.good {color:#5eead4; border-color:rgba(94,234,212,.5); background:rgba(94,234,212,.08)}
.pill.bad {color:#f9a8d4; border-color:rgba(244,114,182,.5); background:rgba(244,114,182,.08)}
.pill.skip {color:#c9c2ff; border-color:rgba(155,131,255,.5); background:rgba(155,131,255,.08)}
.hist-title {font-family: 'Sora', sans-serif;font-size: 18px; font-weight: 700; margin: 20px 0 12px;}
.hist-row {display: grid;grid-template-columns: minmax(75px, auto) minmax(40px, 1fr)
    auto auto auto; align-items: center; gap: 12px; width: 100%; max-width: 100%; box-sizing: border-box; padding: 12px 16px; margin-bottom: 8px;
    border-radius: 14px; background: #111420; border: 1px solid #252b46; font-size: 14px; overflow: hidden;}

.hist-row.best {border-color: #9b83ff;box-shadow: 0 0 20px rgba(139,123,255,.22);}

.hist-bar {
    width: 100%;
    min-width: 0;
    height: 8px;
    border-radius: 999px;
    background: #1d2133;
    overflow: hidden;
}

.hist-bar i {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #8b7bff, #5eead4);
}

@media (max-width: 650px) {
    .hist-row {
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 8px 12px;
    }

    .hist-bar {
        grid-column: 1 / -1;
        grid-row: 2;
    }

    .hist-row > span,
    .hist-row > b {
        min-width: 0;
    }
}
.hist-bar i {display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,#8b7bff,#5eead4)}
.badge-best {padding:2px 9px; border-radius:999px; font-size:11px; font-weight:700; color:#0b0d14; background:linear-gradient(90deg,#a79bff,#5eead4)}
/* Profile card */
.pcard {border-radius:16px; padding:12px; border:1px solid transparent;
        background:linear-gradient(#111420,#111420) padding-box, linear-gradient(135deg,#8b7bff,#5eead4) border-box}
.prow {display:flex; align-items:center; gap:10px}
.pav {width:42px; height:42px; border-radius:50%; flex:none; display:flex; align-items:center; justify-content:center;
      font-family:'Sora',sans-serif; font-weight:700; color:#0b0d14; background:linear-gradient(135deg,#a79bff,#5eead4);
      box-shadow:0 0 22px rgba(139,123,255,.45)}
.pname {font-family:'Sora',sans-serif; font-weight:600; font-size:14px; line-height:1.2}
.pbadge {display:inline-block; margin-top:9px; padding:4px 9px; border-radius:999px; font-size:10.5px; color:#5eead4;
         background:rgba(94,234,212,.08); border:1px solid rgba(94,234,212,.35)}
.picons {display:flex; gap:10px; margin-top:10px}
.pico {width:34px; height:34px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
       background:#151a2a; border:1px solid #252b46; transition:all .18s ease}
.pico:hover {transform:translateY(-2px); border-color:#9b83ff; box-shadow:0 6px 18px rgba(139,123,255,.4)}
.pico img {width:16px; height:16px}
.tech {display:inline-block; font-size:10px; color:#9ba6ca; padding:2px 7px; border:1px solid #252b46; border-radius:6px; margin:8px 4px 0 0}
.flow {display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 12px}
.flow span {padding:5px 10px; border-radius:999px; font-size:12px; background:rgba(139,123,255,.12); border:1px solid rgba(139,123,255,.35)}
/* Splash screen: one smooth, slowing climb from 0 to 99%; it reaches 100% only when loading is really done */
@property --n {syntax:'<integer>'; initial-value:0; inherits:false}
/* two-stage climb: fast to 90% in the first ~2 seconds, then a slow creep toward 99% */
@keyframes crawl-n {0% {--n:0} 7% {--n:90} 100% {--n:99}}
@keyframes crawl-w {0% {width:0} 7% {width:90%} 100% {width:99%}}
@keyframes breathe {0%,100% {transform:scale(1); box-shadow:0 0 40px rgba(139,123,255,.35)}
                    50% {transform:scale(1.09); box-shadow:0 0 100px rgba(94,234,212,.5)}}
/* Futuristic AI particle field */
.splash {
    overflow: hidden;
    isolation: isolate;
}

.splash::before {
    content: "";
    position: absolute;
    inset: -50%;
    z-index: -1;
    pointer-events: none;
    background:
        radial-gradient(circle at 50% 50%, rgba(139,123,255,.16), transparent 22%),
        radial-gradient(circle at 30% 40%, rgba(94,234,212,.10), transparent 25%),
        radial-gradient(circle at 70% 60%, rgba(139,123,255,.10), transparent 28%);
    animation: nebula-drift 9s ease-in-out infinite alternate;
}

.particle-field {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
}

.particle {
    position: absolute;
    left: var(--x);
    top: var(--y);
    width: var(--size);
    height: var(--size);
    border-radius: 50%;
    background: #a79bff;
    box-shadow: 0 0 12px rgba(139,123,255,.8);
    opacity: 0;
    animation: particle-float var(--duration) ease-in-out infinite;
    animation-delay: var(--delay);
}

.particle:nth-child(3n) {
    background: #5eead4;
    box-shadow: 0 0 12px rgba(94,234,212,.8);
}

.splash-orb {
    position: relative;
    z-index: 1;
    filter: drop-shadow(0 0 28px rgba(139,123,255,.35));
}

.splash-title {
    letter-spacing: 5px;
    background: linear-gradient(90deg, #a79bff, #f2f0ff, #5eead4);
    background-size: 200% auto;
    color: transparent;
    background-clip: text;
    -webkit-background-clip: text;
    animation: title-shimmer 3s linear infinite;
}

.splash-subtitle {
    color: #b8b5d6;
    letter-spacing: 1.5px;
    animation: subtitle-pulse 2s ease-in-out infinite;
}

@keyframes particle-float {
    0%   { transform: translate3d(0, 20px, 0) scale(.5); opacity: 0; }
    25%  { opacity: .8; }
    70%  { opacity: .55; }
    100% { transform: translate3d(35px, -100px, 0) scale(1.2); opacity: 0; }
}

@keyframes nebula-drift {
    from { transform: translate3d(-2%, -1%, 0) rotate(0deg); }
    to   { transform: translate3d(2%, 1%, 0) rotate(8deg); }
}

@keyframes title-shimmer {
    from { background-position: 0% center; }
    to   { background-position: 200% center; }
}

@keyframes subtitle-pulse {
    0%, 100% { opacity: .65; }
    50%      { opacity: 1; }
}

.splash {position:fixed; inset:0; z-index:999999999; background:#0b0d14; color:#e8eaf6; display:flex; flex-direction:column;
         align-items:center; justify-content:center; gap:22px; text-align:center}
.splash-orb {width:132px; height:132px; border-radius:50%; animation:breathe 3s ease-in-out infinite}
.track {width:min(420px,80vw); height:10px; border-radius:999px; background:#1d2133; overflow:hidden}
.fill {height:100%; width:0; border-radius:999px; background:linear-gradient(90deg,#8b7bff,#5eead4); animation:crawl-w 30s ease-out forwards}
.pct {font-family:'Sora',sans-serif; font-size:20px; font-weight:600; counter-reset:n var(--n); animation:crawl-n 30s ease-out forwards}
.pct::after {content:counter(n) "%"}
.fill.done {width:100%; animation:none}
.pct.done {animation:none}
.pct.done::after {content:"100%"}
@media (prefers-reduced-motion: reduce) {* {animation-duration:.01ms !important; transition:none !important}}
</style>
"""

# Light mode: the whole page is colour-inverted (every widget stays readable), then photos and avatars are flipped back
LIGHT_CSS = """
<style>
html {filter:invert(1) hue-rotate(180deg)}
html, body {background:#0b0d14}
img.keep, .splash-orb, [data-testid="stChatMessage"] img, [data-testid="stImage"] img, [data-testid="stDialog"] img {filter:hue-rotate(180deg) invert(1)}
</style>
"""
light_mode = bool(st.session_state.get("light", False))  # the switch is created further down; its value is already known here
st.markdown(CSS + (LIGHT_CSS if light_mode else ""), unsafe_allow_html=True)
if PIN_PROFILE:  # sticky profile card (remove this block if it misbehaves on your screen)
    st.markdown("<style>.st-key-profilebox{position:sticky;bottom:0;z-index:5;background:#0b0d14;padding:8px 0 2px;border-top:1px solid #252b46}</style>",
                unsafe_allow_html=True)
    
# ---------- Splash screen (first visit only) ----------

PARTICLES = "".join(
    f'<span class="particle" style="--x:{x}%;--y:{y}%;'
    f'--size:{size}px;--duration:{duration}s;--delay:-{delay}s"></span>'
    for x, y, size, duration, delay in [
        (8, 20, 3, 7, 1),
        (18, 70, 4, 9, 3),
        (28, 35, 2, 8, 2),
        (38, 82, 4, 10, 5),
        (48, 15, 3, 7, 4),
        (58, 60, 2, 9, 1),
        (68, 30, 4, 8, 6),
        (78, 75, 3, 11, 2),
        (88, 45, 2, 7, 3),
        (95, 18, 4, 10, 5),
        (14, 48, 2, 8, 4),
        (42, 55, 3, 9, 6),
        (73, 12, 2, 7, 2),
        (83, 88, 3, 10, 1),
        (53, 90, 4, 8, 5),
    ]
)

SPLASH = (
    '<div class="splash">'
    '<div class="particle-field" aria-hidden="true">' + PARTICLES + '</div>'
    '<img class="splash-orb" src="' + AI_URI + '" alt="DocMind AI">'
    '<div class="splash-title" style="font-size:34px;font-weight:700">DocMind</div>'
    '<div class="splash-subtitle">__LABEL__</div>'
    '<div><div class="track"><div class="fill__DONE__"></div></div>'
    '<div class="pct__DONE__" style="margin-top:12px"></div></div>'
    '</div>'
)

booting = not st.session_state.get("booted", False)
splash = st.empty()
boot_start = time.time()
def show_splash(label, done=False):
    # Shown once at the start (the bar starts its smooth climb) and once at the end (the bar jumps to 100%)
    splash.markdown(SPLASH.replace("__LABEL__", label).replace("__DONE__", " done" if done else ""),
                    unsafe_allow_html=True)
if booting:
    show_splash("Waking up your AI…")

# ---------- Heavy imports (they run while the splash is visible) ----------
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()  # reads GROQ_API_KEY from the .env file
@st.cache_resource  # load the embedding model once and reuse it
def get_embeddings():
    return HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
if booting:
    get_embeddings()  # still loads the model (instant if the server is warm)
    elapsed = time.time() - boot_start
    if elapsed < MIN_SPLASH:                 # loading was fast, so wait out the rest of the time
        time.sleep(MIN_SPLASH - elapsed)
    show_splash("Ready!", done=True)         # always show the 100% screen now
    time.sleep(0.6)
    splash.empty()
    st.session_state["booted"] = True

# ---------- PDF loading and indexing ----------
def load_pdf(uploaded_file):
    # PyPDFLoader needs a file path, so save the upload to a temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        pages = PyPDFLoader(path).load()
    finally:
        if os.path.exists(path):
            os.remove(path)
    for page in pages:
        page.page_content = " ".join(page.page_content.split())  # fix one-word-per-line text
        page.metadata["source"] = uploaded_file.name              # real file name for citations
        page.metadata.setdefault("page_label", str(page.metadata.get("page", 0) + 1))
    return pages
def build_index(files):
    # Read every PDF, cut it into chunks, embed the chunks and store them in FAISS
    pages, doc_pages = [], {}
    for f in files:
        loaded = load_pdf(f)
        doc_pages[f.name] = len(loaded)
        pages.extend(loaded)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages)
    if not chunks:
        raise ValueError("No readable text found in these PDFs. Scanned (image-only) PDFs aren't supported yet.")
    return FAISS.from_documents(chunks, get_embeddings()), chunks, doc_pages
@st.cache_data(show_spinner=False, max_entries=48)
def render_pdf_page(data, page_index, dpi=130):
    # Draws one PDF page as a PNG so it can be shown in the source popup. Returns (png, total_pages, page_index)
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    doc = pymupdf.open(stream=data, filetype="pdf")
    page_index = min(max(int(page_index), 0), doc.page_count - 1)
    return doc[page_index].get_pixmap(dpi=dpi).tobytes("png"), doc.page_count, page_index
# ---------- LLM, prompts and chains ----------
# If the main model fails (rate limit etc.), LangChain retries the request on the backup model
llm = ChatGroq(model=PRIMARY_MODEL, temperature=0).with_fallbacks([ChatGroq(model=BACKUP_MODEL, temperature=0)])
parser = StrOutputParser()
# Rewrites the question into a clear search query (expands "RF", resolves "that")
rewriter = PromptTemplate.from_template(
    """Rewrite the latest question as a clear, standalone search query for a document.
The documents being searched are: {doc_names}
Use the chat history to resolve words like "that", "it" or "this".
Expand abbreviations into their full form, choosing the meaning that fits these documents.
Return only the rewritten question.
Chat history:
{history}
Latest question: {question}"""
) | llm | parser
# Answers only from the retrieved PDF text
answer_chain = PromptTemplate.from_template(
    """You are a helpful document assistant.
Answer ONLY from the provided PDF context. Do not add facts that are not in the context.
If the context only lists a topic without explaining it, say the document only mentions it.
If the context is insufficient, say you don't know based on the uploaded documents.
Mention the file names and page numbers you used.
Context:
{context}
Chat history:
{history}
Question: {question}"""
) | llm | parser
def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']}, page {d.metadata['page_label']}]\n{d.page_content}" for d in docs)
def retrieve(store, question, doc_names, history, max_distance):
    # 1) rewrite the question, 2) search, 3) keep only chunks that are close enough
    query = rewriter.invoke({"question": question, "doc_names": doc_names, "history": history})
    results = store.similarity_search_with_score(query, k=4)
    return [doc for doc, score in results if score < max_distance]
def stream_answer(inputs, status):
    # Yields the answer piece by piece; the "thinking" pill disappears when the first piece arrives
    first = True
    for piece in answer_chain.stream(inputs):
        if first:
            status.empty()
            first = False
        yield piece
def friendly_error(exc):
    text = str(exc)
    if "429" in text or "rate_limit" in text.lower():
        return "The free AI limit was reached. Please try again in a few minutes."
    if "401" in text or "api key" in text.lower():
        return "The AI key looks invalid. Check GROQ_API_KEY in your .env file."
    return f"The AI request failed. Details: {text[:200]}"
# ---------- Quiz generator ----------
class MCQ(BaseModel):
    question: str
    options: List[str] = Field(description="exactly 4 answer options")
    answer: int = Field(description="position of the correct option, 0 to 3")
    explanation: str = Field(description="one short sentence")
class Quiz(BaseModel):
    questions: List[MCQ]
quiz_parser = PydanticOutputParser(pydantic_object=Quiz)
quiz_chain = PromptTemplate.from_template(
    """Write {n} multiple-choice questions based ONLY on the text below.
Each question needs exactly 4 options and exactly one correct option.
Focus on the document's main subject and key ideas. Ignore headings, page numbers, author names, dates and filler text.
Every question must cover a different idea. Do not use outside knowledge.
{format_instructions}
Text:
{text}""",
    partial_variables={"format_instructions": quiz_parser.get_format_instructions()},
) | llm | quiz_parser
def shuffle_options(q):
    # LLMs like putting the right answer first, so shuffle and track its new position
    correct = q["options"][q["answer"]]
    random.shuffle(q["options"])
    q["answer"] = q["options"].index(correct)
    return q
def pick_chunks(chunks, k=4):
    # One random chunk from each of k zones, so the quiz covers the whole PDF
    if not chunks:
        return []
    size = max(1, math.ceil(len(chunks) / k))
    zones = [chunks[i:i + size] for i in range(0, len(chunks), size)][:k]
    return [random.choice(zone) for zone in zones if zone]
def make_quiz(chunks, n=5, tries=2):
    # More questions need more source text, so sample more chunks (4 to 8) as n grows
    picked = pick_chunks(chunks, k=min(8, max(4, n)))
    if not picked:
        return []
    text = "\n\n".join(c.page_content for c in picked)
    for _ in range(tries):
        try:
            result = quiz_chain.invoke({"n": n, "text": text})
        except OutputParserException:
            continue  # reply wasn't valid JSON, try again
        good = [q.model_dump() for q in result.questions if len(q.options) == 4 and 0 <= q.answer <= 3]
        if good:
            return [shuffle_options(q) for q in good[:n]]
    return []
# ---------- Chat storage ----------
CHATS_FILE = Path(__file__).with_name("chats.json")
def persist():
    # Optional: save chats to disk (only if SAVE_CHATS is True)
    if SAVE_CHATS:
        try:
            CHATS_FILE.write_text(json.dumps({"chats": st.session_state.chats, "active": st.session_state.active_chat}))
        except OSError:
            pass
def load_saved():
    if SAVE_CHATS and CHATS_FILE.exists():
        try:
            data = json.loads(CHATS_FILE.read_text())
            st.session_state.chats, st.session_state.active_chat = data["chats"], data.get("active")
        except (OSError, ValueError, KeyError):
            pass
def init_state():
    defaults = {"chats": {}, "active_chat": None, "confirm_delete": None, "quizzes": {}, "quiz_round": 0,
                "scores": [], "index_key": None, "failed_key": None, "index_error": "", "store": None,
                "chunks": [], "doc_pages": {}, "pdf_bytes": {}}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    if not isinstance(st.session_state.scores, list):  # older versions stored a dict
        st.session_state.scores = []
    if not st.session_state.chats:
        load_saved()
def new_chat():
    current = st.session_state.chats.get(st.session_state.active_chat)
    if current is not None and not current["messages"]:
        return  # the current chat is still empty, so reuse it
    chat_id = uuid.uuid4().hex[:10]
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
    st.session_state.active_chat = chat_id
    persist()
def delete_chat(chat_id):
    st.session_state.chats.pop(chat_id, None)
    if st.session_state.active_chat == chat_id:
        st.session_state.active_chat = None
    persist()
def delete_all_chats():  # used as a button callback, so it may reset the checkbox too
    st.session_state.chats, st.session_state.active_chat = {}, None
    st.session_state.sure_all = False
    persist()
def chat_to_markdown(chat):
    lines = [f"# {chat['title']}", ""]
    for m in chat["messages"]:
        lines += [f"**{'You' if m['role'] == 'user' else 'DocMind'}:** {m['content']}", ""]
        if m.get("sources"):
            lines += ["_Sources: " + ", ".join(f"{s[0]} p.{s[1]}" for s in m["sources"]) + "_", ""]
    return "\n".join(lines)
init_state()
if st.session_state.active_chat not in st.session_state.chats:
    st.session_state.active_chat = next(iter(st.session_state.chats), None)
    if st.session_state.active_chat is None:
        new_chat()
# ---------- Small UI helpers ----------
SUGGESTIONS = ["What are the key concepts?", "Explain the main topic simply", "Give me an example", "What are the important definitions?"]
THINKING = ["Reading your pages…", "Connecting the dots…", "Searching your notes…", "Putting your answer together…"]
QUIZ_WAIT = ["Brewing your questions…", "Shuffling the options…", "Picking the good bits…"]
VIEWS = ["💬 Chat", "📝 Quiz"]
def sources_from(found):
    # Unique (file, page label, page position) triples, sorted by file then page
    triples = {(d.metadata["source"], str(d.metadata["page_label"]), int(d.metadata.get("page", 0))) for d in found}
    return sorted(triples, key=lambda s: (s[0], s[2]))
def normalize_source(s):
    # Accepts old two-part sources too: (file, label) becomes (file, label, position)
    name, label = s[0], str(s[1])
    return name, label, int(s[2]) if len(s) > 2 else (int(label) - 1 if label.isdigit() else 0)
def short(name, limit=34):
    return name if len(name) <= limit else name[:limit - 1] + "…"
def chip_row(key):
    # A wrapping row of buttons (newer Streamlit has horizontal containers; older versions use plain CSS)
    try:
        return st.container(key=key, horizontal=True, wrap=True)
    except TypeError:
        return st.container(key=key)
def show_image(data):
    try:
        st.image(data, width="stretch")
    except Exception:  # older Streamlit
        st.image(data, use_container_width=True)
def step_page(delta):  # button callback for the page popup
    st.session_state["dlg_page"] = max(0, st.session_state.get("dlg_page", 0) + delta)
@st.dialog("Source page", width="large")
def page_dialog():
    # Shows the PDF page that an answer was based on, with Previous / Next buttons
    name = st.session_state.get("dlg_file")
    data = st.session_state.pdf_bytes.get(name)
    if not data:
        st.warning("This PDF is no longer loaded.")
        return
    try:
        png, total, index = render_pdf_page(data, st.session_state.get("dlg_page", 0))
    except ImportError:
        st.error("Page preview needs the pymupdf package. Run:  pip install pymupdf")
        return
    except Exception as exc:
        st.error(f"Couldn't draw this page: {exc}")
        return
    st.session_state["dlg_page"] = index
    st.markdown(f"**{name}** · page {index + 1} of {total}")
    show_image(png)
    prev_col, next_col = st.columns(2)
    prev_col.button("◀ Previous page", key="dlg_prev", on_click=step_page, args=(-1,), disabled=index == 0, **STRETCH)
    next_col.button("Next page ▶", key="dlg_next", on_click=step_page, args=(1,), disabled=index >= total - 1, **STRETCH)
def draw_message(m, loaded, idx, chat_id):
    # One chat bubble: text, clickable source chips, removed-file note and source passages
    role = m["role"]
    with st.chat_message(role, avatar=AI_URI if role == "assistant" else USER_URI):
        st.markdown(m["content"])
        if role != "assistant":
            return
        sources = [normalize_source(s) for s in m.get("sources", [])]
        if sources:
            with chip_row(f"srcrow_{chat_id}_{idx}"):
                for j, (name, label, pos) in enumerate(sources):
                    available = name in loaded
                    if st.button(f"📄 {short(name)} · p.{label}", key=f"src_{chat_id}_{idx}_{j}", disabled=not available,
                                 help="Open this page" if available else "This PDF is no longer loaded"):
                        st.session_state["dlg_file"], st.session_state["dlg_page"] = name, pos
                        page_dialog()
        gone = [f for f in m.get("files", []) if f not in loaded]
        if gone:
            st.markdown(f'<div class="small-note">⚠ Source no longer loaded: {html.escape(", ".join(gone))}</div>',
                        unsafe_allow_html=True)
        if m.get("passages"):
            with st.expander("View source passages"):
                passage_cards = []
                for name, page, passage_text in m["passages"]:
                    passage_cards.append(
                        f'<div class="source-passage">'
                        f'<div class="source-passage-title">{html.escape(name)} · p.{html.escape(str(page))}</div>'
                        f'<div class="source-passage-text">{html.escape(passage_text)}</div>'
                        f'</div>'
                    )
                # Render all passages together to avoid Streamlit adding block gaps between them.
                st.markdown("\n".join(passage_cards), unsafe_allow_html=True)
def profile_html():
    link = lambda href, svg, label: (f'<a class="pico" href="{href}" target="_blank" rel="noopener" '
                                     f'title="{label}"><img src="{svg_uri(svg)}" alt="{label}"></a>')
    return (f'<div class="pcard"><div class="prow"><div class="pav">{OWNER["initials"]}</div><div>'
            f'<div class="pname">{OWNER["name"]}</div><div class="muted">{OWNER["role"]}</div></div></div>'
            f'<div class="pbadge">{OWNER["badge"]}</div>'
            f'<div class="picons">{link(OWNER["github"], GITHUB_SVG, "GitHub")}'
            f'{link(OWNER["linkedin"], LINKEDIN_SVG, "LinkedIn")}{link(OWNER["email"], MAIL_SVG, "Email")}</div>'
            '<div><span class="tech">Streamlit</span><span class="tech">LangChain</span>'
            '<span class="tech">FAISS</span><span class="tech">Groq</span></div></div>')
@st.dialog("How DocMind works")
def about_dialog():
    st.markdown("DocMind answers **only from the PDFs you upload**. This technique is called RAG (retrieval-augmented generation).")
    st.markdown('<div class="flow"><span>1 Read</span><span>2 Chunk</span><span>3 Embed</span>'
                '<span>4 Search</span><span>5 Answer</span></div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Read:** PyPDF extracts the text page by page and keeps the page numbers.\n"
        "2. **Chunk:** the text is cut into ~1000-character pieces with some overlap.\n"
        "3. **Embed:** a MiniLM model turns each chunk into numbers that capture its meaning; FAISS stores them.\n"
        "4. **Search:** your question is rewritten into a clear query, then the closest chunks are found. "
        "Chunks that are too far away are dropped (the strictness slider).\n"
        "5. **Answer:** a Groq LLM answers using only those chunks and cites the file and page.")
    st.caption("Built with Streamlit, LangChain, FAISS, Hugging Face embeddings and Groq.")
# ---------- Interactive hero: clickable upload card and talking AI logo ----------
ORB_LINES = [
    "Fun fact: RAG stands for Retrieval-Augmented Generation. I look things up in your PDFs first, then answer, so no guessing from memory.",
    "Fun fact: the first neural network hardware, the Perceptron, was built in 1958 and filled a whole room. Your laptop is doing better today.",
    "Fun fact: I turn every paragraph of your PDF into a list of numbers. That is how I find the parts that match your question.",
    "Study tip: try the Quiz tab after reading your notes. Testing yourself beats re-reading.",
    "Fun fact: the word 'robot' comes from a Czech word for forced labour. Don't worry, I enjoy this job.",
    "Joke: I told my model a joke about overfitting. It memorized it and told it back word for word.",
    "Joke: my favourite exercise is gradient descent. I never reach the top, but I keep going downhill.",
    "Joke: why did the PDF get promoted? It always showed up on the right page.",
]
def next_orb_line():
    # First tap: a friendly greeting. Later taps: random facts and jokes (never the same one twice in a row)
    previous = st.session_state.get("orb_line")
    if not previous:
        return "Hello there! 👋 I'm DocMind. Upload a PDF and ask me anything about it."
    return random.choice([line for line in ORB_LINES if line != previous])
OPEN_UPLOADER_JS = """<script>
const doc = window.parent.document;
const zone = doc.querySelector('[data-testid="stFileUploaderDropzone"]');
if (zone) {
  zone.scrollIntoView({block: 'center'});
  zone.classList.add('pulse-upload');
  setTimeout(() => zone.classList.remove('pulse-upload'), 2600);
  const target = zone.querySelector('input[type="file"]') || zone.querySelector('button');
  if (target) target.click();
}
</script>"""
def run_js(js):
    # Runs a small script in the browser; newer Streamlit uses st.iframe, older versions use components.html
    if hasattr(st, "iframe"):
        st.iframe(js, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(js, height=0)
def draw_hero():
    left, right = st.columns([3, 1])
    with left:
        with st.container(key="herocard"):  # the big button's ::after stretches over the whole card
            st.markdown('<div class="small-note" style="letter-spacing:1px;text-transform:uppercase;color:#9b83ff;font-weight:700">Your AI workspace</div>'
                        '<div class="sora" style="font-size:22px;font-weight:600;margin:10px 0 6px">Turn documents into understanding.</div>'
                        '<div class="muted" style="font-size:13px;margin-bottom:14px">Upload PDFs, ask grounded questions, explore cited answers, or generate a quiz from your notes.</div>',
                        unsafe_allow_html=True)
            if st.button(":material/upload_file: Upload PDFs", key="herobtn"):
                st.session_state["open_upload"] = True
    with right:
        with st.container(key="orbcard"):  # tap anywhere on the tile for a greeting, fun fact or joke
            st.markdown(f'<div class="orb"><img class="keep" src="{AI_URI}"></div>', unsafe_allow_html=True)
            if st.button("Tap to say hi 👋", key="orbbtn"):
                st.session_state["orb_line"] = next_orb_line()
    if st.session_state.get("orb_line"):
        st.markdown(f'<div class="bubble"><img class="keep" src="{AI_URI}"><div class="said">{html.escape(st.session_state["orb_line"])}</div></div>',
                    unsafe_allow_html=True)
    if st.session_state.pop("open_upload", False):
        # Try to open the file dialog; if the browser blocks it, the uploader still pulses so people can find it
        run_js(OPEN_UPLOADER_JS)
# ---------- Quiz screen helpers ----------
def esc(text):
    return html.escape(str(text))
def feedback_html(q, pick):
    # Result card under a question (teal = right, pink = wrong, violet = skipped)
    correct, why = q["options"][q["answer"]], esc(q["explanation"])
    if pick is None:
        return f'<div class="fb skip"><b>Skipped</b><div class="why">The answer is <b>{esc(correct)}</b>. {why}</div></div>'
    if pick == correct:
        return f'<div class="fb good"><b>✓ Nailed it</b><div class="why">{esc(correct)}. {why}</div></div>'
    return (f'<div class="fb bad"><b>✗ Not quite</b><div class="why">Your answer: {esc(pick)}<br>'
            f'Correct answer: <b>{esc(correct)}</b><br>{why}</div></div>')
def mark_css(key, correct_index, chosen_index):
    # Outlines the right option in teal and a wrong pick in pink, after the quiz has been checked
    group = f'.st-key-{key} [role="radiogroup"] > label:nth-child'
    rules = (f'{group}({correct_index + 1}){{border-color:#5eead4 !important;background:rgba(94,234,212,.10) !important;'
             'box-shadow:0 0 22px rgba(94,234,212,.28) !important}')
    if chosen_index is not None and chosen_index != correct_index:
        rules += (f'{group}({chosen_index + 1}){{border-color:#f472b6 !important;background:rgba(244,114,182,.10) !important;'
                  'box-shadow:0 0 22px rgba(244,114,182,.28) !important}')
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)
def generate_quiz(pdf, n):
    # Builds a new quiz for one PDF and stores it. Returns True on success
    pdf_chunks = [c for c in st.session_state.chunks if c.metadata.get("source") == pdf]
    with st.spinner(random.choice(QUIZ_WAIT)):
        try:
            questions, error = make_quiz(pdf_chunks, n), None
        except Exception as exc:
            questions, error = [], friendly_error(exc)
    if questions:
        st.session_state.quiz_round += 1  # a new id gives new widget keys, so old answers don't carry over
        st.session_state.quizzes[pdf] = {
            "questions": questions,
            "id": st.session_state.quiz_round,
            "attempt": 0,
            "checked": False,
            "score": 0,
            "answers": {},  # durable answers, independent of temporary radio widget state
        }
        return True
    if error:
        st.error(error)
    else:
        st.warning("The model's reply couldn't be read as a quiz. Please try again.")
    return False
def quiz_go_to(nav_key, destination):
    """Select a quiz question/results option before Streamlit reruns."""
    st.session_state[nav_key] = destination


def quiz_save_answer(pdf, nav_key, answer_key, question_index, next_label):
    """Persist the answer before changing questions; widget keys can disappear on rerun."""
    quiz = st.session_state.quizzes.get(pdf)
    if not quiz:
        return
    quiz.setdefault("answers", {})
    selected = st.session_state.get(answer_key)
    quiz["answers"][str(question_index)] = selected
    # Automatically advance after an answer, except on the final question.
    if selected is not None and next_label:
        st.session_state[nav_key] = next_label


def submit_quiz(pdf, quiz, nav_key):
    """Score the durable answer map exactly once and open Results."""
    if quiz.get("checked"):
        st.session_state[nav_key] = "🏁 Results"
        return
    questions = quiz["questions"]
    answers = quiz.setdefault("answers", {})
    score = sum(
        answers.get(str(i)) == q["options"][q["answer"]]
        for i, q in enumerate(questions)
    )
    quiz["checked"] = True
    quiz["score"] = score
    st.session_state.scores.append({
        "pdf": pdf, "score": score, "total": len(questions),
        "time": datetime.now().strftime("%H:%M"),
    })
    st.session_state[nav_key] = "🏁 Results"


def draw_quiz(pdf, quiz):
    # Answers live in quiz["answers"], so Streamlit cannot lose them when a question widget is hidden.
    questions, checked = quiz["questions"], quiz["checked"]
    quiz.setdefault("answers", {})
    n, uid = len(questions), f'{quiz["id"]}_{quiz["attempt"]}'
    nav_key = f"quiz_nav_{uid}"
    labels = [f"Q{i + 1}" for i in range(n)] + ["🏁 Results"]

    # Each new quiz/attempt has a fresh id and therefore starts at Q1.
    st.session_state.setdefault(nav_key, "Q1")
    if st.session_state[nav_key] not in labels:
        st.session_state[nav_key] = "Q1"

    st.markdown(f'<div class="quiz-title">Practice quiz · {esc(pdf)}</div>', unsafe_allow_html=True)
    with st.container(key=f"quiznav_{uid}"):
        if hasattr(st, "segmented_control"):
            st.segmented_control("Quiz navigation", labels, key=nav_key, label_visibility="collapsed")
        else:
            st.radio("Quiz navigation", labels, key=nav_key, horizontal=True, label_visibility="collapsed")

    active = st.session_state.get(nav_key, "Q1")
    picks = [quiz["answers"].get(str(i)) for i in range(n)]

    if active != "🏁 Results":
        i = labels.index(active)
        q = questions[i]
        st.markdown(
            f'<div class="qcard"><span class="qbadge">QUESTION {i + 1} OF {n}</span>'
            f'<div class="qtext">{esc(q["question"])}</div>'
            f'<div class="qprog"><i style="width:{(i + 1) / n * 100:.0f}%"></i></div></div>',
            unsafe_allow_html=True,
        )
        answer_key = f"qz{uid}_{i}"
        saved_pick = quiz["answers"].get(str(i))
        saved_index = q["options"].index(saved_pick) if saved_pick in q["options"] else None
        st.radio(
            f"Question {i + 1}", q["options"], index=saved_index, key=answer_key,
            label_visibility="collapsed", disabled=checked,
            on_change=quiz_save_answer,
            args=(pdf, nav_key, answer_key, i, labels[i + 1] if i < n - 1 else None),
        )
        # Defensive sync supports Streamlit versions with different callback timing.
        current_pick = st.session_state.get(answer_key)
        if current_pick in q["options"]:
            quiz["answers"][str(i)] = current_pick
        picks = [quiz["answers"].get(str(j)) for j in range(n)]

        if checked:
            chosen = picks[i]
            mark_css(answer_key, q["answer"], q["options"].index(chosen) if chosen in q["options"] else None)
            st.markdown(feedback_html(q, chosen), unsafe_allow_html=True)

        if not checked:
            prev_col, next_col = st.columns(2)
            prev_col.button(
                "← Previous", key=f"prev_{uid}_{i}", **STRETCH, disabled=i == 0,
                on_click=quiz_go_to, args=(nav_key, labels[i - 1] if i > 0 else labels[0]),
            )
            if i == n - 1:
                next_col.button(
                    "Finish Quiz →", key=f"next_{uid}_{i}", **STRETCH,
                    on_click=submit_quiz, args=(pdf, quiz, nav_key),
                )
            else:
                next_col.button(
                    "Next →", key=f"next_{uid}_{i}", **STRETCH,
                    on_click=quiz_go_to, args=(nav_key, labels[i + 1]),
                )

    else:
        picks = [quiz["answers"].get(str(i)) for i in range(n)]
        answered = sum(p is not None for p in picks)
        if not checked:
            st.markdown(
                f'<div class="qcard"><span class="qbadge">READY?</span>'
                f'<div class="qtext">You answered {answered} of {n} questions</div>'
                f'<div class="qprog"><i style="width:{answered / n * 100:.0f}%"></i></div></div>',
                unsafe_allow_html=True,
            )
            st.caption("Unanswered questions will be marked as skipped.")
            st.button(
                "Check my answers",
                type="primary",
                key=f"check_{uid}",
                on_click=submit_quiz,
                args=(pdf, quiz, nav_key),
                **STRETCH,
                )
        else:
            score = quiz["score"]
            pct = round(score / n * 100) if n else 0
            message = ("Outstanding!" if pct >= 90 else "Great job!" if pct >= 70 else
                       "Good effort. Review and try again." if pct >= 50 else
                       "Keep going. Re-read the notes and retry.")
            pills = "".join(
                f'<span class="pill {"skip" if p is None else "good" if p == q["options"][q["answer"]] else "bad"}">Q{i + 1} '
                f'{"–" if p is None else "✓" if p == q["options"][q["answer"]] else "✗"}</span>'
                for i, (p, q) in enumerate(zip(picks, questions))
            )
            st.markdown(
                f'<div class="ring-wrap"><div class="ring" style="--p:{pct}"><div class="ring-in"><b>{score}/{n}</b>'
                f'<span class="muted">{pct}%</span></div></div><div><div class="ring-msg">{message}</div>{pills}</div></div>',
                unsafe_allow_html=True,
            )
            retry_col, new_col = st.columns(2)
            if retry_col.button("↺ Retry this quiz", key=f"retry_{uid}", **STRETCH):
                quiz.update(attempt=quiz["attempt"] + 1, checked=False, score=0, answers={})
                st.rerun()
            if new_col.button("✨ New quiz", key=f"new_{uid}", **STRETCH):
                if generate_quiz(pdf, st.session_state.get("quiz_n", 5)):
                    st.rerun()


        # Score history stays available on question and results screens.
    st.markdown('<div class="gap"></div><div class="hist-title">Your attempts</div>', unsafe_allow_html=True)
    history = [s for s in st.session_state.scores if s["pdf"] == pdf]
    if not history:
        st.markdown(
            '<div class="muted" style="font-size:13px">No attempts yet. Finish a quiz to see your score here.</div>',
            unsafe_allow_html=True,
        )
    best = max((s["score"] / s["total"] for s in history), default=0)
    for number, s in enumerate(history, start=1):
        ratio = s["score"] / s["total"]
        tag = '<span class="badge-best">Best</span>' if ratio == best and len(history) > 1 else ""
        st.markdown(
            f'<div class="hist-row{" best" if tag else ""}"><span>Attempt {number}</span>'
            f'<div class="hist-bar"><i style="width:{ratio * 100:.0f}%"></i></div>'
            f'<b>{s["score"]}/{s["total"]}</b><span class="muted">{s["time"]}</span>{tag}</div>',
            unsafe_allow_html=True,
        )
# ---------- Animated network background (step 1: test canvas) ----------
BG_JS = """<script>
const win = window.parent, doc = win.document;   // the iframe lives inside Streamlit, so we reach out to the real page
let canvas = doc.getElementById('docmind-bg');   // Streamlit reruns often, so reuse the canvas if it exists
if (!canvas) {
  canvas = doc.createElement('canvas');
  canvas.id = 'docmind-bg';
  // fixed = stays put while scrolling, z-index -1 = behind everything, pointer-events none = never blocks clicks
  canvas.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;';
  doc.body.appendChild(canvas);
}
const ctx = canvas.getContext('2d');
let W = 0, H = 0, particles = [], grad = null;
const mouse = {x: -9999, y: -9999};     // far off-screen = "no mouse yet"
const calm = win.matchMedia('(prefers-reduced-motion: reduce)').matches;  // respect "reduce motion" setting
const LINK = 95;                        // dots closer than this get a line
const PUSH_R = 160;                      // how far the mouse pushes dots
const GLOW_R = 240;                      // how far the mouse brightens lines

function waveY(x, t) {                    // centre line: a bigger, smoother S-curve
  return H * 0.5
    + Math.sin(x * 0.0036 + t * 0.00025) * H * 0.16
    + Math.sin(x * 0.0075 - t * 0.0004) * H * 0.035;
}

// How wide the ribbon is at x: it pinches and widens as it flows, like in your image
function ribbonWidth(x, t) {
  return H * (0.2 + 0.045 * Math.sin(x * 0.0023 - t * 0.00035 + 1.2));
}

function colorAt(x, a = 1) {             // teal -> purple by x position, with opacity a
  const k = Math.min(1, Math.max(0, x / W));
  const r = Math.round(94 + (167 - 94) * k);
  const g = Math.round(234 + (139 - 234) * k);
  const b = Math.round(212 + (250 - 212) * k);
  return `rgba(${r},${g},${b},${a})`;
}

function makeParticles() {
  const count = Math.round(Math.min(450, Math.max(140, W * H / 3200)));  // far more dots now
  particles = [];
  for (let i = 0; i < count; i++) {
    const stray = Math.random() < 0.12;  // ~12% wander far from the ribbon
    // adding 3 random numbers bunches results near 0: dense core, thin edges
    const core = (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;
    particles.push({
      x: (i + Math.random()) / count * W,          // evenly spread across the whole width
      offset: stray ? (Math.random() - 0.5) * 5 : core,
      phase: Math.random() * Math.PI * 2,
      speed: 0.3,
      size: stray ? 0.8 + Math.random() * 0.8 : 1 + Math.random() * 1.6,
      ox: 0, oy: 0,                                // how far the mouse has pushed this dot
      rx: 0, ry: 0,                                // final drawn position
    });
  }
}

function resize() {
  const dpr = Math.min(win.devicePixelRatio || 1, 2);
  W = win.innerWidth; H = win.innerHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  grad = ctx.createLinearGradient(0, 0, W, 0);     // colour for the flowing strands
  grad.addColorStop(0, '#5eead4'); grad.addColorStop(1, '#c084fc');
  makeParticles();
}

const RIBBONS = 28;                              // many fine strands, like silk threads

// Bends a point away from the mouse (same soft bump the strands use)
function bend(x, y) {
  const dx = x - mouse.x, dy = y - mouse.y;
  return y + Math.sign(dy) * 45 * Math.exp(-(dx * dx + dy * dy) / (2 * 130 * 130));
}

// The filled glow: stacked translucent shapes, wide to narrow, all following the ribbon
function drawBand(t) {
  ctx.shadowBlur = 0;
  ctx.globalCompositeOperation = 'lighter';     // layers add their light, so the middle glows most
  ctx.fillStyle = grad;                         // same teal -> purple gradient
  ctx.globalAlpha = 0.045;                       // each layer is faint; stacking builds the colour
  for (const s of [0.80, 0.95, 0.7, 0.45, 0.2]) {   // s = how much of the ribbon's width this layer covers
    ctx.beginPath();
    for (let x = -10; x <= W + 10; x += 10) {          // top edge, left to right
      const y = bend(x, waveY(x, t) - ribbonWidth(x, t) * s);
      if (x === -10) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    for (let x = W + 10; x >= -10; x -= 10) {          // bottom edge, right back to left
      ctx.lineTo(x, bend(x, waveY(x, t) + ribbonWidth(x, t) * s));
    }
    ctx.closePath();
    ctx.fill();                                        // colours the area between the two edges
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
}

function drawRibbons(t) {
  ctx.shadowBlur = 0;
  ctx.strokeStyle = grad;                        // teal -> purple
  ctx.globalCompositeOperation = 'lighter';      // overlapping strands add their light together
  for (let k = 0; k < RIBBONS; k++) {
    const u = (k / (RIBBONS - 1)) * 2 - 1;       // -1 = top strand ... +1 = bottom strand
    const core = 1 - Math.abs(u);                // 1 in the middle, 0 at the outer strands
    for (let pass = 0; pass < 2; pass++) {       // pass 0 = soft wide haze, pass 1 = thin bright thread
      ctx.lineWidth = pass ? 0.7 + core * 1.1 : 9;
      ctx.globalAlpha = pass ? 0.14 + core * 0.30 : 0.012 + core * 0.02;
      ctx.beginPath();
      for (let x = -10; x <= W + 10; x += 10) {
        const w = ribbonWidth(x, t);             // half-width of the bundle right here
        // strand position = centre line + its share of the width, which breathes in and out
        // per strand (cos) so the threads squeeze together and fan apart; sin adds a fine ripple
        let y = waveY(x, t)
          + u * w * (0.55 + 0.45 * Math.cos(x * 0.0042 + k * 0.22 + t * 0.0005))
          + Math.sin(x * 0.009 + k * 0.9 - t * 0.0006) * H * 0.012 * (1 - Math.abs(u) * 0.5);
        // the mouse bends nearby strands away
        const dx = x - mouse.x, dy = y - mouse.y;
        y += Math.sign(dy) * 45 * Math.exp(-(dx * dx + dy * dy) / (2 * 130 * 130));
        if (x === -10) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
}

function frame(t) {
  ctx.clearRect(0, 0, W, H);

    drawBand(t);      // the filled colour first...
    drawRibbons(t);   // ...then the silky strands on top

  // 2) Move every dot, then push it away from the mouse
  for (const p of particles) {
    if (!calm) p.x += p.speed;
    if (p.x > W + 20) p.x = -20;
    const by = waveY(p.x, t) + p.offset * ribbonWidth(p.x, t) * 1.1 + Math.sin(t * 0.001 + p.phase) * 6;
    let tx = 0, ty = 0;                              // where the mouse wants to push this dot
    const dx = p.x - mouse.x, dy = by - mouse.y, d2 = dx * dx + dy * dy;
    if (!calm && d2 < PUSH_R * PUSH_R && d2 > 1) {
      const d = Math.sqrt(d2), f = 1 - d / PUSH_R;   // closer to the mouse = stronger push
      tx = dx / d * f * 55; ty = dy / d * f * 55;
    }
    p.ox += (tx - p.ox) * 0.12;                      // ease toward the push (smooth, springy)
    p.oy += (ty - p.oy) * 0.12;
    p.rx = p.x + p.ox; p.ry = by + p.oy;
  }

  // 3) Lines between close dots; brighter near the mouse
  ctx.shadowBlur = 0; ctx.lineWidth = 1;
  const L2 = LINK * LINK;
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const a = particles[i], b = particles[j];
      const dx = a.rx - b.rx, dy = a.ry - b.ry, d2 = dx * dx + dy * dy;
      if (d2 < L2) {                                 // cheap check first, sqrt only if close
        const d = Math.sqrt(d2);
        let fade = (1 - d / LINK) * 0.45;
        const mx = (a.rx + b.rx) / 2 - mouse.x, my = (a.ry + b.ry) / 2 - mouse.y;
        const md = Math.sqrt(mx * mx + my * my);
        if (md < GLOW_R) fade += (1 - md / GLOW_R) * 0.5;   // mouse makes nearby lines glow
        ctx.strokeStyle = colorAt((a.rx + b.rx) / 2, fade);
        ctx.beginPath(); ctx.moveTo(a.rx, a.ry); ctx.lineTo(b.rx, b.ry); ctx.stroke();
      }
    }
  }

  // 4) Dots on top (only the bigger ones get a glow, since glow is expensive to draw)
  for (const p of particles) {
    const c = colorAt(p.rx);
    ctx.shadowBlur = p.size > 2 ? 14 : 0; ctx.shadowColor = c;
    ctx.fillStyle = c;
    ctx.beginPath(); ctx.arc(p.rx, p.ry, p.size, 0, Math.PI * 2); ctx.fill();
  }
  win.__dmFrame = win.requestAnimationFrame(frame);
}

// Clean up old loop/listeners (Streamlit reruns this script), then start fresh
if (win.__dmFrame) win.cancelAnimationFrame(win.__dmFrame);
if (win.__dmResize) win.removeEventListener('resize', win.__dmResize);
if (win.__dmMove) win.removeEventListener('mousemove', win.__dmMove);
if (win.__dmLeave) doc.documentElement.removeEventListener('mouseleave', win.__dmLeave);
win.__dmResize = resize;
win.__dmMove = e => { mouse.x = e.clientX; mouse.y = e.clientY; };         // remember the cursor
win.__dmLeave = () => { mouse.x = -9999; mouse.y = -9999; };               // cursor left the page
win.addEventListener('resize', resize);
win.addEventListener('mousemove', win.__dmMove);
doc.documentElement.addEventListener('mouseleave', win.__dmLeave);
resize();
win.__dmFrame = win.requestAnimationFrame(frame);
</script>"""
with st.container(key="bgjs"):   # wrapper so the CSS above can find and hide the iframe
    run_js(BG_JS)

# ---------- Theme switch (fixed at the top right) ----------
with st.container(key="themebox"):
    st.toggle(":material/light_mode: Light mode", key="light")
# ---------- Sidebar layout (containers keep the order fixed) ----------
side_brand = st.sidebar.container()
side_chats = st.sidebar.container()
side_docs = st.sidebar.container()
side_settings = st.sidebar.container()
side_profile = st.sidebar.container(key="profilebox")
with side_brand:
    st.markdown(f'<div class="brand-row"><img class="keep" src="{AI_URI}" width="42"><div><div class="brand-name">DocMind</div>'
                '<div class="brand-sub">AI Document Assistant</div></div></div>', unsafe_allow_html=True)
with side_chats:
    if st.button(":material/add: New chat", key="newchat", **STRETCH):
        new_chat()
        st.rerun()
    st.markdown('<div class="section-label">Recent chats</div>', unsafe_allow_html=True)
    for chat_id, chat in reversed(list(st.session_state.chats.items())):
        if st.session_state.confirm_delete == chat_id:  # second step: confirm the delete
            c1, c2, c3 = st.columns([2.2, 1, 1])
            c1.markdown('<div class="small-note" style="padding-top:9px">Delete this chat?</div>', unsafe_allow_html=True)
            if c2.button("Yes", key=f"yes_{chat_id}"):
                st.session_state.confirm_delete = None
                delete_chat(chat_id)
                st.rerun()
            if c3.button("No", key=f"no_{chat_id}"):
                st.session_state.confirm_delete = None
                st.rerun()
        else:
            row = st.columns([5, 1])
            is_active = chat_id == st.session_state.active_chat
            if row[0].button(chat["title"][:28] or "New chat", key=f"select_{chat_id}", type="primary" if is_active else "secondary", **STRETCH):
                st.session_state.active_chat = chat_id
                st.rerun()
            if row[1].button(":material/delete:", key=f"delete_{chat_id}", help="Delete this chat"):
                st.session_state.confirm_delete = chat_id
                st.rerun()
with side_docs:
    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop your PDFs here or browse files", type=["pdf"], accept_multiple_files=True,
                                      help="Upload one or more text-based PDFs.", label_visibility="collapsed")
# ---------- Index the PDFs when the uploaded files change ----------
if uploaded_files:
    key = [(f.name, f.size) for f in uploaded_files]
    if st.session_state.index_key != key and st.session_state.failed_key != key:
        with st.spinner("Reading and indexing your PDFs…"):
            try:
                store, chunks, doc_pages = build_index(uploaded_files)
                st.session_state.update(index_key=key, store=store, chunks=chunks, doc_pages=doc_pages, failed_key=None,
                                        index_error="", quizzes={}, pdf_bytes={f.name: f.getvalue() for f in uploaded_files})
            except Exception as exc:  # remember the failure so it isn't retried on every click
                st.session_state.update(failed_key=key, index_error=str(exc), index_key=None, store=None,
                                        chunks=[], doc_pages={}, pdf_bytes={})
else:
    st.session_state.update(index_key=None, failed_key=None, index_error="", store=None, chunks=[], doc_pages={}, pdf_bytes={})
doc_pages = st.session_state.doc_pages
chunks = st.session_state.chunks
active_id = st.session_state.active_chat
active_chat = st.session_state.chats[active_id]
with side_docs:  # status card: one compact list instead of repeating big cards
    if doc_pages:
        rows = "".join(f'<div class="doc-row"><span class="doc-name" title="{html.escape(n)}">📕 {html.escape(n)}</span>'
                       f'<span class="muted">{p} p</span></div>' for n, p in doc_pages.items())
        st.markdown(f'<div class="dm-card"><div class="small-note" style="margin-bottom:4px">🟢 Ready</div>{rows}'
                    f'<div class="stats"><div><b>{len(doc_pages)}</b><span>Docs</span></div>'
                    f'<div><b>{sum(doc_pages.values())}</b><span>Pages</span></div>'
                    f'<div><b>{len(chunks)}</b><span>Chunks</span></div></div></div>', unsafe_allow_html=True)
    else:
        st.caption("Upload a PDF to see its indexing status.")
with side_settings:
    with st.expander("Settings"):
        strict = st.slider("Answer strictness", 1, 10, 5, help="Higher strictness means fewer, more relevant sources.")
        st.caption("Stricter = fewer, more relevant sources")
        sure = st.checkbox("Yes, I want to delete all my chats", key="sure_all")
        st.button("Delete all chats", disabled=not sure, on_click=delete_all_chats)
max_distance = 2.0 - 0.1 * strict  # strictness 5 gives the 1.5 cutoff we tested
with side_profile:
    st.markdown(profile_html(), unsafe_allow_html=True)
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)  # gap between the card and the button
    if st.button("About this project", key="about", **STRETCH):
        about_dialog()
# ---------- Main area: header and view switch ----------
st.markdown('<div class="hero-title">Your documents, answered.</div>'
            '<div class="subhead">Ask anything, get answers with page-level sources, or turn your notes into a quiz.</div>',
            unsafe_allow_html=True)
if st.session_state.failed_key is not None:
    st.error(f"Couldn't index the uploaded PDFs: {st.session_state.index_error}")
nav_left, nav_right = st.columns([4, 1])
with nav_left:
    if hasattr(st, "segmented_control"):
        view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed")
    else:
        view = st.radio("View", VIEWS, horizontal=True, key="view", label_visibility="collapsed")
    view = view or st.session_state.get("last_view", VIEWS[0])  # clicking the active pill must not switch views
    st.session_state["last_view"] = view
# ---------- Chat view ----------
if view == VIEWS[0]:
    with nav_right:  # chat tools: rename, export, clear
        with st.popover(":material/more_horiz: Chat tools", **STRETCH):
            new_name = st.text_input("Rename chat", value=active_chat["title"], key=f"rename_{active_id}")
            if st.button("Save name", key=f"save_{active_id}"):
                active_chat["title"] = new_name.strip()[:40] or active_chat["title"]
                persist()
                st.rerun()
            st.download_button(":material/download: Export chat (.md)", data=chat_to_markdown(active_chat),
                               file_name=re.sub(r"[^\w-]+", "_", active_chat["title"])[:30] + ".md",
                               mime="text/markdown", **STRETCH)
            if st.button("Clear messages", key=f"clear_{active_id}", **STRETCH):
                active_chat["messages"] = []
                active_chat["title"] = "New chat"
                persist()
                st.rerun()
    # The input is at page level (outside tabs), so it stays pinned to the bottom of the screen
    typed = st.chat_input("Ask something about your PDFs…" if doc_pages else "Upload a PDF in the sidebar to start",
                          disabled=not doc_pages)
    question = typed or st.session_state.pop("pending_question", None)
    if not active_chat["messages"] and not question:
        draw_hero()
    for index, m in enumerate(active_chat["messages"]):
        draw_message(m, doc_pages, index, active_id)
    if doc_pages and not active_chat["messages"] and not question:  # suggestion buttons (hidden while answering)
        st.markdown("**Try asking**")
        for col, text in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
            if col.button("✨ " + text, key=f"sug_{active_id}_{text}", **STRETCH):
                st.session_state["pending_question"] = text
                st.rerun()
    if question and doc_pages:
        draw_message({"role": "user", "content": question}, doc_pages, len(active_chat["messages"]), active_id)
        # History comes from earlier turns only; the new pair is saved after a successful answer
        history = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in active_chat["messages"][-6:]) or "(none)"
        with st.chat_message("assistant", avatar=AI_URI):
            status = st.empty()
            status.markdown('<span class="think"><span class="dot"></span><span class="dot"></span>'
                            f'<span class="dot"></span> {random.choice(THINKING)}</span>', unsafe_allow_html=True)
            try:
                found = retrieve(st.session_state.store, question, ", ".join(doc_pages), history, max_distance)
                if not found:
                    status.empty()
                    answer = "I couldn't find anything relevant to that in your PDFs."
                    st.markdown(answer)
                else:
                    answer = st.write_stream(stream_answer(
                        {"context": format_docs(found), "question": question, "history": history}, status))
                    status.empty()
            except Exception as exc:
                status.empty()
                st.error(friendly_error(exc))
                st.stop()  # nothing is saved, so the history never keeps a question without an answer
        sources = sources_from(found)
        if active_chat["title"] == "New chat":
            active_chat["title"] = question.strip()[:32] + ("…" if len(question.strip()) > 32 else "")
        active_chat["messages"].append({"role": "user", "content": question})
        active_chat["messages"].append({
            "role": "assistant", "content": answer, "sources": sources,
            "files": sorted({s[0] for s in sources}),
            "passages": [(d.metadata["source"], str(d.metadata["page_label"]), d.page_content[:400]) for d in found],
        })
        persist()
        st.rerun()
# ---------- Quiz view ----------
else:
    if not doc_pages:
        st.info("Upload one or more PDFs in the sidebar to generate a quiz.")
    else:
        q1, q2 = st.columns([3, 2])
        selected_pdf = q1.selectbox("Choose a PDF", list(doc_pages), key="quiz_pdf")
        n_questions = q2.slider("Number of questions", 3, 10, 5, key="quiz_n")
        quiz = st.session_state.quizzes.get(selected_pdf)
        if quiz is not None and not isinstance(quiz, dict):  # a quiz saved by an older version
            st.session_state.quizzes.pop(selected_pdf, None)
            quiz = None
        if st.button("✨ New quiz" if quiz else "✨ Generate quiz", type="primary"):
            if generate_quiz(selected_pdf, n_questions):
                st.rerun()
        if quiz:
            draw_quiz(selected_pdf, quiz)
st.markdown('<div style="text-align:center;color:#9ba6ca;font-size:10px;padding:22px 0 8px">'
            'DocMind · Answers grounded in your documents · Built with Streamlit, LangChain and FAISS</div>',
            unsafe_allow_html=True)
