"""Visual theme.

Matches the TruthGuard design language: a near-black navy ground with a faint dot grid,
Instrument Serif for display type at a large optical size, Inter for everything else, and
a small set of saturated accents used only to carry meaning (teal for confirmed, amber
for caution, coral for a problem).

Streamlit's defaults read as a data tool, which is wrong for a product about clothes, so
the overrides here are extensive. Keeping them in one stylesheet rather than scattered
`unsafe_allow_html` calls means the design is legible in one place and the app module
stays about behaviour.
"""

from __future__ import annotations

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:        #06080e;
    --text:      #e6edf7;
    --muted:     rgba(255,255,255,0.45);
    --dim:       rgba(255,255,255,0.62);
    --line:      rgba(255,255,255,0.08);
    --line-soft: rgba(255,255,255,0.05);
    --surface:   rgba(255,255,255,0.02);
    --teal:      #39d2c0;
    --purple:    #bc8cff;
    --amber:     #e3b341;
    --coral:     #ff6b6b;
}

/* Faint dot grid over the whole ground, exactly as the reference does it. */
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(circle at 1px 1px, rgba(255,255,255,0.045) 1px, transparent 0);
    background-size: 48px 48px;
}

html, body, [class*="css"], .stMarkdown, p, li, label, input, select, textarea {
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    color: var(--text);
}

.block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1280px; }
#MainMenu, footer, header { visibility: hidden; }
img { border-radius: 10px; }

/* ---- Nav --------------------------------------------------------------- */

.nav {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.85rem 0 1rem;
    border-bottom: 1px solid var(--line-soft);
    margin-bottom: 3.2rem;
}
.nav .mark {
    width: 22px; height: 22px; border-radius: 6px;
    background: conic-gradient(from 210deg, #39d2c0, #bc8cff, #ff8c66, #39d2c0);
    flex: none;
}
.nav .brand { font-weight: 600; font-size: 0.95rem; letter-spacing: -0.01em; }
.nav .spacer { flex: 1; }
.nav .links { display: flex; gap: 1.6rem; }
.nav .links span { font-size: 0.84rem; color: var(--muted); }

/* ---- Hero -------------------------------------------------------------- */

.eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.185em;
    text-transform: uppercase;
    color: var(--teal);
    font-weight: 500;
    margin-bottom: 1.1rem;
}
.display {
    font-family: 'Instrument Serif', ui-serif, Georgia, serif;
    font-weight: 400;
    font-size: clamp(2.6rem, 6vw, 5.25rem);
    line-height: 0.98;
    letter-spacing: -0.01em;
    margin: 0 0 1.4rem;
}
.display .dim { color: rgba(255,255,255,0.38); font-style: italic; }
.standfirst {
    font-size: 1rem; line-height: 1.65; color: var(--dim);
    max-width: 62ch; font-weight: 300; margin: 0 0 1rem;
}
.standfirst strong { color: var(--text); font-weight: 500; }

/* ---- Section headings -------------------------------------------------- */

.sect { display: flex; align-items: center; gap: 1rem; margin: 2.6rem 0 1.2rem; }
.sect .t {
    font-family: 'Instrument Serif', ui-serif, Georgia, serif;
    font-size: 1.75rem; letter-spacing: -0.01em; white-space: nowrap;
}
.sect .rule { flex: 1; height: 1px; background: var(--line-soft); }
.sect .count { font-size: 0.78rem; color: var(--muted); white-space: nowrap; }

/* ---- Stat tiles -------------------------------------------------------- */

.stats { display: flex; gap: 0.75rem; flex-wrap: wrap; }
.tile {
    flex: 1 1 150px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1rem 1.15rem;
}
.tile .k {
    font-size: 0.66rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.45rem;
}
.tile .v {
    font-family: 'Instrument Serif', ui-serif, Georgia, serif;
    font-size: 1.85rem; line-height: 1; letter-spacing: -0.01em;
}
.tile.teal   { background: rgba(57,210,192,0.07);  border-color: rgba(57,210,192,0.25); }
.tile.teal .v   { color: var(--teal); }
.tile.amber  { background: rgba(227,179,65,0.07);  border-color: rgba(227,179,65,0.25); }
.tile.amber .v  { color: var(--amber); }
.tile.purple { background: rgba(188,140,255,0.07); border-color: rgba(188,140,255,0.25); }
.tile.purple .v { color: var(--purple); }

/* ---- Cards ------------------------------------------------------------- */

.card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.25rem 1.35rem 1.15rem;
    height: 100%;
    transition: border-color 160ms ease, background 160ms ease;
}
.card:hover { border-color: rgba(255,255,255,0.16); background: rgba(255,255,255,0.035); }
.card .rank {
    font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--teal); margin-bottom: 0.55rem;
}
.card h3 {
    font-family: 'Instrument Serif', ui-serif, Georgia, serif;
    font-weight: 400; font-size: 1.65rem; line-height: 1.1;
    letter-spacing: -0.01em; margin: 0 0 0.65rem; text-transform: capitalize;
}
.card .why {
    font-size: 0.88rem; line-height: 1.65; color: var(--dim);
    font-weight: 300; margin: 0.6rem 0 0.8rem;
}
.card .fix {
    font-size: 0.82rem; line-height: 1.55; color: var(--amber);
    background: rgba(227,179,65,0.06); border: 1px solid rgba(227,179,65,0.2);
    border-radius: 8px; padding: 0.5rem 0.7rem; margin: 0.4rem 0; font-weight: 300;
}
.card .credit {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.66rem; color: var(--muted);
    border-top: 1px solid var(--line-soft);
    padding-top: 0.65rem; margin-top: 0.9rem; line-height: 1.6; word-break: break-word;
}

/* ---- Chips ------------------------------------------------------------- */

.chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.chip {
    font-size: 0.7rem; padding: 0.22rem 0.6rem; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted);
    background: rgba(255,255,255,0.02); white-space: nowrap;
}
.chip.teal {
    color: var(--teal); border-color: rgba(57,210,192,0.3);
    background: rgba(57,210,192,0.08);
}
.chip.purple {
    color: var(--purple); border-color: rgba(188,140,255,0.3);
    background: rgba(188,140,255,0.08);
}

.voted {
    font-size: 0.74rem; letter-spacing: 0.04em; color: var(--teal);
    background: rgba(57,210,192,0.07); border: 1px solid rgba(57,210,192,0.22);
    border-radius: 8px; padding: 0.45rem 0.7rem; text-align: center;
    margin-top: 0.45rem;
}

/* ---- Notices ----------------------------------------------------------- */

.notice {
    border: 1px solid var(--line); background: var(--surface);
    border-radius: 12px; padding: 0.85rem 1.05rem; margin: 0.6rem 0 1.1rem;
    font-size: 0.85rem; line-height: 1.62; color: var(--dim); font-weight: 300;
}
.notice strong { color: var(--text); font-weight: 500; }
.notice.amber { background: rgba(227,179,65,0.06); border-color: rgba(227,179,65,0.28); }
.notice.coral { background: rgba(255,107,107,0.06); border-color: rgba(255,107,107,0.28); }
.notice.teal  { background: rgba(57,210,192,0.06); border-color: rgba(57,210,192,0.25); }

/* ---- Sidebar ----------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #04060b; border-right: 1px solid var(--line-soft);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.sb-h {
    font-size: 0.66rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 0.9rem; font-weight: 500;
}
.sb-big {
    font-family: 'Instrument Serif', ui-serif, Georgia, serif;
    font-size: 3.1rem; line-height: 1; color: var(--teal); margin-bottom: 0.15rem;
}
.sb-sub {
    font-size: 0.66rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 1.5rem;
}
.sb-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0; border-bottom: 1px solid var(--line-soft); font-size: 0.8rem;
}
.sb-row .k { color: var(--muted); }
.sb-row .v {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem; color: var(--text);
}
.sb-row .v.off { color: var(--muted); }
.sb-row .v.on  { color: var(--teal); }

/* ---- Streamlit controls ------------------------------------------------ */

.stButton > button {
    background: #ffffff; color: #06080e; border: none; border-radius: 8px;
    font-weight: 500; font-size: 0.9rem; padding: 0.65rem 1.25rem;
    transition: opacity 140ms ease; width: 100%;
}
.stButton > button:hover { opacity: 0.88; color: #06080e; }
.stButton > button:focus:not(:active) { color: #06080e; }

div[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--line) !important;
    border-radius: 12px !important;
}
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
}
.stTextInput label, .stSelectbox label, .stSlider label,
div[data-testid="stFileUploader"] label {
    font-size: 0.66rem !important; letter-spacing: 0.16em !important;
    text-transform: uppercase !important; color: var(--muted) !important;
    font-weight: 500 !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--teal) !important; }
div[data-testid="stSpinner"] > div { border-top-color: var(--teal) !important; }
</style>
"""

NAV = """
<div class="nav">
  <div class="mark"></div>
  <div class="brand">Atelier</div>
  <div class="spacer"></div>
  <div class="links">
    <span>Body geometry</span><span>Cross-cultural</span><span>Provenance</span>
  </div>
</div>
"""

HERO = """
<div class="eyebrow">Body-geometry matching · Western ↔ Indian wardrobes</div>
<div class="display">Outfits chosen by<br><span class="dim">shape, not ethnicity</span></div>
<p class="standfirst">
An A-line anarkali flatters a pear frame for the same structural reasons an A-line dress
does. This retrieves outfits worn by people who share your proportions, then
<strong>explains the cut</strong> — and tells you exactly what to adjust when it is close
but not right.
</p>
"""


def sect(title: str, count: str = "") -> str:
    tail = f'<span class="count">{count}</span>' if count else ""
    return f'<div class="sect"><span class="t">{title}</span><span class="rule"></span>{tail}</div>'


def tile(key: str, value: str, tone: str = "") -> str:
    return f'<div class="tile {tone}"><div class="k">{key}</div><div class="v">{value}</div></div>'


def tiles(items: list[str]) -> str:
    return f'<div class="stats">{"".join(items)}</div>'


def chip(text: str, tone: str = "") -> str:
    return f'<span class="chip {tone}">{text}</span>'


def chips(items: list[str]) -> str:
    return f'<div class="chips">{"".join(items)}</div>'


def notice(text: str, tone: str = "") -> str:
    return f'<div class="notice {tone}">{text}</div>'


def sb_row(key: str, value: str, tone: str = "") -> str:
    return (
        f'<div class="sb-row"><span class="k">{key}</span>'
        f'<span class="v {tone}">{value}</span></div>'
    )
