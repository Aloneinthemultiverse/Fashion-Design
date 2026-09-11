"""Streamlit user interface.

Runs the pipeline in-process rather than against the HTTP API: Streamlit reruns the whole
script on every interaction, so polling a local server would add a second process and a
second failure mode for no benefit. The API exists for real clients; this exists to make
the system usable and demonstrable today.

The body-shape confirmation step is the one piece of interaction design that carries real
weight. Single-photo inference is corrupted by pose, camera angle and loose clothing, so a
low-confidence reading is presented as a question rather than a verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from fashion.adapters.jobs_store import InMemoryJobStore
from fashion.config import load_settings
from fashion.core.feedback import FeedbackEvent, FeedbackLog, Verdict
from fashion.core.jobs import Job, StageName
from fashion.core.models import BodyShape, Culture, Occasion, UserQuery
from fashion.factory import (
    build_embedder,
    build_generation,
    build_store,
    build_tryon,
    build_vision,
    load_corpus,
)
from fashion.pipeline.recommend import RecommendationPipeline
from fashion.ui import theme

SHAPE_LABELS = {
    BodyShape.HOURGLASS: "Hourglass — balanced shoulders and hips, defined waist",
    BodyShape.PEAR: "Pear — hips wider than shoulders",
    BodyShape.RECTANGLE: "Rectangle — balanced, little waist definition",
    BodyShape.APPLE: "Apple — weight carried around the midsection",
    BodyShape.INVERTED_TRIANGLE: "Inverted triangle — shoulders wider than hips",
}

ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]


@st.cache_resource
def load_engine() -> tuple[Any, Any, Any, Any, Any, int]:
    """Build providers and seed the index once per session.

    Cached because loading CLIP weights and rebuilding the index on every rerun would
    make the app unusable -- Streamlit reruns the script on every widget interaction.
    """
    settings = load_settings()
    vision = build_vision(settings)
    embedder = build_embedder(settings)
    store = build_store(settings)
    generator = build_generation(settings)
    tryon = build_tryon(settings)
    load_corpus(embedder, store, settings)
    return vision, embedder, store, generator, tryon, store.count()


def stage_result(job: Job, name: StageName) -> dict[str, Any]:
    return job.stage(name).result or {}


def render_sidebar(settings: Any, corpus_size: int, generator: Any, tryon: Any) -> None:
    with st.sidebar:
        st.markdown('<div class="sb-h">Corpus</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sb-big">{corpus_size}</div><div class="sb-sub">outfits indexed</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-h">Backends</div>', unsafe_allow_html=True)
        rows = [
            theme.sb_row("Vision", settings.vision_provider),
            theme.sb_row("Embeddings", settings.embed_provider),
            theme.sb_row("Store", settings.store_provider),
            theme.sb_row(
                "Generation",
                "live" if generator.available else "off",
                "on" if generator.available else "off",
            ),
            theme.sb_row(
                "Try-on",
                "live" if tryon.available else "off",
                "on" if tryon.available else "off",
            ),
        ]
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

        if settings.vision_provider == "fake":
            st.markdown(
                theme.notice(
                    "<strong>Placeholder labels.</strong> The fake vision model is "
                    "active, so every garment tag and body reading below is "
                    "meaningless. Set a Gemini key and "
                    "<code>FASHION_VISION_PROVIDER=gemini</code> for real results.",
                    "amber",
                ),
                unsafe_allow_html=True,
            )
        if not generator.available:
            st.markdown(
                theme.notice(
                    "No generation backend, so you are seeing real photographed "
                    "outfits rather than synthesised previews."
                ),
                unsafe_allow_html=True,
            )


def render_body_reading(analyze: dict[str, Any]) -> None:
    st.markdown(theme.sect("Your proportions"), unsafe_allow_html=True)

    confidence = float(analyze.get("confidence", 0))
    uncertain = bool(analyze.get("needs_confirmation"))
    st.markdown(
        theme.tiles(
            [
                theme.tile("Shape", str(analyze["shape"]).replace("_", " ").title(), "teal"),
                theme.tile("Build", str(analyze["build"]).title()),
                theme.tile("Height", str(analyze["height_band"]).title()),
                theme.tile("Confidence", f"{confidence:.0%}", "amber" if uncertain else "purple"),
            ]
        ),
        unsafe_allow_html=True,
    )

    if not uncertain:
        return

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        theme.notice(
            "<strong>This reading is uncertain.</strong> Pose, camera angle and loose "
            "clothing all distort measured proportions. Correcting it takes one click "
            "and noticeably improves what comes back.",
            "amber",
        ),
        unsafe_allow_html=True,
    )
    current = BodyShape(analyze["shape"])
    left, right = st.columns([3, 1])
    with left:
        chosen = st.selectbox(
            "Which is closer?",
            list(BodyShape),
            index=list(BodyShape).index(current),
            format_func=lambda s: SHAPE_LABELS[s],
            label_visibility="collapsed",
        )
    with right:
        if chosen is not None and st.button("Use this"):
            st.session_state["confirmed_shape"] = chosen.value
            st.rerun()


def render_feedback(rec: dict[str, Any], analyze: dict[str, Any], query_text: str) -> None:
    """Thumbs up/down for one recommendation.

    Records whether the body shape was user-confirmed alongside the verdict. Without
    that flag a thumbs-down is uninterpretable: it could mean the outfit was wrong, or
    that the shape it was chosen for was wrong, and those need different fixes.
    """
    log = feedback_log()
    outfit_id = str(rec["outfit_id"])
    given = st.session_state.setdefault("feedback_given", {})

    if outfit_id in given:
        st.markdown(
            f'<div class="voted">Noted &mdash; {given[outfit_id]}</div>',
            unsafe_allow_html=True,
        )
        return

    up_col, down_col = st.columns(2)
    for column, verdict, label in (
        (up_col, Verdict.UP, "Works"),
        (down_col, Verdict.DOWN, "Not for me"),
    ):
        with column, st.container():
            if st.button(label, key=f"fb-{verdict.value}-{outfit_id}"):
                log.record(
                    FeedbackEvent(
                        outfit_id=outfit_id,
                        verdict=verdict,
                        body_shape=str(analyze["shape"]),
                        shape_was_confirmed=bool(analyze.get("user_confirmed")),
                        query_text=query_text,
                        matched_on=tuple(rec.get("matched_on", ())),
                    )
                )
                given[outfit_id] = label.lower()
                st.rerun()


@st.cache_resource
def celebrity_profiles() -> dict[str, Any]:
    """Profiles used to resolve a named body reference."""
    from fashion.core.dataset import CelebrityRepository

    settings = load_settings()
    return CelebrityRepository(settings.data_dir / "celebrity_profiles.jsonl").index()


@st.cache_resource
def feedback_log() -> FeedbackLog:
    return FeedbackLog(load_settings().data_dir / "feedback.jsonl")


def render_card(rec: dict[str, Any], position: int) -> None:
    rank = ORDINALS[position] if position < len(ORDINALS) else f"#{position + 1}"
    attributes = theme.chips(
        [
            theme.chip(str(rec["silhouette"]).replace("_", "-"), "teal"),
            theme.chip(str(rec["neckline"]).replace("_", "-")),
            theme.chip(str(rec["culture"]), "purple"),
            theme.chip(str(rec["occasion"])),
        ]
    )
    fixes = "".join(f'<div class="fix">{f}</div>' for f in rec.get("adjustments", []))
    matched_list = [str(m) for m in rec.get("matched_on", [])]
    gap = next((m.split(":", 1)[1] for m in matched_list if m.startswith("gap:")), None)
    matched = ", ".join(matched_list)

    # A gap-filled item is not the Nth best match; it was retrieved to cover something
    # the request asked for and the first pass missed, so it says that instead.
    header = (
        f'<div class="rank">retrieved for &ldquo;{gap}&rdquo;</div>'
        if gap
        else f'<div class="rank">{rank} choice &middot; {rec["score"]:.4f}</div>'
    )
    st.markdown(
        f'<div class="card">'
        f"{header}"
        f"<h3>{rec['garment_type']}</h3>"
        f"{attributes}"
        f'<div class="why">{rec["rationale"]}</div>'
        f"{fixes}"
        f'<div class="credit">matched on {matched}<br>{rec.get("license", "")}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_results(job: Job, photo: bytes, query_text: str = "") -> None:
    analyze = stage_result(job, StageName.ANALYZE)
    if not analyze:
        st.markdown(
            theme.notice(job.error or "Analysis did not complete.", "coral"),
            unsafe_allow_html=True,
        )
        return

    render_body_reading(analyze)

    retrieve = stage_result(job, StageName.RETRIEVE)
    recommendations = retrieve.get("recommendations", [])
    if not recommendations:
        st.markdown(
            theme.notice(
                "<strong>Nothing matched.</strong> Most often this means the corpus "
                "holds no outfits from that wardrobe yet. Try setting Wardrobe to "
                "<code>any</code>, or widening the style and occasion filters."
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        theme.sect(
            "What suits you",
            f"{len(recommendations)} of {retrieve['considered']} considered &middot; "
            f"{', '.join(retrieve.get('channels', []))}",
        ),
        unsafe_allow_html=True,
    )

    if retrieve.get("relaxed"):
        st.markdown(
            theme.notice(
                "<strong>No outfits in the corpus are worn by anyone with this body "
                "shape</strong>, so these span other shapes. The adjustment notes "
                "matter more than usual here.",
                "amber",
            ),
            unsafe_allow_html=True,
        )

    refine = stage_result(job, StageName.REFINE)
    unmet = refine.get("unmet") or []
    filled = [g for g in refine.get("gaps", []) if g.get("filled")]
    if filled:
        covered = ", ".join(str(g["concept"]) for g in filled)
        st.markdown(
            theme.notice(
                f"<strong>Searched again for {covered}.</strong> Your request mentioned "
                "these and the first pass missed them, so extra outfits were retrieved "
                "specifically to cover them &mdash; marked below.",
                "teal",
            ),
            unsafe_allow_html=True,
        )
    if unmet:
        st.markdown(
            theme.notice(
                "<strong>Not found in this wardrobe: "
                + ", ".join(str(u) for u in unmet)
                + ".</strong> Nothing in the indexed corpus matches those, so the "
                "results below do not include them.",
                "amber",
            ),
            unsafe_allow_html=True,
        )

    for start in range(0, len(recommendations), 3):
        row = recommendations[start : start + 3]
        for offset, (column, rec) in enumerate(zip(st.columns(len(row)), row, strict=True)):
            with column:
                path = Path(str(rec.get("image_path", "")))
                if path.exists():
                    st.image(str(path), use_container_width=True)
                render_card(rec, start + offset)
                render_feedback(rec, analyze, query_text)

    tryon_stage = job.stage(StageName.TRYON)
    if tryon_stage.detail:
        st.markdown(theme.sect("Try-on"), unsafe_allow_html=True)
        st.markdown(theme.notice(tryon_stage.detail), unsafe_allow_html=True)
        first = Path(str(recommendations[0].get("image_path", "")))
        if first.exists():
            a, b = st.columns(2)
            a.image(photo, caption="You", use_container_width=True)
            b.image(str(first), caption="Recommended", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Atelier", page_icon="◆", layout="wide")
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.markdown(theme.NAV, unsafe_allow_html=True)

    vision, embedder, store, generator, tryon, corpus_size = load_engine()
    settings = load_settings()
    render_sidebar(settings, corpus_size, generator, tryon)

    st.markdown(theme.HERO, unsafe_allow_html=True)

    if corpus_size == 0:
        st.markdown(
            theme.notice(
                "<strong>The index is empty.</strong> Run "
                "<code>scripts/fetch_roster.py</code>, then "
                "<code>scripts/ingest.py</code>.",
                "coral",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(theme.sect("Start here"), unsafe_allow_html=True)
    photo_col, form_col = st.columns([1, 2], gap="large")

    with photo_col:
        upload = st.file_uploader(
            "Your photo — full body works best", type=["jpg", "jpeg", "png", "webp"]
        )
        if upload:
            st.image(upload, use_container_width=True)
        reference_upload = st.file_uploader(
            "Or an outfit you like — finds similar ones",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if reference_upload:
            st.image(reference_upload, use_container_width=True)

    with form_col:
        text = st.text_input("What you are looking for", value="festive ethnic wear")
        a, b = st.columns(2)
        culture = a.selectbox("Style", ["any", *[c.value for c in Culture]])
        occasion = b.selectbox("Occasion", ["any", *[o.value for o in Occasion]])
        c, d = st.columns(2)
        # These two were one confusingly-labelled field. They are opposite ends of the
        # cross-cultural match: whose body you share, and whose wardrobe you want.
        body_reference = c.text_input("Build like… (optional)", placeholder="e.g. Zendaya")
        celebrity = d.text_input(
            "Only this celebrity's wardrobe (optional)", placeholder="e.g. Deepika Padukone"
        )
        e, f = st.columns(2)
        region = e.selectbox("Wardrobe", ["indian", "american", "british", "any"])
        top_k = f.slider("How many", 3, 12, 6)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        go = st.button("Find my outfits")

    reference_bytes = reference_upload.getvalue() if reference_upload else None

    if not go:
        # Streamlit reruns the whole script on every interaction, including a feedback
        # click. Without re-rendering the stored run, voting on a card would make the
        # entire results section vanish.
        stored = st.session_state.get("last_run")
        if stored:
            render_results(stored["job"], stored["photo"], query_text=stored["query_text"])
        return

    if upload is None:
        st.markdown(
            theme.notice(
                "Upload a photo of yourself, or name someone whose build you share.",
                "amber",
            ),
            unsafe_allow_html=True,
        )
        return

    photo = upload.getvalue()
    jobs = InMemoryJobStore()
    pipeline = RecommendationPipeline(
        vision=vision,
        embedder=embedder,
        store=store,
        generator=generator,
        tryon=tryon,
        jobs=jobs,
        profiles=celebrity_profiles(),
    )
    query = UserQuery(
        text=text,
        culture=Culture(culture) if culture != "any" else None,
        occasion=Occasion(occasion) if occasion != "any" else None,
        celebrity_name=celebrity or None,
        region=None if region == "any" else region,
        body_reference=body_reference or None,
        top_k=top_k,
    )
    # A shape the user corrected survives reruns, so confirming once is enough.
    confirmed = st.session_state.get("confirmed_shape")

    job = Job()
    jobs.create(job)
    with st.spinner("Reading proportions and retrieving..."):
        job = pipeline.run(
            job,
            photo,
            query,
            confirmed_shape=BodyShape(confirmed) if confirmed else None,
            reference_image=reference_bytes,
        )

    # Kept so feedback clicks (and any other rerun) can redraw the same results.
    st.session_state["last_run"] = {"job": job, "photo": photo, "query_text": text}
    render_results(job, photo, query_text=text)


if __name__ == "__main__":
    main()
