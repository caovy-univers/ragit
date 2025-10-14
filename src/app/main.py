"""Streamlit front-end for the Rag'it RAG pipeline."""

from __future__ import annotations

import streamlit as st
from rag.main_pipeline import make_rag, resolve_corpus_dirs
from rag.wrapper import SimpleRAG
from pathlib import Path
from typing import Iterable, Tuple

st.set_page_config(page_title="Rag'it Explorer", layout="wide")
st.title("Rag'it Explorer 🧭")
st.write(
    "Query the Nam Phong / Tân Việt corpora with a Retrieval-Augmented Generation pipeline."
    " Ensure dependencies are installed locally with `poetry install`."
)


def _parse_dirs(raw: str) -> Tuple[Path, ...]:
    """Convert a comma-separated list into concrete ``Path`` objects."""
    items = [item.strip() for item in (raw or "").split(",") if item.strip()]
    return tuple(Path(item) for item in items)


@st.cache_resource(show_spinner=False)
def load_pipeline(
    corpus: str,
    backend: str,
    model_name: str,
    target_lang: str,
    use_4bit: bool,
    custom_dirs: Tuple[Path, ...],
) -> SimpleRAG:
    """Instantiate the RAG pipeline once per configuration."""
    if corpus == "custom":
        if not custom_dirs:
            raise ValueError("Provide at least one data directory when using the 'custom' corpus option.")
        data_dirs: Iterable[Path] = custom_dirs
    else:
        data_dirs = resolve_corpus_dirs(corpus)

    return make_rag(
        data_dirs=list(data_dirs),
        backend=backend,
        model_name=model_name,
        target_lang=target_lang,
        use_4bit=use_4bit,
    )


with st.sidebar:
    st.header("Configuration")
    corpus = st.selectbox(
        "Corpus",
        options=[("all", "Nam Phong + Tân Việt"), ("nam", "Nam Phong"), ("tan", "Tân Việt"), ("custom", "Custom directories")],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    backend = st.selectbox("LLM backend", options=["ollama", "phogpt"], index=0)

    local_phogpt = Path("models/PhoGPT-4B-Chat")
    default_source = "Hub" if backend == "ollama" else ("Local" if local_phogpt.exists() else "Hub")

    if "_backend" not in st.session_state:
        st.session_state._backend = backend
    if "_source" not in st.session_state:
        st.session_state._source = default_source
    if "_target_lang" not in st.session_state:
        st.session_state._target_lang = "fr"

    if st.session_state._backend != backend:
        st.session_state._backend = backend
        st.session_state._source = "Hub" if backend == "ollama" else ("Local" if local_phogpt.exists() else "Hub")
        st.session_state._target_lang = "vi" if backend == "phogpt" else "fr"

    source_options = ["Hub"] if backend == "ollama" else ["Local", "Hub"]
    if st.session_state._source not in source_options:
        st.session_state._source = source_options[0]

    source = st.selectbox(
        "Model source",
        options=source_options,
        index=source_options.index(st.session_state._source),
        key="_source",
    )

    target_options = ["vi"] if backend == "phogpt" else ["fr", "vi", "en"]
    if st.session_state._target_lang not in target_options:
        st.session_state._target_lang = target_options[0]

    target_lang = st.selectbox(
        "Target language",
        options=target_options,
        index=target_options.index(st.session_state._target_lang),
        key="_target_lang",
    )

    use_4bit = False
    if backend == "phogpt":
        use_4bit = st.checkbox("Load PhoGPT in 4-bit (requires bitsandbytes)", value=True)

    custom_raw = st.text_area(
        "Custom directories (comma separated)",
        placeholder="/path/to/folder1, /path/to/folder2",
        height=80,
    )

    config_summary = st.container()

# Determine model path/name based on backend and source selection
if backend == "ollama":
    model_name = "mistral"
else:
    if source == "Local":
        if local_phogpt.exists():
            model_name = str(local_phogpt)
        else:
            st.warning(
                "Local PhoGPT weights not found under `models/PhoGPT-4B-Chat`. Falling back to the Hugging Face Hub."
            )
            st.session_state._source = "Hub"
            source = "Hub"
            model_name = "vinai/PhoGPT-4B-Chat"
    else:
        model_name = "vinai/PhoGPT-4B-Chat"

st.subheader("Pipeline")

pipeline_error = None
rag_instance: SimpleRAG | None = None

try:
    with st.spinner("Loading RAG pipeline…"):
        rag_instance = load_pipeline(
            corpus=corpus,
            backend=backend,
            model_name=model_name,
            target_lang=target_lang,
            use_4bit=use_4bit,
            custom_dirs=_parse_dirs(custom_raw),
        )
except ModuleNotFoundError as exc:
    missing = ", ".join(map(str, exc.args)) or "module"
    pipeline_error = f"Missing modules: `{missing}`. Run `poetry install` and verify optional deps are present."
except ValueError as exc:
    pipeline_error = str(exc)
except FileNotFoundError as exc:
    pipeline_error = f"Cannot access data files: {exc}"
except Exception as exc:  # pragma: no cover - Streamlit runtime safeguard
    pipeline_error = f"Failed to initialize the pipeline: {exc}"

if pipeline_error:
    st.error(pipeline_error)
    st.stop()

config_summary.success(
    f"Corpus: `{corpus}` · Backend: `{backend}` · Source: `{source}` · Model: `{model_name}` · Target language: `{target_lang}`"
)

st.success("Pipeline ready. Ask a question below.")

question = st.text_area(
    "Question",
    placeholder="e.g. What recurring themes about education appear across the corpus?",
    height=120,
)
submit = st.button("Run query", type="primary")

if submit:
    if not question.strip():
        st.warning("Please enter a question before running the query.")
    else:
        with st.spinner("Generating answer…"):
            try:
                output = rag_instance.ask(question.strip())
            except Exception as exc:  # pragma: no cover - handles runtime issues
                st.error(f"Generation failed: {exc}")
            else:
                st.markdown("### Answer")
                st.write(output.get("result", "(no answer)"))

                docs = output.get("source_documents") or []
                if docs:
                    st.markdown("### Sources")
                    for idx, doc in enumerate(docs, start=1):
                        meta = doc.metadata or {}
                        title = meta.get("title_main") or "Unknown title"
                        authors = ", ".join(meta.get("authors") or []) or "Unknown authors"
                        date = meta.get("publication_date") or "Unknown date"
                        st.write(f"{idx}. **{title}** — {authors} ({date})")
                        with st.expander("Excerpt"):
                            st.write(doc.page_content)
else:
    st.info("Configure the pipeline on the left, then submit a question here.")
