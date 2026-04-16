"""Streamlit front-end for lexicon matching on user-uploaded texts."""

from __future__ import annotations

import streamlit as st

from rag.lexicon_definition_match import run_definition_match
from rag.lexicon_exact_match import make_text_record_from_string


def _decode_uploaded_file(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode the uploaded file. Please upload a UTF-8 text file.")


def _render_exact_results(payload: dict) -> None:
    for result in payload["results"]:
        entry = result["lexicon_entry"]
        st.markdown("### Lexicon entry")
        st.write(
            {
                "vi": entry["vi"],
                "vi-hani": entry["vi-hani"],
                "fr": entry["fr"],
                "vi-def": entry["vi-def"],
                "source": f"{entry['source_csv']}:{entry['row_number']}",
            }
        )

        occurrences = result["occurrences"]
        if not occurrences:
            st.warning("No occurrence of this term was found in the uploaded text.")
            continue

        st.markdown("### Occurrences")
        for idx, occurrence in enumerate(occurrences, start=1):
            with st.expander(f"Occurrence {idx} · {occurrence['page']} · line {occurrence['line_number']}"):
                st.write(f"Matched term: {occurrence['matched_text']}")
                st.code(occurrence["context"])


def _render_definition_results(payload: dict) -> None:
    for result in payload["results"]:
        entry = result["query_entry"]
        st.markdown("### Definition source")
        st.write(
            {
                "vi-def": entry["vi-def"],
                "vi": entry["vi"],
                "vi-hani": entry["vi-hani"],
                "fr": entry["fr"],
                "source": f"{entry['source_csv']}:{entry['row_number']}",
            }
        )

        matches = result["matches"]
        if not matches:
            st.warning("No similar lexicon term found in the uploaded text.")
            continue

        st.markdown("### Similar matches")
        for idx, match in enumerate(matches, start=1):
            lexicon_entry = match["lexicon_entry"]
            with st.expander(f"Match {idx} · {match['term_in_text']} · score {match['similarity']:.4f}"):
                st.write(
                    {
                        "term_in_text": match["term_in_text"],
                        "similarity": match["similarity"],
                        "vi-def": lexicon_entry["vi-def"],
                        "vi": lexicon_entry["vi"],
                        "vi-hani": lexicon_entry["vi-hani"],
                        "fr": lexicon_entry["fr"],
                    }
                )
                for occ in match["occurrences"]:
                    st.caption(f"{occ['page']} · line {occ['line_number']}")
                    st.code(occ["context"])


st.set_page_config(page_title="Rag'it Lexicon Explorer", layout="wide")
st.title("Rag'it Lexicon Explorer")
st.write("Upload a text file, then query the published lexicon against that text.")

uploaded_file = st.file_uploader("Text to analyze", type=["txt", "md"])

with st.sidebar:
    st.header("Query mode")
    mode = st.radio(
        "Choose a search mode",
        options=("Exact term", "Similar definition"),
        index=0,
    )
    context_window = st.number_input("Context window (lines)", min_value=0, max_value=10, value=1, step=1)

if uploaded_file is None:
    st.info("Upload a `.txt` or `.md` file to begin.")
    st.stop()

try:
    text_content = _decode_uploaded_file(uploaded_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

text_record = make_text_record_from_string(
    text_content,
    title=uploaded_file.name,
    source_path=uploaded_file.name,
)

if mode == "Exact term":
    col1, col2 = st.columns(2)
    with col1:
        term = st.text_input("Lexicon term")
    with col2:
        term_column = st.selectbox("Lexicon column", options=["vi", "vi-hani", "fr"], index=0)

    if st.button("Run exact lookup", type="primary"):
        if not term.strip():
            st.warning("Enter a term from the lexicon.")
        else:
            try:
                from rag.lexicon_exact_match import run_exact_lexicon_match

                payload = run_exact_lexicon_match(
                    term=term.strip(),
                    term_column=term_column,
                    text_record=text_record,
                    context_window=int(context_window),
                )
            except Exception as exc:  # pragma: no cover
                st.error(str(exc))
            else:
                _render_exact_results(payload)
else:
    col1, col2 = st.columns(2)
    with col1:
        definition_term = st.text_input("Lexicon term used as definition source")
        definition_column = st.selectbox("Definition source column", options=["vi", "vi-hani", "fr"], index=0)
    with col2:
        candidate_column = st.selectbox("Search terms in uploaded text by column", options=["vi", "vi-hani", "fr"], index=0)
        max_results = st.number_input("Max results", min_value=1, max_value=20, value=5, step=1)
        min_similarity = st.number_input("Minimum similarity", min_value=0.0, max_value=1.0, value=0.08, step=0.01)

    definition_text = st.text_area(
        "Or provide a custom definition text",
        placeholder="If filled, this definition is used instead of the term above.",
        height=120,
    )

    if st.button("Run definition lookup", type="primary"):
        if not definition_text.strip() and not definition_term.strip():
            st.warning("Enter either a lexicon term or a custom definition text.")
        else:
            try:
                payload = run_definition_match(
                    definition_text=definition_text.strip() or None,
                    definition_term=definition_term.strip() or None,
                    text_record=text_record,
                    definition_column=definition_column,
                    candidate_column=candidate_column,
                    context_window=int(context_window),
                    max_results=int(max_results),
                    min_similarity=float(min_similarity),
                )
            except Exception as exc:  # pragma: no cover
                st.error(str(exc))
            else:
                _render_definition_results(payload)
