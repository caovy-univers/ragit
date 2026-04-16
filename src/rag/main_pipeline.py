import argparse
import sys
from pathlib import Path

from rag.lexicon_definition_match import run_definition_match
from rag.lexicon_exact_match import load_text_record_from_file, run_exact_lexicon_match

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def print_exact_match_report(payload: dict) -> None:
    text = payload["text"]
    print(f"Text: {text['title_main']}")
    print(f"Source file: {text['source_path']}")
    print(f"Lexicon query: column='{payload['term_column']}', term='{payload['term']}'")

    for idx, result in enumerate(payload["results"], start=1):
        entry = result["lexicon_entry"]
        print(f"\nEntry {idx}")
        print(f"- vi: {entry['vi']}")
        print(f"- vi-hani: {entry['vi-hani']}")
        print(f"- fr: {entry['fr']}")
        print(f"- vi-def: {entry['vi-def']}")
        print(f"- source: {entry['source_csv']}:{entry['row_number']}")

        occurrences = result["occurrences"]
        if not occurrences:
            print("- occurrences: none found in selected text")
            continue

        for occ_idx, occurrence in enumerate(occurrences, start=1):
            print(f"\n  Occurrence {occ_idx}")
            print(f"  page: {occurrence['page']}")
            print(f"  line: {occurrence['line_number']}")
            print(f"  matched term: {occurrence['matched_text']}")
            print("  context:")
            for line in occurrence["context"].splitlines():
                print(f"    {line}")


def print_definition_match_report(payload: dict) -> None:
    text = payload["text"]
    print(f"Text: {text['title_main']}")
    print(f"Source file: {text['source_path']}")
    print(
        "Definition query: "
        f"definition_column='{payload['definition_column']}', candidate_column='{payload['candidate_column']}'"
    )

    for idx, result in enumerate(payload["results"], start=1):
        query_entry = result["query_entry"]
        print(f"\nDefinition source {idx}")
        print(f"- vi-def: {query_entry['vi-def']}")
        if query_entry["vi"] or query_entry["vi-hani"] or query_entry["fr"]:
            print(f"- vi: {query_entry['vi']}")
            print(f"- vi-hani: {query_entry['vi-hani']}")
            print(f"- fr: {query_entry['fr']}")
            print(f"- source: {query_entry['source_csv']}:{query_entry['row_number']}")

        matches = result["matches"]
        if not matches:
            print("- matches: none found")
            continue

        for match_idx, match in enumerate(matches, start=1):
            entry = match["lexicon_entry"]
            print(f"\n  Match {match_idx}")
            print(f"  term in text: {match['term_in_text']}")
            print(f"  similarity: {match['similarity']:.4f}")
            print(f"  lexicon vi-def: {entry['vi-def']}")
            print(f"  lexicon terms: vi='{entry['vi']}' | vi-hani='{entry['vi-hani']}' | fr='{entry['fr']}'")

            for occ_idx, occurrence in enumerate(match["occurrences"], start=1):
                print(f"\n    Occurrence {occ_idx}")
                print(f"    page: {occurrence['page']}")
                print(f"    line: {occurrence['line_number']}")
                print(f"    matched term: {occurrence['matched_text']}")
                print("    context:")
                for line in occurrence["context"].splitlines():
                    print(f"      {line}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lexicon matching on a user-supplied text file.")
    parser.add_argument(
        "--lexicon-exact",
        action="store_true",
        help="Run an exact lexicon lookup on column 'vi', 'vi-hani', or 'fr', then search that term in the text file.",
    )
    parser.add_argument(
        "--lexicon-definition-match",
        action="store_true",
        help="Find lexicon terms present in the text file whose 'vi-def' is similar to a given definition.",
    )
    parser.add_argument(
        "--term",
        type=str,
        help="Term to look up in the lexicon for exact mode.",
    )
    parser.add_argument(
        "--term-column",
        choices=["vi", "vi-hani", "fr"],
        help="Lexicon column used for the exact term lookup.",
    )
    parser.add_argument(
        "--definition-text",
        type=str,
        help="Definition text used as the semantic query.",
    )
    parser.add_argument(
        "--definition-term",
        type=str,
        help="Lexicon term whose associated 'vi-def' should be used as the semantic query.",
    )
    parser.add_argument(
        "--definition-column",
        choices=["vi", "vi-hani", "fr"],
        default="vi",
        help="Lexicon column used to resolve --definition-term.",
    )
    parser.add_argument(
        "--candidate-column",
        choices=["vi", "vi-hani", "fr"],
        default="vi",
        help="Lexicon column whose terms are searched inside the text file.",
    )
    parser.add_argument(
        "--lexicon-dir",
        type=str,
        help="Directory containing lexicon CSV files. Defaults to 'lexicon' or '../lexicon' if available.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=1,
        help="Number of surrounding lines to include around each match.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of similar-definition matches to display.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.08,
        help="Minimum similarity score required for similar-definition matches.",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        required=True,
        help="Plain text file supplied by the user and used as the target text.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.lexicon_exact == args.lexicon_definition_match:
        raise ValueError("Choose exactly one mode: --lexicon-exact or --lexicon-definition-match.")

    text_path = Path(args.text_file)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    text_record = load_text_record_from_file(text_path)

    if args.lexicon_exact:
        if not args.term or not args.term_column:
            raise ValueError("Exact lexicon mode requires --term and --term-column.")
        payload = run_exact_lexicon_match(
            term=args.term,
            term_column=args.term_column,
            text_record=text_record,
            lexicon_dir=args.lexicon_dir,
            context_window=max(0, args.context_window),
        )
        print_exact_match_report(payload)
        return 0

    if not args.definition_text and not args.definition_term:
        raise ValueError("Definition match mode requires --definition-text or --definition-term.")
    payload = run_definition_match(
        definition_text=args.definition_text,
        definition_term=args.definition_term,
        text_record=text_record,
        definition_column=args.definition_column,
        candidate_column=args.candidate_column,
        lexicon_dir=args.lexicon_dir,
        context_window=max(0, args.context_window),
        max_results=max(1, args.max_results),
        min_similarity=max(0.0, args.min_similarity),
    )
    print_definition_match_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
