# ============================================================================
# RAGit — Makefile
# Default UX:
#   - `make` -> show available targets
# ============================================================================

.DEFAULT_GOAL := help

# ---- UTF-8 environment (avoid garbled accents in terminal) -----------------
export PYTHONUTF8=1
export PYTHONIOENCODING?=utf-8
export LANG?=C.UTF-8
export LC_ALL?=C.UTF-8

# ---- Tooling ---------------------------------------------------------------
PY              ?= poetry run
ENTRY           := rag.main_pipeline
BASE            := $(PY) python -m $(ENTRY)

# ---- Inputs ----------------------------------------------------------------
TEXT_FILE       ?=
LEXICON_DIR     ?= ../lexicon

# ---- Phony -----------------------------------------------------------------
.PHONY: help \
        exact \
        definition \
        test

# ---- Helper macros ---------------------------------------------------------
define REQUIRE_TEXT_FILE
	@test -n "$(TEXT_FILE)" || (echo "Set TEXT_FILE=/path/to/text.txt" && exit 1)
endef

# ---- Help ------------------------------------------------------------------
help: ## Show this help with available targets
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-24s %s", $$1, $$2}'

exact: ## Run exact lexicon lookup; requires TEXT_FILE and TERM, optional TERM_COLUMN=vi
	$(call REQUIRE_TEXT_FILE)
	@test -n "$(TERM)" || (echo "Set TERM='...'" && exit 1)
	$(BASE) --lexicon-exact --term "$(TERM)" --term-column "$(or $(TERM_COLUMN),vi)" --text-file "$(TEXT_FILE)" --lexicon-dir "$(LEXICON_DIR)"

definition: ## Run similar-definition lookup; requires TEXT_FILE and either DEFINITION_TERM or DEFINITION_TEXT
	$(call REQUIRE_TEXT_FILE)
	@test -n "$(DEFINITION_TERM)$(DEFINITION_TEXT)" || (echo "Set DEFINITION_TERM='...' or DEFINITION_TEXT='...'" && exit 1)
	$(BASE) --lexicon-definition-match --definition-term "$(DEFINITION_TERM)" --definition-text "$(DEFINITION_TEXT)" --definition-column "$(or $(DEFINITION_COLUMN),vi)" --candidate-column "$(or $(CANDIDATE_COLUMN),vi)" --text-file "$(TEXT_FILE)" --lexicon-dir "$(LEXICON_DIR)"

test: ## Run the automated test suite
	$(PY) python -m unittest discover -s tests -p 'test_*.py'
