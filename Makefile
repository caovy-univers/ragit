# ------------------------------------------------------------------------------
# Makefile for RAGit
# - Default: interactive (Ollama, full corpus)
# - Extras: PhoGPT interactive, subcorpus shortcuts
# - Tests: 1.a, 1.b, 2.a, 2.b, 3.a, 3.b + test-full
# ------------------------------------------------------------------------------

# ---- Tooling ---------------------------------------------------------------
PY            ?= poetry run
ENTRY         := rag.main_pipeline
BASE          := $(PY) python -m $(ENTRY)

TESTS_DIR     ?= src/rag/tests

# ---- Corpora ---------------------------------------------------------------
CORPUS ?= all
CORPUS_ALL    := all
CORPUS_NAM    := nam
CORPUS_TAN    := tan

# ---- Models & languages ----------------------------------------------------
# Ollama (default backend)
OLLAMA_MODEL  ?= mistral
LANG_OLLAMA   ?= fr

# PhoGPT (optional backend)
# Can be either a repo id "vinai/PhoGPT-7B5" or a local path "models/PhoGPT-4B"
PHOGPT_MODEL  ?= models/PhoGPT-4B
LANG_PHOGPT   ?= vi
PHOGPT_LOCAL_DIR ?= models/PhoGPT-4B

# ---- Defaults --------------------------------------------------------------
.DEFAULT_GOAL := interactive

# ---- Phony targets ---------------------------------------------------------
.PHONY: interactive phogpt nam tan phogpt-nam phogpt-tan \
        test-all test-all-phogpt \
        test-nam-ollama test-nam-phogpt \
        test-tan-ollama test-tan-phogpt \
        test-full help \
        download-phogpt-hf download-phogpt-ssh clean-phogpt

# ---- Helper macros ---------------------------------------------------------
# Interactive run (no test suite)
define RUN_INTERACTIVE
	@echo "=== INTERACTIVE: CORPUS=$(1) BACKEND=$(2) MODEL=$(3) LANG=$(4) ==="
	$(BASE) --corpus $(1) -b $(2) -m $(3) -L $(4)
endef

# Batch tests (reads $(TESTS_DIR)/*.yaml)
define RUN_TESTS
	@echo "=== TESTS: CORPUS=$(1) BACKEND=$(2) MODEL=$(3) LANG=$(4) SCOPE=$(5) ==="
	$(BASE) --corpus $(1) -b $(2) -m $(3) -L $(4) --tests-dir $(TESTS_DIR) -t --test-scope $(5)
endef

# ---- Interactive shortcuts -------------------------------------------------
# Default: Ollama, full corpus
interactive: ## Interactive pipeline (Ollama, full corpus)
	$(call RUN_INTERACTIVE,$(CORPUS_ALL),ollama,$(OLLAMA_MODEL),$(LANG_OLLAMA))

# PhoGPT, full corpus
phogpt: ## Interactive pipeline (PhoGPT, full corpus)
	@echo "=== INTERACTIVE: CORPUS=$(CORPUS) BACKEND=phogpt MODEL=$(PHOGPT_MODEL) LANG=$(LANG_PHOGPT) ==="
	$(BASE) --corpus $(CORPUS) -b phogpt -m $(PHOGPT_MODEL) -L $(LANG_PHOGPT) --use-4bit

# Ollama on a specific subcorpus
nam: ## Interactive (Ollama) on Nam-Phong
	$(call RUN_INTERACTIVE,$(CORPUS_NAM),ollama,$(OLLAMA_MODEL),$(LANG_OLLAMA))

tan: ## Interactive (Ollama) on Tân-Việt
	$(call RUN_INTERACTIVE,$(CORPUS_TAN),ollama,$(OLLAMA_MODEL),$(LANG_OLLAMA))

# PhoGPT on a specific subcorpus
phogpt-nam: ## Interactive (PhoGPT) on Nam-Phong
	$(call RUN_INTERACTIVE,$(CORPUS_NAM),phogpt,$(PHOGPT_MODEL),$(LANG_PHOGPT))

phogpt-tan: ## Interactive (PhoGPT) on Tân-Việt
	$(call RUN_INTERACTIVE,$(CORPUS_TAN),phogpt,$(PHOGPT_MODEL),$(LANG_PHOGPT))

# ---- Batch tests: 1.a / 1.b -----------------------------------------------
test-all: ## 1.a – Run tests on full corpus with Ollama
	$(call RUN_TESTS,$(CORPUS_ALL),ollama,$(OLLAMA_MODEL),$(LANG_OLLAMA),all)

test-all-phogpt: ## 1.b – Run tests on full corpus with PhoGPT
	$(call RUN_TESTS,$(CORPUS_ALL),phogpt,$(PHOGPT_MODEL),$(LANG_PHOGPT),all)

# ---- Batch tests: 2.a / 2.b / 3.a / 3.b -----------------------------------
test-nam-ollama: ## 2.a – Ollama on Nam-Phong
	$(call RUN_TESTS,$(CORPUS_NAM),ollama,$(OLLAMA_MODEL),$(LANG_OLLAMA),nam)

test-nam-phogpt: ## 2.b – PhoGPT on Nam-Phong
	$(call RUN_TESTS,$(CORPUS_NAM),phogpt,$(PHOGPT_MODEL),$(LANG_PHOGPT),nam)

test-tan-ollama: ## 3.a – Ollama on Tân-Việt
	$(call RUN_TESTS,$(CORPUS_TAN),ollama,$(OLLAMA_MODEL),$(LANG_OLLAMA),tan)

test-tan-phogpt: ## 3.b – PhoGPT on Tân-Việt
	$(call RUN_TESTS,$(CORPUS_TAN),phogpt,$(PHOGPT_MODEL),$(LANG_PHOGPT),tan)

# ---- Combined subcorpus matrix --------------------------------------------
test-full: ## Run 2.a, 2.b, 3.a, 3.b in sequence
	@echo "=== TESTS: NAM-PHONG / OLLAMA ==="
	$(call RUN_TESTS,$(CORPUS_NAM),ollama,$(OLLAMA_MODEL),$(LANG_OLLAMA),nam)
	@echo "=== TESTS: NAM-PHONG / PHOGPT ==="
	$(call RUN_TESTS,$(CORPUS_NAM),phogpt,$(PHOGPT_MODEL),$(LANG_PHOGPT),nam)
	@echo "=== TESTS: TAN-VIET / OLLAMA ==="
	$(call RUN_TESTS,$(CORPUS_TAN),ollama,$(OLLAMA_MODEL),$(LANG_OLLAMA),tan)
	@echo "=== TESTS: TAN-VIET / PHOGPT ==="
	$(call RUN_TESTS,$(CORPUS_TAN),phogpt,$(PHOGPT_MODEL),$(LANG_PHOGPT),tan)

# ---- PhoGPT model download helpers ----------------------------------------
# Requires:
#   1) Accept terms on https://huggingface.co/vinai/PhoGPT-7B5
#   2) `huggingface-cli login` (token) or SSH auth set up with HF
download-phogpt-hf: ## Download PhoGPT via HF CLI (non-deprecated) to $(PHOGPT_LOCAL_DIR)
	@echo "=== DOWNLOAD PHOGPT VIA HF CLI ==="
	@echo "Ensure: 1) you accepted model terms, 2) 'huggingface-cli login' succeeded."
	mkdir -p $(PHOGPT_LOCAL_DIR)
	# Force plain HTTP client (avoid xet/transfer issues on some networks)
	HF_HUB_ENABLE_XET=0 HF_HUB_ENABLE_HF_TRANSFER=0 HF_ENDPOINT=https://huggingface.co \
		$(PY) hf download vinai/PhoGPT-7B5 --repo-type model --local-dir $(PHOGPT_LOCAL_DIR) --include '*' --exclude '.gitattributes'

download-phogpt-ssh: ## Download PhoGPT via Git+SSH (requires HF SSH auth + Git LFS)
	@echo "=== DOWNLOAD PHOGPT VIA GIT+SSH ==="
	@echo "Ensure: you accepted model terms and configured SSH with Hugging Face."
	rm -rf $(PHOGPT_LOCAL_DIR)
	git lfs clone git@hf.co:vinai/PhoGPT-7B5 $(PHOGPT_LOCAL_DIR)

clean-phogpt: ## Remove local PhoGPT folder
	rm -rf $(PHOGPT_LOCAL_DIR)

# ---- Help ------------------------------------------------------------------
help: ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'
