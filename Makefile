# ============================================================================
# RAGit — Makefile
# Default UX:
#   - `make` or `make interactive` -> Ollama backend (default model = $(OLLAMA_MODEL))
#   - `make phogpt`  -> PhoGPT **local** (models/PhoGPT-4B-Chat)
#   - `make phogpt-hub` -> PhoGPT from Hugging Face Hub (repo id)
# ============================================================================

# Make 'interactive' the default goal so bare `make` runs Ollama
.DEFAULT_GOAL := interactive

# ---- UTF-8 environment (avoid garbled accents in terminal) -----------------
export PYTHONUTF8=1
export PYTHONIOENCODING?=utf-8
export LANG?=C.UTF-8
export LC_ALL?=C.UTF-8

# ---- Tooling ---------------------------------------------------------------
PY              ?= poetry run
ENTRY           := rag.main_pipeline
BASE            := $(PY) python -m $(ENTRY)

# ---- Paths & datasets ------------------------------------------------------
TESTS_DIR       ?= src/rag/tests
CORPUS          ?= all     # all | nam | tan
ANSWER_LANG     ?= fr      # vi | fr | en

# ---- Backends --------------------------------------------------------------
# Ollama (default backend for interactive targets)
OLLAMA_MODEL        ?= mistral

# PhoGPT (instruction-tuned; repo id or local dir)
PHOGPT_REPO         ?= vinai/PhoGPT-4B-Chat
PHOGPT_LOCAL_DIR    ?= models/PhoGPT-4B-Chat
# Local by default (user request)
PHOGPT_MODEL        ?= $(PHOGPT_LOCAL_DIR)
USE_4BIT            ?= 0

# ---- Phony -----------------------------------------------------------------
.PHONY: help \
        interactive interactive-all interactive-nam interactive-tan \
        nam tan \
        phogpt phogpt-nam phogpt-tan phogpt-hub phogpt-hub-nam phogpt-hub-tan \
        test-all test-all-ollama test-all-phogpt \
        test-nam-ollama test-nam-phogpt \
        test-tan-ollama test-tan-phogpt \
        test-full full \
        download-phogpt-https download-phogpt-ssh download-phogpt-hf clean-phogpt

# ---- Helper macros ---------------------------------------------------------
# Interactive run (no test suite)
define RUN_INTERACTIVE
	@echo "=== INTERACTIVE: CORPUS=$(1) BACKEND=$(2) MODEL=$(3) LANG=$(4) ==="
	$(BASE) --corpus $(1) -b $(2) -m $(3) -L $(4)
endef

# Batch tests (reads $(TESTS_DIR)/*.yaml)
define RUN_TESTS
	@echo "=== TESTS: CORPUS=$(1) BACKEND=$(2) MODEL=$(3) LANG=$(4) ==="
	$(BASE) --corpus $(1) -b $(2) -m $(3) -L $(4) --tests-dir $(TESTS_DIR) -t --test
endef

# ---- Help ------------------------------------------------------------------
help: ## Show this help with available targets
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-24s %s", $$1, $$2}'

# ---- Interactive (Ollama = default) ---------------------------------------
interactive: interactive-$(CORPUS) ## Interactive shell (Ollama) for selected CORPUS (default: all)
interactive-all: ## Interactive shell (Ollama) on full corpus
	$(call RUN_INTERACTIVE,all,ollama,$(OLLAMA_MODEL),$(ANSWER_LANG))
interactive-nam: ## Interactive shell (Ollama) on Nam Phong
	$(call RUN_INTERACTIVE,nam,ollama,$(OLLAMA_MODEL),$(ANSWER_LANG))
interactive-tan: ## Interactive shell (Ollama) on Tân-Việt
	$(call RUN_INTERACTIVE,tan,ollama,$(OLLAMA_MODEL),$(ANSWER_LANG))

# Short aliases
nam: interactive-nam ## Alias: Ollama on Nam Phong
tan: interactive-tan ## Alias: Ollama on Tân-Việt

# ---- Interactive (PhoGPT local = default) ---------------------------------
phogpt: ## Interactive shell with PhoGPT-4B-Chat (LOCAL folder)
	$(call RUN_INTERACTIVE,$(CORPUS),phogpt,$(PHOGPT_LOCAL_DIR),$(ANSWER_LANG))
phogpt-nam: ## PhoGPT local on Nam Phong
	$(call RUN_INTERACTIVE,nam,phogpt,$(PHOGPT_LOCAL_DIR),$(ANSWER_LANG))
phogpt-tan: ## PhoGPT local on Tân-Việt
	$(call RUN_INTERACTIVE,tan,phogpt,$(PHOGPT_LOCAL_DIR),$(ANSWER_LANG))

# PhoGPT from Hub (repo id)
phogpt-hub: ## Interactive shell with PhoGPT-4B-Chat (HUB repo id)
	$(call RUN_INTERACTIVE,$(CORPUS),phogpt,$(PHOGPT_REPO),$(ANSWER_LANG))
phogpt-hub-nam: ## PhoGPT Hub on Nam Phong
	$(call RUN_INTERACTIVE,nam,phogpt,$(PHOGPT_REPO),$(ANSWER_LANG))
phogpt-hub-tan: ## PhoGPT Hub on Tân-Việt
	$(call RUN_INTERACTIVE,tan,phogpt,$(PHOGPT_REPO),$(ANSWER_LANG))

# ---- Test matrix -----------------------------------------------------------
test-all-ollama: ## Run tests on full corpus with Ollama
	$(call RUN_TESTS,all,ollama,$(OLLAMA_MODEL),$(ANSWER_LANG))

test-all-phogpt: ## Run tests on full corpus with PhoGPT (LOCAL by default)
	$(call RUN_TESTS,all,phogpt,$(PHOGPT_MODEL),$(ANSWER_LANG))

test-nam-ollama: ## Run tests on Nam Phong with Ollama
	$(call RUN_TESTS,nam,ollama,$(OLLAMA_MODEL),$(ANSWER_LANG))

test-nam-phogpt: ## Run tests on Nam Phong with PhoGPT (LOCAL by default)
	$(call RUN_TESTS,nam,phogpt,$(PHOGPT_MODEL),$(ANSWER_LANG))

test-tan-ollama: ## Run tests on Tân-Việt with Ollama
	$(call RUN_TESTS,tan,ollama,$(OLLAMA_MODEL),$(ANSWER_LANG))

test-tan-phogpt: ## Run tests on Tân-Việt with PhoGPT (LOCAL by default)
	$(call RUN_TESTS,tan,phogpt,$(PHOGPT_MODEL),$(ANSWER_LANG))

# Aliases
full: test-full ## Alias for test-full (matrix)
test-all: test-all-ollama ## Alias for Ollama tests on full corpus

test-full: ## Run all tests (Ollama + PhoGPT across all corpora)
	$(MAKE) test-all-ollama
	$(MAKE) test-all-phogpt
	$(MAKE) test-nam-ollama
	$(MAKE) test-nam-phogpt
	$(MAKE) test-tan-ollama
	$(MAKE) test-tan-phogpt

# ---- Model download: Git LFS (recommended) --------------------------------
download-phogpt-https: ## Download PhoGPT via Git LFS over HTTPS into $(PHOGPT_LOCAL_DIR)
	@echo "=== DOWNLOAD PHOGPT VIA GIT LFS (HTTPS) ==="
	@echo "Ensure: 1) accepted model terms for $(PHOGPT_REPO), 2) git-lfs is installed."
	rm -rf "$(PHOGPT_LOCAL_DIR)"
	git lfs clone https://huggingface.co/$(PHOGPT_REPO) "$(PHOGPT_LOCAL_DIR)"

download-phogpt-ssh: ## Download PhoGPT via Git LFS over SSH into $(PHOGPT_LOCAL_DIR)
	@echo "=== DOWNLOAD PHOGPT VIA GIT LFS (SSH) ==="
	@echo "Ensure: 1) accepted model terms for $(PHOGPT_REPO), 2) HF SSH is configured."
	rm -rf "$(PHOGPT_LOCAL_DIR)"
	git lfs clone git@hf.co:$(PHOGPT_REPO) "$(PHOGPT_LOCAL_DIR)"

clean-phogpt: ## Remove the local PhoGPT folder
	rm -rf "$(PHOGPT_LOCAL_DIR)"

# ---- Model download: HF CLI snapshot --------------------------------------
download-phogpt-hf: ## Snapshot the model with huggingface_hub into $(PHOGPT_LOCAL_DIR)
	@echo "=== DOWNLOAD PHOGPT VIA HF CLI ==="
	@echo "Ensure: 1) accepted model terms for $(PHOGPT_REPO), 2) 'huggingface-cli login' succeeded."
	mkdir -p "$(PHOGPT_LOCAL_DIR)"
	HF_HUB_ENABLE_XET=0 HF_HUB_ENABLE_HF_TRANSFER=0 HF_ENDPOINT=https://huggingface.co \
		$(PY) hf download "$(PHOGPT_REPO)" --repo-type model --local-dir "$(PHOGPT_LOCAL_DIR)" --include '*' --exclude '.gitattributes'
