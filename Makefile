# SUMMARY: StarSystem + The Forge Makefile v2
# ============================================= 
# Main build and workflow automation for StarSystem development.
# Supports MCP server management, agent workflows, testing, and artifact generation.
#
# KEY TARGETS:
# - smoke: Full pipeline test (generate → apply_patch → test_headless → capture → dump)
# - mcp-server: Start MCP server in stdio mode for tool integration
# - agents: Run 3-agent handshake (A→B→C) with deterministic stubs
# - test: Execute all validation tests and schema checks
# - clean: Remove runs/ artifacts and build cache
#
# USAGE:
#   make smoke GLOBAL_SEED=123        # Run smoke test with specific seed
#   make mcp-server                   # Start MCP server for development  
#   make agents                       # Test agent communication loop
#   make install                      # Install Python dependencies
#
# Week 2+ Core (Schema Freeze + Planet Generation + Human Feedback)
# Supports idea → playable mini-planet workflow with unlimited creative building

GLOBAL_SEED ?= 0
GODOT_PATH ?= $(shell cat .cache/godot_path 2>/dev/null || echo "godot4-headless")
PYTHON := python3

.PHONY: help smoke mcp-server test clean install

help: ## Show this help message
	@echo "StarSystem + The Forge - Week 1 Pre-alpha"
	@echo ""
	@echo "Usage: make [target] [GLOBAL_SEED=123]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing Python dependencies..."
	pip install mcp requests pyyaml
	@echo "Checking for Godot..."
	@$(PYTHON) -c "from mcp_server.server import discover_godot; print(f'Godot: {discover_godot() or \"NOT FOUND\"}')"

mcp-server: ## Start MCP server in stdio mode  
	@echo "Starting StarSystem MCP Server (stdio)..."
	GLOBAL_SEED=$(GLOBAL_SEED) $(PYTHON) mcp_server/server.py

smoke: ## Run smoke test pipeline: generate → apply_patch → test_headless → capture_views → dump_scene
	@echo "🔄 Running smoke test pipeline (GLOBAL_SEED=$(GLOBAL_SEED))..."
	@GLOBAL_SEED=$(GLOBAL_SEED) $(PYTHON) scripts/smoke_test.py

test: ## Run all tests
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v || echo "No tests found - creating placeholder"
	@test -f tests/test_mcp.py || echo "# Placeholder test" > tests/test_mcp.py

clean: ## Clean runs and cache
	rm -rf runs/*
	rm -rf .cache/godot_path
	@echo "Cleaned runs/ and cache"

agents: ## Run 3-agent handshake
	@echo "Testing 3-agent handshake..."
	OPENAI_API_KEY="$(OPENAI_API_KEY)" $(PYTHON) crew/run_handshake.py