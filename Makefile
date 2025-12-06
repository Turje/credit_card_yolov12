.PHONY: help install setup download-credit-card clean test check-env visualize overlay stats obscure

# Default values
WORKSPACE ?= 
PROJECT ?= 
VERSION ?= 
FORMAT ?= yolov8
DATASET_DIR ?= datasets

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

setup: install ## Setup project (install deps and create directories)
	mkdir -p datasets models outputs
	@echo "Setup complete! Don't forget to set ROBOFLOW_API_KEY environment variable."

download-credit-card: check-env ## Download credit card dataset (example: make download-credit-card WORKSPACE=your-workspace PROJECT=credit-card VERSION=1)
	@if [ -z "$(WORKSPACE)" ] || [ -z "$(PROJECT)" ] || [ -z "$(VERSION)" ]; then \
		echo "Error: WORKSPACE, PROJECT, and VERSION must be set"; \
		echo "Example: make download-credit-card WORKSPACE=my-workspace PROJECT=credit-card VERSION=1"; \
		exit 1; \
	fi
	python src/downloader.py --workspace $(WORKSPACE) --project $(PROJECT) --version $(VERSION) --format $(FORMAT) --location $(DATASET_DIR)

download: check-env ## Download any dataset (make download WORKSPACE=ws PROJECT=proj VERSION=1)
	@if [ -z "$(WORKSPACE)" ] || [ -z "$(PROJECT)" ] || [ -z "$(VERSION)" ]; then \
		echo "Error: WORKSPACE, PROJECT, and VERSION must be set"; \
		exit 1; \
	fi
	python src/downloader.py --workspace $(WORKSPACE) --project $(PROJECT) --version $(VERSION) --format $(FORMAT) --location $(DATASET_DIR)

check-env: ## Check if ROBOFLOW_API_KEY is set
	@if [ -z "$$ROBOFLOW_API_KEY" ]; then \
		echo "Warning: ROBOFLOW_API_KEY environment variable is not set"; \
		echo "Set it with: export ROBOFLOW_API_KEY=your_api_key"; \
		exit 1; \
	fi

clean: ## Clean downloaded datasets and outputs
	rm -rf datasets/* models/* outputs/*
	@echo "Cleaned datasets, models, and outputs directories"

clean-datasets: ## Clean only datasets directory
	rm -rf datasets/*
	@echo "Cleaned datasets directory"

list-datasets: ## List downloaded datasets
	@if [ -d "datasets" ]; then \
		ls -la datasets/; \
	else \
		echo "No datasets directory found"; \
	fi

test: ## Run basic tests
	python -c "from src import config; print('Config loaded successfully')"
	@echo "Basic test passed"

visualize: ## Create overlay images and generate statistics (make visualize DATASET=datasets/credit-cards-coco)
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET must be set"; \
		echo "Example: make visualize DATASET=datasets/credit-cards-coco"; \
		exit 1; \
	fi
	python3 src/visualize.py --dataset $(DATASET) --overlay --stats

overlay: ## Create overlay images only (make overlay DATASET=datasets/credit-cards-coco)
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET must be set"; \
		exit 1; \
	fi
	python3 src/visualize.py --dataset $(DATASET) --overlay

stats: ## Generate statistics only (make stats DATASET=datasets/credit-cards-coco)
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET must be set"; \
		exit 1; \
	fi
	python3 src/visualize.py --dataset $(DATASET) --stats

obscure: ## Create partially obscured dataset (make obscure DATASET=datasets/credit-cards-coco OUTPUT=datasets/credit-cards-obscured TYPE=patch RATIO=0.3)
	@if [ -z "$(DATASET)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Error: DATASET and OUTPUT must be set"; \
		echo "Example: make obscure DATASET=datasets/credit-cards-coco OUTPUT=datasets/credit-cards-obscured TYPE=patch RATIO=0.3"; \
		exit 1; \
	fi
	python3 src/obscure.py --dataset $(DATASET) --output $(OUTPUT) --type $(TYPE) --ratio $(RATIO)

