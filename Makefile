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

# Phase 1: Baseline Model Workflow
split-dataset: ## Split dataset into train/val/test (make split-dataset DATASET=datasets/credit-cards-coco)
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET must be set"; \
		echo "Example: make split-dataset DATASET=datasets/credit-cards-coco"; \
		exit 1; \
	fi
	python3 src/split_dataset.py --dataset $(DATASET) --seed 42

prepare-progressive-tests: ## Generate progressive occlusion test sets (make prepare-progressive-tests TEST_DATASET=datasets/credit-cards-coco_split/test)
	@if [ -z "$(TEST_DATASET)" ]; then \
		echo "Error: TEST_DATASET must be set"; \
		echo "Example: make prepare-progressive-tests TEST_DATASET=datasets/credit-cards-coco_split/test"; \
		exit 1; \
	fi
	python3 src/prepare_progressive_tests.py --test-dataset $(TEST_DATASET) --type $(or $(OCCLUSION_TYPE),crop) --seed 42

visualize-crops: ## Visualize cropped regions (make visualize-crops ORIGINAL=datasets/credit-cards-coco OCCLUDED=datasets/credit-cards-coco_split/test_occlusion_25)
	@if [ -z "$(ORIGINAL)" ] || [ -z "$(OCCLUDED)" ]; then \
		echo "Error: ORIGINAL and OCCLUDED must be set"; \
		echo "Example: make visualize-crops ORIGINAL=datasets/credit-cards-coco OCCLUDED=datasets/credit-cards-coco_split/test_occlusion_25"; \
		exit 1; \
	fi
	python3 src/visualize_crops.py --original $(ORIGINAL) --occluded $(OCCLUDED) --samples $(or $(SAMPLES),6)

visualize-progressive: ## Visualize progressive occlusion (make visualize-progressive ORIGINAL=datasets/credit-cards-coco_split/test OCC25=... OCC50=... OCC75=...)
	@if [ -z "$(ORIGINAL)" ] || [ -z "$(OCC25)" ] || [ -z "$(OCC50)" ] || [ -z "$(OCC75)" ]; then \
		echo "Error: ORIGINAL, OCC25, OCC50, OCC75 must be set"; \
		echo "Example: make visualize-progressive ORIGINAL=datasets/credit-cards-coco_split/test OCC25=datasets/credit-cards-coco_split/test_occlusion_25 OCC50=datasets/credit-cards-coco_split/test_occlusion_50 OCC75=datasets/credit-cards-coco_split/test_occlusion_75"; \
		exit 1; \
	fi
	python3 src/visualize_progressive.py --original $(ORIGINAL) --occlusion-25 $(OCC25) --occlusion-50 $(OCC50) --occlusion-75 $(OCC75) $(if $(IMAGE),--image $(IMAGE))

train-model: ## Train YOLOv8 model (make train-model DATASET=datasets/credit-cards-coco_split/train MODEL_SIZE=n EPOCHS=100)
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET must be set"; \
		echo "Example: make train-model DATASET=datasets/credit-cards-coco_split/train MODEL_SIZE=n EPOCHS=100"; \
		exit 1; \
	fi
	python3 src/train.py --dataset $(DATASET) --model-size $(or $(MODEL_SIZE),n) --epochs $(or $(EPOCHS),100) --batch $(or $(BATCH),16)

evaluate-progressive: ## Evaluate model on progressive occlusion (make evaluate-progressive MODEL=models/credit_card_n/weights/best.pt TEST_SETS=datasets/credit-cards-coco_split)
	@if [ -z "$(MODEL)" ] || [ -z "$(TEST_SETS)" ]; then \
		echo "Error: MODEL and TEST_SETS must be set"; \
		echo "Example: make evaluate-progressive MODEL=models/credit_card_n/weights/best.pt TEST_SETS=datasets/credit-cards-coco_split"; \
		exit 1; \
	fi
	python3 src/evaluate_progressive.py --model $(MODEL) --test-sets $(TEST_SETS)

phase1-all: split-dataset prepare-progressive-tests ## Run Phase 1 workflow: split dataset and prepare progressive tests
	@echo "Phase 1 preparation complete!"
	@echo "Next steps:"
	@echo "  1. Train model: make train-model DATASET=datasets/credit-cards-coco_split/train"
	@echo "  2. Evaluate: make evaluate-progressive MODEL=models/.../best.pt TEST_SETS=datasets/credit-cards-coco_split"

# Unified Training and Inference (New)
train-unified: ## Train model using unified config (make train-unified DATASET=datasets/my_dataset MODEL_NAME=yolov8n EPOCHS=100)
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET must be set"; \
		echo "Example: make train-unified DATASET=datasets/my_dataset MODEL_NAME=yolov8n EPOCHS=100"; \
		exit 1; \
	fi
	python3 src/train_unified.py \
		--dataset-path $(DATASET) \
		--framework $(or $(FRAMEWORK),ultralytics) \
		--model-name $(or $(MODEL_NAME),yolov8n) \
		--epochs $(or $(EPOCHS),100) \
		--batch-size $(or $(BATCH_SIZE),16) \
		--img-size $(or $(IMG_SIZE),640) \
		--output-dir $(or $(OUTPUT_DIR),models) \
		--device $(or $(DEVICE),cuda) \
		$(if $(CONFIG),--config $(CONFIG))

inference-video: ## Run video inference (make inference-video MODEL=models/.../best.pt VIDEO=input.mp4 OUTPUT=output.mp4)
	@if [ -z "$(MODEL)" ] || [ -z "$(VIDEO)" ]; then \
		echo "Error: MODEL and VIDEO must be set"; \
		echo "Example: make inference-video MODEL=models/model_n/weights/best.pt VIDEO=input.mp4 OUTPUT=output.mp4"; \
		exit 1; \
	fi
	python3 src/inference.py \
		--model $(MODEL) \
		--video $(VIDEO) \
		--output $(or $(OUTPUT),outputs/$(shell basename $(VIDEO) .mp4)_detected.mp4) \
		--conf-threshold $(or $(CONF_THRESHOLD),0.25) \
		--device $(or $(DEVICE),cuda) \
		$(if $(CONFIG),--config $(CONFIG)) \
		$(if $(SHOW),--show)

inference-image: ## Run image inference (make inference-image MODEL=models/.../best.pt IMAGE=input.jpg OUTPUT=output.jpg)
	@if [ -z "$(MODEL)" ] || [ -z "$(IMAGE)" ]; then \
		echo "Error: MODEL and IMAGE must be set"; \
		echo "Example: make inference-image MODEL=models/model_n/weights/best.pt IMAGE=input.jpg OUTPUT=output.jpg"; \
		exit 1; \
	fi
	python3 src/inference.py \
		--model $(MODEL) \
		--image $(IMAGE) \
		--output $(or $(OUTPUT),outputs/$(shell basename $(IMAGE))) \
		--conf-threshold $(or $(CONF_THRESHOLD),0.25) \
		--device $(or $(DEVICE),cuda) \
		$(if $(CONFIG),--config $(CONFIG))

