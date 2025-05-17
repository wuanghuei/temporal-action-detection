# Temporal Action Detection

## Introduction

This project aims to detect temporal action segments in videos. It utilizes a multi-stream architecture combining information from RGB (using MViT and a 2D CNN) and Pose data (using LSTM). Subsequently, an RNN model (LSTM/GRU) is employed for post-processing the predictions from the main model to refine the final results.

This project is based on the MERL Shopping Dataset.

## Directory Structure

```
.
├── .git/                     # Git repository data (usually hidden)
├── .flake8                   # Flake8 configuration file
├── pyproject.toml            # Python project configuration
├── requirements.txt          # List of required Python libraries
├── README.md                 # This file
├── configs/                  # Configuration files
│   └── config.yaml           # Main configuration file (paths, hyperparameters)
├── checkpoints/              # Stores base model checkpoint files (created during training)
├── rnn_checkpoints/          # Stores RNN model checkpoint files (created during training)
├── rnn_processed_data/       # Processed data for RNN (created by generate_rnn_data.py)
├── scripts/                  # Scripts to run different steps of the pipeline
│   ├── preprocess_raw_data.py # Preprocess original videos -> frames npz, annotations json
│   ├── extract_pose_features.py # Extract pose features -> pose npz
│   ├── train_base_model.py   # Train the base TemporalActionDetector model
│   ├── generate_rnn_data.py  # Generate RNN input data from base model results
│   ├── train_rnn.py          # Train the RNN post-processor model
│   └── evaluate_pipeline.py  # Evaluate the entire pipeline (base + RNN)
└── src/                      # Main source code (model definitions, dataloader, utils, ...)
    ├── models/               # Model class definitions
    ├── utils/                # Utility functions (helpers, metrics, postprocessing, ...)
    ├── dataloader.py         # Dataset and DataLoader definitions for the base model
    ├── evaluation.py         # Function for calculating aggregate metrics
    └── losses.py             # Loss function class definitions
```

**(Note:** The `Data/` and `logs/` directories mentioned in the setup steps are expected to be created manually or by running the preprocessing/training scripts respectively. They are not part of the initial repository structure).**

## Installation

1.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
2.  **Install libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare data:**
    *   Place the original MERL Shopping Dataset videos and labels into `Data/Videos_MERL_Shopping_Dataset` and `Data/Labels_MERL_Shopping_Dataset` respectively, following the train/test structure.
    *   The preprocessing scripts will generate data in `Data/full_videos`.
4.  **Configure the pipeline:**
    *   Edit `configs/config.yaml` to set paths, hyperparameters, and other settings for the different components (data loading, base model, RNN model, training parameters). Refer to the comments within the file for guidance.

## Usage

Run the scripts in the `scripts/` directory in the following order:

1.  **Preprocess Data:** Convert original videos to the required format. Reads configuration from `configs/config.yaml`.
    ```bash
    python scripts/preprocess_raw_data.py
    ```
    *(Note: Ensure `configs/config.yaml` specifies the correct raw data paths and desired output settings).*

2.  **Extract Pose Features:** Reads configuration from `configs/config.yaml`.
    ```bash
    # Process all splits defined in config (train, val, test)
    python scripts/extract_pose_features.py --split all 
    # Or process specific splits
    # python scripts/extract_pose_features.py --split train --split val
    # Optional: Specify a different config file
    # python scripts/extract_pose_features.py --split all --config path/to/other_config.yaml 
    ```
    *(Assuming `--split` and `--config` arguments exist as per standard practice, though not fully visible in analysis).*

3.  **Train Base Model:** Reads all settings from `configs/config.yaml`.
    ```bash
    python scripts/train_base_model.py
    ```
    *   Checkpoints are saved based on the `checkpoint_dir` in the config.
    *   Modify `configs/config.yaml` to change hyperparameters, paths, etc.

4.  **Generate RNN Data:** Use the trained base model to create input for the RNN. Reads paths and settings from config.
    ```bash
    python scripts/generate_rnn_data.py --config configs/config.yaml
    # Optional: Override the base model checkpoint path from the config
    # python scripts/generate_rnn_data.py --config configs/config.yaml --checkpoint_path checkpoints/your_specific_model.pth
    ```
    *   The base model checkpoint path must be set either in the config file (`base_model_training -> checkpoint_path`) or via the `--checkpoint_path` argument.
    *   Data will be saved in `rnn_processed_data/`. A pickle file containing raw inference results will also be created (e.g., `train_inference_raw.pkl`, `val_inference_raw.pkl`).

5.  **Train RNN Post-processor:** Reads all settings from `configs/config.yaml`.
    ```bash
    python scripts/train_rnn.py --config configs/config.yaml
    ```
    *   Checkpoints are saved based on the `checkpoint_dir` in the config (under `data`).
    *   Modify `configs/config.yaml` (`rnn_model` and `rnn_training` sections) to change hyperparameters, paths, etc.

6.  **Evaluate Pipeline:** Evaluate the combined performance. Reads settings from config.
    ```bash
    python scripts/evaluate_pipeline.py --config configs/config.yaml
    # Example: Override specific paths needed for evaluation
    # python scripts/evaluate_pipeline.py --config configs/config.yaml --rnn_checkpoint_path rnn_checkpoints/best_rnn_model.pth --inference_output_path logs/val_inference_raw.pkl
    # Example: Evaluate and visualize a specific video
    # python scripts/evaluate_pipeline.py --config configs/config.yaml --visualize_video_id video123 
    ```
    *   Requires the inference results file (`.pkl`) from step 4 and the RNN checkpoint from step 5. These paths must be set either in the config or via the `--inference_output_path` and `--rnn_checkpoint_path` arguments.
    *   See `scripts/evaluate_pipeline.py --help` for all evaluation and visualization options.

## Configuration

Project configuration (data paths, hyperparameters for base and RNN models, training settings) is managed through the `configs/config.yaml` file.

Key sections include:
*   `global`: Global settings like number of classes.
*   `data`: Paths for datasets, checkpoints, logs, etc.
*   `base_model`: Architecture details for the base temporal action detector.
*   `base_model_training`: Hyperparameters for training the base model.
*   `rnn_model`: Architecture details for the RNN post-processor.
*   `rnn_training`: Hyperparameters for training the RNN model.

Command-line arguments can be used with some scripts (e.g., `train_base_model.py`, `train_rnn.py`) to override specific settings defined in the config file. Run scripts with `-h` or `--help` to see available command-line options.

