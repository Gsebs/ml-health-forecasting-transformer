# Transformer-based RUL Prediction

This project implements a Transformer-based model for predicting the Remaining Useful Life (RUL) of aircraft engines using the NASA C-MAPSS dataset.

## Project Structure

```plaintext
├── data/                   # Data directory
│   ├── artifacts/          # Generated artifacts (scalers, processed data)
│   ├── test_FD001.txt      # Raw test data
│   ├── train_FD001.txt     # Raw training data
│   └── RUL_FD001.txt       # Ground truth RUL
├── src/                    # Source code
│   ├── model.py            # Shared Transformer model definition
│   └── generate_features_list.py # Helper to save feature configuration
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── feature_engineering.py  # Feature engineering script
└── requirements.txt        # Python dependencies
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    Ensure your raw data files (`train_FD001.txt`, etc.) are in the `data/` folder.

## Reproducible Pipeline

Follow these steps to train and evaluate the model from scratch.

### 1. Feature Engineering
Process the raw data, generate artifacts (scalers, `.npy` files), and save the feature configuration.

```bash
# Generates data/artifacts/ and src/input_features.json
python feature_engineering.py
python src/generate_features_list.py
```

### 2. Training
Train the Transformer model. This will save checkpoints to `training_logs/transformer_baseline/`.

```bash
python train.py --epochs 30 --batch_size 64
```
*Output: `training_logs/transformer_baseline/best_model.pth`*

### 3. Evaluation
Evaluate the best model on the test set. This script handles accurate data alignment and produces an RMSE score.

```bash
python evaluate.py training_logs/transformer_baseline/best_model.pth
```
*Output: `evaluation_results.png` (Plot of Predicted vs Actual RUL)*

## Model Architecture

The model is a pure PyTorch implementation defined in `src/model.py`. It consists of:
- **Input Projection**: Linear layer to project features to `d_model`.
- **Positional Encoding**: Sinusoidal encodings to retain temporal information.
- **Transformer Encoder**: Stack of Transformer layers to capture sequential dependencies.
- **Global Average Pooling**: Aggregates time-step outputs.
- **Decoder**: MLP to predict the final RUL value.
