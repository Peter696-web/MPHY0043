# MPHY0043 Surgical Phase Prediction

## AI Usage Declaration
This project was developed with the assistance of AI tools (e.g., GitHub Copilot, LLMs) for code generation, debugging, and documentation. 

## 1. Directory Structure

```
MPHY0043/
├── cholec80/             # Raw dataset (videos/frames)
├── data/
│   ├── features/           # Extracted video features (e.g., dinov2)
│   ├── labels/             # Raw and aligned labels
│   │   └── aligned_labels/ # JSON labels synced with features
│   ├── processed/          # Preprocessed tensors for training
│   ├── estimated_times/    # Generated trajectory predictions for Task B
│   └── split_manifest.json # Data split configuration
├── results/
│   └── models/             # Checkpoints, logs, and prediction outputs
├
├── src/
│   ├── model/              # Model architecture, datasets, loss functions
│   ├── prepare/            # Preprocessing and generation scripts
│   ├── utils/              # Visualization utilities
│   ├── train_A.py          # Training script for Task A
│   ├── train_B.py          # Training script for Task B
│   ├── test_A.py           # Testing script for Task A
│   └── test_B.py           # Testing script for Task B
└── README.md
```

## 2. Usage Workflow

### Step 1: Data Preparation

Run the following scripts in order to build labels, generate splits, extract features, and align data.

```bash
# 1. Build labels from raw annotations
python src/prepare/new_label.py

# 2. Generate training/validation/test split manifest
python src/prepare/make_split.py --output data/split_manifest.json

# 3. Extract features (if not already done)
# Refer to src/prepare/feature_extraction.py for arguments
python src/prepare/feature_extraction.py \
    --dataset cholec80 \
    --data_root ./cholec80 \
    --output_dir ./data/features \
    --models base \
    --split_manifest data/split_manifest.json

# 4. Align features and labels into processed format
python src/prepare/preprocess.py
```
*Output: Processed data is saved to `data/processed` and `data/labels/aligned_labels`.*

### Step 2: Train Task A (Multi-task Baseline)

Train the model to predict both surgical phases and future remaining time (schedule).

```bash
python src/train_A.py \
    --exp_name task_A \
    --save_dir results/models/task_A \
    --epochs 50 \
    --batch_size 1
```

### Step 3: Generate Trajectories for Task B

Use the best performing Task A model (Best MAE) to predict time schedules for the entire dataset. These predictions serve as inputs for Task B.

```bash
python src/prepare/generate_trajectory.py \
    --checkpoint results/models/task_A/checkpoint_best_mae.pth \
    --output_dir data/estimated_times
```

### Step 4: Train Task B (Phase Prediction)

Train the dedicated phase prediction model. You can run two variations: without external time injection (baseline) and with it (proposed method).

**Option 1: Without External Injection (Baseline)**
```bash
python src/train_B.py \
    --exp_name task_B_no_injection \
    --save_dir results/models/task_B_no_injection \
    --use_external_time_input 0
```

**Option 2: With External Injection (Proposed)**
```bash
python src/train_B.py \
    --exp_name task_B_with_injection \
    --save_dir results/models/task_B_input_injection \
    --use_external_time_input 1
```

### Step 5: Testing

Evaluate the trained models on the test set.

**Test Task A:**
```bash
python src/test_A.py \
    --checkpoint results/models/task_A/checkpoint_best_mae.pth \
    --split test
```

**Test Task B:**
```bash
python src/test_B.py \
    --checkpoint results/models/task_B_with_injection/checkpoint_best_f1.pth \
    --split test
```

### Step 6: Visualization

Generate visualization plots for model predictions alongside ground truth. Ensure you point to the `.pt` prediction files generated during testing/validation.

**Visualize Task A:**
```bash
python src/utils/visualize_video_A.py \
    --pred_path results/models/task_A/test_predictions.pt \
    --batch \
    --split test
```

**Visualize Task B:**
```bash
python src/utils/visualize_video_B.py \
    --pred_path results/models/task_B_with_injection/test_predictions.pt \
    --batch \
    --split test
```

## 3. Environment
Before running the code, please ensure all dependencies are installed. Refer to `requirements.txt` for the specific package versions.

```bash
pip install -r requirements.txt
```


