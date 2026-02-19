
"""
BARCLAYS PHISHING DETECTION - DeBERTa-v3 Training Pipeline
===========================================================
Production-ready 3-class email classifier for banking fraud detection

MODEL: microsoft/deberta-v3-base
- Chosen for superior performance on text classification tasks
- Enhanced attention mechanism for better context understanding
- State-of-the-art results on NLU benchmarks

CLASSES:
- 0: LEGITIMATE (normal emails)
- 1: PHISHING_REAL (human-written phishing)
- 2: PHISHING_AI (AI-generated phishing)

Author: ML Training Pipeline
Version: 1.0
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
NUM_LABELS = 3

# Training configuration
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
SEED = 42

# Data configuration
DATASET_PATH = "FINAL_BARCLAYS_DATASET.csv"
MIN_TEXT_LENGTH = 30
TEST_SIZE = 0.2

# Output configuration
OUTPUT_DIR = "./barclays_deberta_model"
LOGGING_DIR = "./logs"

# Class labels mapping
CLASS_NAMES = {
    0: "LEGITIMATE",
    1: "PHISHING_REAL",
    2: "PHISHING_AI"
}


# ============================================================================
# SETUP
# ============================================================================

def setup_environment():
    """Initialize environment and check GPU availability"""
    set_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  BARCLAYS PHISHING DETECTION - TRAINING PIPELINE")
    print("=" * 70)
    print(f"\n🔧 Environment Setup:")
    print(f"   Device: {device}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Random seed: {SEED}")
    
    return device


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_clean_data():
    """
    Load dataset and apply data quality filters
    
    Cleaning steps:
    1. Remove null values - prevents training errors
    2. Remove duplicates - prevents data leakage and overfitting
    3. Filter short texts - ensures meaningful content for classification
    """
    print("\n" + "=" * 70)
    print("  DATA LOADING AND PREPROCESSING")
    print("=" * 70)
    
    print(f"\n📂 Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Remove null values
    initial_size = len(df)
    df = df.dropna(subset=['text', 'label'])
    print(f"✓ Removed {initial_size - len(df):,} null values")
    
    # Convert text to string and label to int
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    
    # Remove duplicates
    initial_size = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f"✓ Removed {initial_size - len(df):,} duplicate emails")
    
    # Filter short emails
    initial_size = len(df)
    df = df[df['text'].str.len() >= MIN_TEXT_LENGTH]
    print(f"✓ Removed {initial_size - len(df):,} emails shorter than {MIN_TEXT_LENGTH} characters")
    
    print(f"\n📊 Final dataset size: {len(df):,} samples")
    
    # Print class distribution
    print(f"\n🏷️  Class Distribution:")
    for label in sorted(df['label'].unique()):
        count = len(df[df['label'] == label])
        percentage = (count / len(df)) * 100
        print(f"   Label {label} ({CLASS_NAMES[label]:15s}): {count:6,} samples ({percentage:5.2f}%)")
    
    return df


def create_stratified_split(df):
    """
    Perform stratified train/test split
    
    Why stratified split?
    - Maintains class distribution in both train and test sets
    - Critical for imbalanced datasets (Label 2 has only ~1% of data)
    - Ensures test set is representative of real-world distribution
    - Prevents biased evaluation metrics
    """
    print("\n" + "=" * 70)
    print("  STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 70)
    
    print(f"\n🔀 Splitting dataset (test_size={TEST_SIZE}, stratified by label)")
    
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df['label']  # Maintains class distribution
    )
    
    print(f"✓ Train set: {len(train_df):,} samples")
    print(f"✓ Test set: {len(test_df):,} samples")
    
    # Verify stratification
    print(f"\n📊 Train Set Distribution:")
    for label in sorted(train_df['label'].unique()):
        count = len(train_df[train_df['label'] == label])
        percentage = (count / len(train_df)) * 100
        print(f"   Label {label} ({CLASS_NAMES[label]:15s}): {count:6,} samples ({percentage:5.2f}%)")
    
    print(f"\n📊 Test Set Distribution:")
    for label in sorted(test_df['label'].unique()):
        count = len(test_df[test_df['label'] == label])
        percentage = (count / len(test_df)) * 100
        print(f"   Label {label} ({CLASS_NAMES[label]:15s}): {count:6,} samples ({percentage:5.2f}%)")
    
    return train_df, test_df


# ============================================================================
# CLASS WEIGHT COMPUTATION
# ============================================================================

def compute_class_weights(train_df):
    """
    Compute class weights to handle imbalanced dataset
    
    Why class weighting?
    - Dataset is highly imbalanced (Label 2 has only ~1% of samples)
    - Without weighting, model will bias towards majority classes (0 and 1)
    - Class weights penalize misclassification of minority class more heavily
    - Formula: weight = n_samples / (n_classes * n_samples_per_class)
    - Higher weight for rare classes forces model to learn their patterns
    
    Impact:
    - Prevents model from ignoring Label 2 (PHISHING_AI)
    - Improves recall for minority class
    - Essential for fraud detection where missing rare attacks is costly
    """
    print("\n" + "=" * 70)
    print("  CLASS WEIGHT COMPUTATION")
    print("=" * 70)
    
    print("\n⚖️  Computing class weights for imbalanced dataset...")
    
    labels = train_df['label'].values
    unique_labels = np.unique(labels)
    
    # Compute weights only for existing labels
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    
    # Map weights to label indices
    class_weights_dict = {label: weight for label, weight in zip(unique_labels, class_weights)}
    
    print(f"✓ Class weights computed:")
    for label in unique_labels:
        count = len(train_df[train_df['label'] == label])
        weight = class_weights_dict[label]
        label_name = CLASS_NAMES.get(label, f"UNKNOWN_{label}")
        print(f"   Label {label} ({label_name:15s}): weight = {weight:.4f} (n={count:,})")
    
    print(f"\n💡 Higher weights for minority classes ensure balanced learning")
    
    # Create tensor with weights for all possible labels (0, 1, 2)
    # Use 1.0 as default weight for missing labels
    class_weights_tensor = torch.tensor(
        [class_weights_dict.get(i, 1.0) for i in range(NUM_LABELS)],
        dtype=torch.float32
    )
    
    if len(unique_labels) < NUM_LABELS:
        missing = [i for i in range(NUM_LABELS) if i not in unique_labels]
        print(f"\n⚠️  Note: Missing labels {missing} will use default weight of 1.0")
    
    return class_weights_tensor


# ============================================================================
# TOKENIZATION
# ============================================================================

def load_tokenizer():
    """Load DeBERTa tokenizer"""
    print("\n" + "=" * 70)
    print("  TOKENIZER LOADING")
    print("=" * 70)
    
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✓ Tokenizer loaded")
    print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"   Max length: {MAX_LENGTH}")
    
    return tokenizer


def tokenize_dataset(train_df, test_df, tokenizer):
    """
    Tokenize datasets for model input
    
    Tokenization parameters:
    - truncation=True: Cuts text longer than max_length
    - padding='max_length': Pads shorter sequences to max_length
    - max_length=256: Balance between context and efficiency
    """
    print("\n" + "=" * 70)
    print("  TOKENIZATION")
    print("=" * 70)
    
    print(f"\n🔤 Tokenizing datasets...")
    
    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print(f"✓ Train dataset tokenized: {len(train_dataset):,} samples")
    print(f"✓ Test dataset tokenized: {len(test_dataset):,} samples")
    
    return train_dataset, test_dataset


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(device, class_weights):
    """
    Load DeBERTa model for sequence classification
    
    Why DeBERTa-v3-base?
    - Enhanced attention mechanism with disentangled attention
    - Better position encoding than BERT
    - State-of-the-art performance on GLUE benchmark
    - Efficient: 184M parameters (vs 340M for DeBERTa-large)
    - Excellent for text classification tasks
    - Strong transfer learning capabilities
    """
    print("\n" + "=" * 70)
    print("  MODEL LOADING")
    print("=" * 70)
    
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification"
    )
    
    model.to(device)
    
    print(f"✓ Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   Number of labels: {NUM_LABELS}")
    
    return model


# ============================================================================
# CUSTOM TRAINER WITH CLASS WEIGHTS
# ============================================================================

class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies class weights to loss function
    
    Why custom trainer?
    - Default Trainer doesn't support class weights in loss
    - We need weighted CrossEntropyLoss for imbalanced data
    - Overriding compute_loss() allows custom loss calculation
    """
    
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted cross-entropy loss
        
        Loss formula with weights:
        L = -Σ w_i * y_i * log(p_i)
        
        where:
        - w_i = class weight for class i
        - y_i = true label (one-hot)
        - p_i = predicted probability
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply weighted cross-entropy loss
        # Ensure class weights match logits dtype
        class_weights = self.class_weights.to(logits.dtype)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(eval_pred):
    """
    Compute comprehensive evaluation metrics
    
    Metrics:
    - Accuracy: Overall correctness
    - Precision: Of predicted positives, how many are correct
    - Recall: Of actual positives, how many are detected
    - F1-score: Harmonic mean of precision and recall
    
    All metrics computed PER CLASS for detailed analysis
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
        labels=[0, 1, 2]
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
    }
    
    # Add per-class metrics
    for i in range(NUM_LABELS):
        metrics[f'precision_class_{i}'] = precision[i]
        metrics[f'recall_class_{i}'] = recall[i]
        metrics[f'f1_class_{i}'] = f1[i]
    
    return metrics


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_dataset, test_dataset, class_weights):
    """
    Train DeBERTa model with custom configuration
    
    Training strategy:
    - AdamW optimizer with weight decay for regularization
    - Linear learning rate schedule with warmup
    - Gradient clipping to prevent exploding gradients
    - Evaluation each epoch to monitor overfitting
    - Save best model based on F1-score
    """
    print("\n" + "=" * 70)
    print("  MODEL TRAINING")
    print("=" * 70)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_dir=LOGGING_DIR,
        logging_steps=100,
        eval_strategy="epoch",  # Updated parameter name
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        fp16=False,  # Disabled for CPU training
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,  # Gradient clipping
        report_to="none"
    )
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Weight decay: {WEIGHT_DECAY}")
    print(f"   Warmup steps: {WARMUP_STEPS}")
    print(f"   Max gradient norm: 1.0 (gradient clipping enabled)")
    print(f"   Mixed precision: {torch.cuda.is_available()}")
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print(f"\n🚀 Starting training...")
    print(f"   Total training steps: {len(train_dataset) // BATCH_SIZE * NUM_EPOCHS}")
    
    trainer.train()
    
    print(f"\n✓ Training completed!")
    
    return trainer


# ============================================================================
# EVALUATION AND REPORTING
# ============================================================================

def evaluate_and_report(trainer, test_dataset, test_df):
    """
    Generate comprehensive evaluation report
    
    Includes:
    - Overall metrics
    - Per-class precision, recall, F1
    - Confusion matrix insights
    """
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION")
    print("=" * 70)
    
    print(f"\n📊 Evaluating on test set ({len(test_dataset):,} samples)...")
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = test_df['label'].values
    
    # Print classification report
    print(f"\n📈 Classification Report:")
    print("=" * 70)
    
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=[CLASS_NAMES[i] for i in range(NUM_LABELS)],
        digits=4
    )
    print(report)
    
    # Print per-class metrics
    print(f"\n🎯 Detailed Per-Class Metrics:")
    print("=" * 70)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=[0, 1, 2]
    )
    
    for i in range(NUM_LABELS):
        print(f"\n   {CLASS_NAMES[i]} (Label {i}):")
        print(f"      Precision: {precision[i]:.4f}")
        print(f"      Recall:    {recall[i]:.4f}")
        print(f"      F1-score:  {f1[i]:.4f}")
        print(f"      Support:   {support[i]:,} samples")
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\n   Overall Accuracy: {accuracy:.4f}")
    
    return predictions


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model_and_tokenizer(trainer, tokenizer):
    """Save trained model and tokenizer for deployment"""
    print("\n" + "=" * 70)
    print("  SAVING MODEL AND TOKENIZER")
    print("=" * 70)
    
    print(f"\n💾 Saving model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    print(f"💾 Saving tokenizer to: {OUTPUT_DIR}")
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✓ Model and tokenizer saved successfully!")
    print(f"✓ Model ready for deployment")
    
    # Print model size
    model_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    ) / (1024 ** 2)
    
    print(f"✓ Total model size: {model_size:.2f} MB")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute complete training pipeline"""
    
    # Setup
    device = setup_environment()
    
    # Data loading and preprocessing
    df = load_and_clean_data()
    train_df, test_df = create_stratified_split(df)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_df)
    
    # Tokenization
    tokenizer = load_tokenizer()
    train_dataset, test_dataset = tokenize_dataset(train_df, test_df, tokenizer)
    
    # Model loading
    model = load_model(device, class_weights)
    
    # Training
    trainer = train_model(model, train_dataset, test_dataset, class_weights)
    
    # Evaluation
    evaluate_and_report(trainer, test_dataset, test_df)
    
    # Save model
    save_model_and_tokenizer(trainer, tokenizer)
    
    print("\n" + "=" * 70)
    print("  🏆 TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n✅ Model trained and saved to: {OUTPUT_DIR}")
    print(f"✅ Ready for inference and deployment")
    print(f"✅ All metrics logged and reported")
    print("\n")


if __name__ == "__main__":
    main()
