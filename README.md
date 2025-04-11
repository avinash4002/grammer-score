

# DeBERTa-v3-Large Fine-Tuning for Grammar Scoring (1–5 Scale with 0.5 Intervals)

This repository contains the full pipeline for fine-tuning `deberta-v3-large` on a grammar scoring task. The goal is to predict scores between **1.0 to 5.0**, at **0.5 intervals**, treated as a **9-class classification problem**.

---

## 📁 Directory Structure

```
.
├── finetune-deberta.ipynb           # Fine-tunes the model using augmented dataset
├── model-inference.ipynb            # Loads saved model and performs predictions on test data
├── model-pipeline.ipynb             # Demonstrates full pipeline: text ➝ prediction
├── test_labelsPrediction.csv        # CSV file containing model predictions on test data
├── transcription.ipynb              # Auxiliary tasks or audio-related experiments (optional)
```

---

## 🧩 Full Process

### 1. 📊 Dataset Overview

- **Original Dataset**: ~900 samples
- **Label Range**: `[1.0, 1.5, ..., 5.0]` (mapped to integer class indices `0-8`)
- **Task**: Classification (multi-class with 9 labels)
- **Balance**: Dataset is initially balanced per label

---

### 2. 🧪 Preprocessing & Augmentation

To increase training data and improve generalization:

- **Augmentation Library**: [`nlpaug`](https://github.com/makcedward/nlpaug)
- **Techniques Used**:
  - Synonym replacement (WordNet)
  - Random word insertion
  - Random word swap
  - Contextual word replacement (optional, using BERT-like models)

Augmented samples were **added to the original dataset** to create a richer training set before fine-tuning.

---

### 3. 🧠 Model Details

- **Base Model**: [`deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
- **Architecture**: `DebertaV2ForSequenceClassification`
- **Head**: Classification (9 output logits for 9 labels)
- **Label Mapping**:
  ```
  1.0 → 0, 1.5 → 1, ..., 5.0 → 8
  ```

---

### 4. 🛠 Fine-Tuning Setup

- **Training Framework**: Hugging Face Transformers
- **Optimizer Settings**:
  - Learning Rate: `2e-5`
  - Epochs: `3–5`
  - Batch Size: 1 (with gradient accumulation)
  - Weight Decay: `0.01`
- **Evaluation Metric**: Pearson Correlation (`pearsonr`)
- **Additional Metrics**: Accuracy

TrainingArguments sample:
```python
TrainingArguments(
  evaluation_strategy="epoch",
  save_strategy="epoch",
  metric_for_best_model="pearsonr",
  greater_is_better=True,
  per_device_train_batch_size=1,
  gradient_accumulation_steps=4,
  num_train_epochs=5,
  fp16=True,
  load_best_model_at_end=True
)
```

---

### 5. 💾 Model Saving

The best checkpoint (based on validation Pearson) is:
- Saved in a local directory
- Exported to a Kaggle Dataset for reuse

---

### 6. 🚀 Inference Pipeline

The model is loaded from the Kaggle dataset and used to generate predictions:

Steps:
1. Load model and tokenizer
2. Preprocess test text
3. Predict class logits
4. Map predicted class index (0-8) back to original label (1.0–5.0)
5. Store results in `test_labelsPrediction.csv`

---

### 7. 📈 Results

- **Validation Pearson**: ~0.90
- **Test Pearson**: ~0.70
- Predictions include both accuracy and Pearson for evaluation

---

## 🔧 Requirements

```bash
pip install transformers datasets evaluate nlpaug scikit-learn
```

---

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Microsoft DeBERTa-v3](https://huggingface.co/microsoft/deberta-v3-large)
- [NLP Augmentation Toolkit (nlpaug)](https://github.com/makcedward/nlpaug)
- [Kaggle](https://www.kaggle.com/) for compute and dataset storage

---

## 📬 Contact

Feel free to open issues or reach out for questions, improvements, or ideas!
