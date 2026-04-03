# 🎭 MVSA Multimodal Sentiment — Streamlit App

A modern, dark-themed two-page Streamlit application for the MVSA-Single multimodal sentiment model.

## Architecture
**BERT-base-uncased** (text) + **ResNet-50** (image) → **Gated Cross-Attention Fusion** → 3-class classifier  
Classes: **Negative · Neutral · Positive**

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your assets
Copy the saved training plots into the `assets/` folder:
```
assets/
├── training_curves.png       # from outputs/training_curves.png
├── confusion_matrix.png      # from outputs/confusion_matrix.png
├── class_distribution.png    # from outputs/class_distribution.png
├── sample_images.png         # sample images per class
└── inference_demo.png        # random test samples demo
```
These are **already included** if you downloaded the full app package.

### 3. Run the app
```bash
streamlit run app.py
```

---

## Pages

### 🔮 Predict
- Upload any image (JPG/PNG/WEBP)
- Enter accompanying text (tweet, caption, post)
- Upload your `best_model.pt` checkpoint via the sidebar
- Get sentiment prediction with confidence + per-class probability bars

### 📊 Training Details
- Model architecture breakdown (Text/Image/Fusion/Classifier branches)
- Full training configuration table
- Epoch-by-epoch training history (interactive dataframe)
- Training curves (loss · accuracy · F1)
- Confusion matrix (counts + normalised)
- Per-class precision / recall / F1 / support cards
- Class distribution chart & sample images
- Dataset tags and deployment package info

---

## Model Checkpoint
The app expects a PyTorch checkpoint (`best_model.pt`) saved with:
```python
torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_f1': val_f1}, path)
```
Upload it via the **sidebar** on the Predict page.

---

## Results
| Metric | Value |
|---|---|
| Test Accuracy | 69.63% |
| Test Weighted F1 | 0.6968 |
| Test Macro F1 | 0.6910 |
| Best Val F1 | 0.7006 (epoch 7) |
| Total Parameters | 136M |
| Trainable Parameters | ~40M (29.4%) |
