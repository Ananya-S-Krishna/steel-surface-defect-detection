# Steel Surface Defect Detection using Deep Learning

This project presents an **AI-powered visual inspection system** designed to detect **surface defects in hot-rolled steel strips**, combining **mechanical domain expertise** with **deep learning-based image classification**.

---

## Problem Statement

In industries relying on **cold and hot rolling processes**, surface defects can severely affect the **functional performance**, **visual appeal**, and **structural integrity** of steel sheets. Traditional **manual inspection** methods are labor-intensive and error-prone.

To address this, we built a **deep learning solution** to classify defects automatically using steel surface images. This can help in:

- Reducing **downtime and cost** associated with defective batches  
- Enhancing **manufacturing quality assurance**  
- Enabling **real-time fault detection** in production lines

## Dataset

- **Source**: NEU Surface Defect Database (NEU-DET) (Kaggle)
- **Total Images**: 1800 grayscale images (converted to RGB)
- **Image Size**: 200×200 pixels
- **Defect Categories**:
  1. **Crazing**
  2. **Inclusion**
  3. **Patches**
  4. **Pitted Surface**
  5. **Rolled-in Scale**
  6. **Scratches**

- **Split**:
  - **Training**: 1440 images  
  - **Validation**: 360 images (60 per class)

## Deep Learning Workflow

- **Framework**: PyTorch  
- **Architecture**: Transfer Learning using **ResNet18** pretrained on ImageNet  
- **Modifications**:
  - Final layer adapted to 6 defect classes  
  - Early layers frozen for feature reuse  
- **Loss Function**: Cross Entropy Loss  
- **Optimizer**: Adam  
- **Training Duration**: 10 epochs  
- **Hardware**: Trained on **NVIDIA RTX 3050 GPU**


## Model Performance

| Metric              | Value        |
|---------------------|--------------|
| Validation Accuracy | **95.28%**   |
| F1-score (Avg)      | **0.95**     |
| Best Class Accuracy | **100%**     |
| Worst Class F1      | 0.86 (Inclusion) |

Confusion matrix and classification report were generated for deeper evaluation.


## Features & Highlights

- Developed an **automated fault detection system** to assist steel inspection engineers  
- Integrated **transfer learning** to reduce training time and improve generalization  
- Designed **real-time inference function** for practical usability  
- Handled **balanced data loading**, **custom augmentation**, and GPU utilization

---

## Sample Inference

```python
image_path = "path/to/defective_steel_image.jpg"
prediction = predict_defect(image_path, model, device)
print("Predicted Defect Type:", prediction)
