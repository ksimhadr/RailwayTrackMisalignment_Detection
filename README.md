# RailwayTrackMisalignment_Detection
Railway Track Misalignment Detection with pretrained networks and its performance comparison
# 🚆 Railway Track Misalignment Detection using Deep Learning

This project uses deep learning to detect **misalignments in railway tracks** from image data. Ensuring track alignment is critical to railway safety, and this model helps automate the inspection process using visual data.

---

## 🧠 Project Summary

- **Goal**: Automatically classify railway track images as `aligned` or `misaligned`.
- **Approach**: Fine-tuned deep convolutional networks (VGG16, ResNet50, InceptionV3, EfficientNetB0).
- **Best Performance**: Achieved **98.2% test accuracy** using **VGG16** with transfer learning.

---

---

## 📸 Dataset

- Contains real-world or synthetic images of railway tracks.
- Two classes: aligned and misaligned.
- Images are preprocessed to size *224x224 RGB*.

> ⚠️ Data not included in this repo due to restricted access

---

---
## 🧠 Models Used

| Model           | Accuracy (%) | Notes |
|----------------|--------------|-------|
| VGG16          | *98.2*     | Best performer |
| ResNet50       | 98.2         | Good generalization |
| InceptionV3    | 94.5         | Deeper, slower training |
| EfficientNetB0 | 92.7         | Lightweight, fast |

- All models use *transfer learning* (ImageNet weights) and *fine-tuning*.

---

Dataset Overview The Railway Track Misalignment Detection dataset consists of two image classes: Normal, Defective

Pre-trained Networks Used We evaluated and fine-tuned the following ImageNet-pretrained models using torchvision.models:

VGG16, ResNet50, InceptionV3, EfficientNet-B0

Training Setup Data Augmentation:
Resized to 224x224 (299x299 for InceptionV3)
Random horizontal, vertical flips
Random rotations (±15-30°)
Gaussian Blur applied
Color jittering
Standard ImageNet normalization
Training Details:

Optimizer: AdamW
Scheduler: CosineAnnealingLR
Loss Function: CrossEntropyLoss with label smoothing = 0.1
Regularization: MixUp (α = 0.2), Gradient Clipping
Epochs: 20
Batch Size: 32
Device: GPU (CUDA)
Architecture Modifications
Model	Final Layer Modified
VGG16	classifier[6] → Linear(4096, 2)
ResNet50	fc → Linear(2048, 2)
InceptionV3	fc → Linear(2048, 2)
AuxLogits.fc → Linear(768, 2)
EfficientNet-B0	classifier[1] → Linear(1280, 2)

