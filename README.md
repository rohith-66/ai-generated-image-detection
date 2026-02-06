# AI-Generated Image Detection

## Overview
This project builds a machine-learning system to distinguish real photographs from AI-generated images.
The goal is to address real-world problems such as misinformation, fake news, scams, and manipulated media,
where AI-generated images are increasingly indistinguishable from real ones.

This is a final-year Data Science capstone project designed to be reproducible, scalable,
defensible in viva, and recruiter-friendly.

--------------------------------------------------

## Problem Definition
Task: Binary image classification
Label 0 → Real image
Label 1 → AI-generated image

Challenge:
Modern generative models (e.g., Stable Diffusion) produce highly realistic images.

Solution:
Train and evaluate deep learning models that learn discriminative patterns
between real and AI-generated images.

--------------------------------------------------

## Datasets (Locked)

MS COCO 2017 — Real Images
Source: Official COCO dataset
Usage:
- train2017 → training
- val2017 → validation
Image count:
- Train: ~118,000
- Validation: 5,000

GenImage — AI-Generated Images
Source: GenImage dataset (Stable Diffusion subset)
Official release hosted on Baidu Yunpan (not globally accessible)
Accessed via public mirrors
Final subset used: 10,000 images
Generator: Stable Diffusion

--------------------------------------------------

## Dataset Storage (IMPORTANT)

WARNING: Datasets are NOT stored in this GitHub repository.

All images are stored externally in a shared Google Drive folder:

AI_Image_Detection_Data/
├── coco/
│   ├── train2017/
│   └── val2017/
│
├── genimage/
│   └── stable_diffusion/
│
├── index/
│   └── dataset_index.csv
│
└── checkpoints/

GitHub contains ONLY code and documentation.

--------------------------------------------------

## Dataset Index (Single Source of Truth)

All experiments rely on:

AI_Image_Detection_Data/index/dataset_index.csv

CSV columns:
filepath, label, source, generator, split

label:
0 = real
1 = AI-generated

split:
train / val / test

Balanced training set:
- 10,000 real images (sampled from COCO train)
- 10,000 AI-generated images (GenImage)
- 5,000 real validation images (COCO val)

Balancing is handled via the CSV, not by deleting image files.

--------------------------------------------------

## Environment Setup

Recommended platform:
- Google Colab
- Python version ≥ 3.9

Clone repository:
git clone https://github.com/<USERNAME>/ai-generated-image-detection.git
cd ai-generated-image-detection

Install dependencies:
pip install -r requirements.txt

Mount Google Drive (Colab):
from google.colab import drive
drive.mount('/content/drive')

--------------------------------------------------

## Project Workflow

Stage 1 — Data Collection (Completed)
- Download MS COCO 2017 (train + val)
- Download GenImage (Stable Diffusion subset)
- Verify image integrity
- Convert all images to RGB JPEG
- Create unified dataset index
- Balance training data via CSV sampling

Stage 2 — Data Preprocessing
- Resize images to 224 x 224
- Normalize using ImageNet statistics
- Apply data augmentation to training images only
- Create train / validation / test splits
- Build PyTorch DataLoader
- Visually inspect random batches

Stage 3 — Modeling
- Baseline CNN
- Transfer learning (ResNet / EfficientNet)
- Balanced training
- Robust evaluation

Stage 4 — Analysis & Interpretability
- Error analysis
- Grad-CAM / saliency maps
- Testing on unseen AI-generated images

--------------------------------------------------

## Preprocessing Decisions
Input size: 224 x 224
Color format: RGB
Normalization: ImageNet mean and standard deviation
Augmentation: training data only (horizontal flip, rotation)

--------------------------------------------------

## Reproducibility
- CSV-driven pipeline
- No file duplication
- Fixed random seeds where applicable
- No hard-coded paths

--------------------------------------------------

## Notes
- Do NOT upload datasets to GitHub
- Do NOT move or duplicate image files
- All experiments must read from dataset_index.csv

--------------------------------------------------

## License
Academic and educational use only.
