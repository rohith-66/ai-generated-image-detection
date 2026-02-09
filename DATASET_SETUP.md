# Dataset Setup Guide

This document explains how datasets are stored and accessed for the
AI-Generated Image Detection project.

--------------------------------------------------

## Dataset Storage Location

All datasets are stored externally in a shared Google Drive folder.

Folder name:
AI_Image_Detection_Data

NOTE:
Datasets are NOT stored in this GitHub repository.

--------------------------------------------------

## Folder Structure (DO NOT CHANGE)

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

--------------------------------------------------

## Dataset Index (Single Source of Truth)

All experiments rely on the following file:

AI_Image_Detection_Data/index/dataset_index.csv

This CSV defines:
- image file paths
- class labels (real vs AI)
- dataset splits (train / val / test)
- dataset balancing strategy

Image files must NOT be moved or duplicated.
Any changes to the dataset must be reflected in this CSV.

--------------------------------------------------

## Rules (IMPORTANT)

- Do NOT upload datasets to GitHub.
- Do NOT move, rename, or duplicate image files.
- Only one shared copy of the dataset exists.
- All training, evaluation, and inference code must read from dataset_index.csv.
- Dataset balancing is handled logically (via CSV), not physically.

--------------------------------------------------

## Using the Dataset in Google Colab

Mount Google Drive at the start of every notebook:

from google.colab import drive
drive.mount('/content/drive')

Set the dataset root path:

DATA_ROOT = "/content/drive/MyDrive/AI_Image_Detection_Data"
Drivelink : https://drive.google.com/drive/folders/1_eOJYn3O3o-pz9xirupHQ0Cm9f8Rgfc9

All dataset paths should be constructed relative to DATA_ROOT.

--------------------------------------------------

## Notes

- MS COCO images are used as-is from the official dataset.
- GenImage images are converted to RGB JPEG format during data collection.
- Image resizing, normalization, and augmentation are applied during training,
  not at the dataset storage level.

--------------------------------------------------

End of file.
