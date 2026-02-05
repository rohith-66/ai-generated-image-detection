# Dataset Setup (Shared Drive + Colab)

## Storage location (Google Drive)
Shared folder name: AI_Image_Detection_Data  
Drive link: https://drive.google.com/drive/folders/1_eOJYn3O3o-pz9xirupHQ0Cm9f8Rgfc9?usp=drive_link

## Expected Drive structure

AI_Image_Detection_Data/
├── coco/
│   ├── train2017/
│   └── val2017/
│
├── genimage/
│   ├── stable_diffusion/
│   ├── stylegan/
│   ├── biggan/
│   └── adm/
│
├── index/
│   └── dataset_index.csv
│
└── checkpoints/

## Rules
- Do NOT upload datasets to GitHub.
- All experiments must read images using index/dataset_index.csv.
- Only one teammate downloads datasets.
- Do not rename folders or move files.

## Colab mount snippet

Use this at the top of every notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

DATA_ROOT = "/content/drive/MyDrive/AI_Image_Detection_Data"
