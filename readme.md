# Project Title

## Overview
This project involves training diffusion models on various datasets to perform generative/denoising task and evaluated on classification tasks. The main focus is on image classification, including synthetic datasets and real-world images from the MNIST,Oxford Pets, CIFAR-10 datasets.

## Synthetic Data and Trained Model
You can access the Trained Models weights and generated synthetic dataset here:
https://drive.google.com/drive/folders/1X8F23DauNZ7J7rhDnHPShvA5guvnlVts?usp=sharing
https://drive.google.com/drive/folders/1wytT0t7CirUZw0_w3qFvhVn4dUgtPjX1?usp=sharing

## Directories

- `synthetic_dataset`: Contains synthetic datasets created by the trained diffusion model and useed for classification task.
- `result`: Stores output results/visualizations from model training and evaluations.
- `models`: Contains saved models after training.
- `MNIST`: training and evaluation directory for the MNIST with the baseline model, and check usability before adding more features to UNET and before training more complex dataset.

## Python Notebooks

- `train_OxfordPet.ipynb`: Jupyter notebook for training models on the Oxford Pets dataset.
- `train_car.ipynb`: Jupyter notebook dedicated to training car image classification models with original and synthetic dataset.
- `Sampling.ipynb`: Notebook for sampling methods(generate image from pure noise, denoise images).
- `OxfordPet_dog_vs_cat_synthetic_data_classification.ipynb`: Notebook for training a classifier on Oxford Pets data focusing on dogs vs. cats using synthetic data.
- `Cifar10_car_vs_truck_synthetic_data_classification.ipynb`: Notebook for the classification between cars and trucks in the CIFAR-10 dataset with synthetic data.

## Python Scripts

- `utils.py`: Utility functions used across different notebooks and scripts, include forward diffusion calculation, sampling function, and visualization function.
- `diffusion_model.py`: Contain diffusion model.

## Model Architecture
![model architecture](https://github.com/EricFeng20001120/Generative-Diffusion-Model/blob/main/result/Model%20Architecture%20Visualization/U-Net.drawio.png)

