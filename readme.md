# Generative Diffusion Model

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

## Qualitative Result
![car](https://github.com/EricFeng20001120/Generative-Diffusion-Model/blob/main/result/Qualitative%20Result/car.png)
![dog](https://github.com/EricFeng20001120/Generative-Diffusion-Model/blob/main/result/Qualitative%20Result/dog.png)

## Quantitative Result
### Table 1: CIFAR-10 Car vs Truck Numerical Results

| Dataset Type     | Dataset Size | Train Risk | Test Risk | Test Accuracy |
|------------------|--------------|------------|-----------|---------------|
| Original         | 20000        | 0.1695     | 0.3082    | 87.95%        |
| Original+Synthetic | 21000      | 0.1276     | 0.2989    | 88.55%        |

### Table 2: OxfordPet Cat vs Dog Numerical Results

| Dataset Type       | Dataset Size | Train Risk | Test Risk | Test Accuracy |
|--------------------|--------------|------------|-----------|---------------|
| Original           | 4766         | 0.0899     | 1.1439    | 75.25%        |
| Original+Synthetic | 5376         | 0.1479     | 0.7020    | 79.44%        |

## References

1. Ho, J., Jain, A., and Abbeel, P. (2020). Denoising diffusion probabilistic models.
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L.; and Polosukhin, I. (2017). Attention Is All You Need.
3. Wang, P. Denoising Diffusion Probabilistic Model in Pytorch. Github repository, [GitHub](https://github.com/lucidrains/denoising-diffusion-pytorch).
4. Rogge, N., and Rasul, K. Google Colaboratory. Google Colab, Google Colab Notebook, [Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb).
5. LeCun, Y., Cortes, C., and Burges, C.J.C. (1998). The MNIST Database of Handwritten Digits. New York, USA. [MNIST Database](http://yann.lecun.com/exdb/mnist/).
6. Schuler, J. P. S. (2021). CIFAR-10 64x64 Resized via CAI Super Resolution. Version 1. [Kaggle Discussion](https://www.kaggle.com/discussions/general/46091).
7. Parkhi, O. M., Vedaldi, A., Zisserman, A., Jawahar, C. V. (2012). Cats and Dogs, IEEE Conference on Computer Vision and Pattern Recognition.
8. DeepFindr. (2022). Diffusion Models from Scratch in Pytorch. [YouTube Video](https://www.youtube.com/watch?v=a4Yfz2FxXiY).
9. Arora, A. (2020). U-Net A PyTorch Implementation in 60 Lines of Code. [Aman Arora’s Blog](https://amaarora.github.io/posts/2020-09-13-unet.html).
10. Hedu AI by Batool Haider. (2020). Visual Guide to Transformer Neural Networks – (Episode 2) Multi-Head & Self-Attention. [YouTube Video](https://www.youtube.com/watch?v=mMa2PmYJlCo).
11. Outlier. (2022). Diffusion Models | Pytorch Implementation. [YouTube Video](https://www.youtube.com/watch?v=TBCRlnwJtZU).

