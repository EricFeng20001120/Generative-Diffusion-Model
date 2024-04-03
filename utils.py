import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np


def linear_schedule(timesteps, start=0.0001, end=0.02):
    """
    Creates a linear schedule for beta values over a specified number of timesteps.

    Parameters:
    - timesteps (int): Number of steps in the diffusion process.
    - start (float): Starting beta value.
    - end (float): Ending beta value.

    Returns:
    - torch.Tensor: 1D tensor of beta values of shape [timesteps].
    """
    return torch.linspace(start, end, timesteps)

def sigmoid_beta_schedule(timesteps):
    '''
    Parameters:
    - timesteps (int): The total number of timesteps for which to generate the beta values. This parameter determines the length of the beta schedule.
    Returns:
    - torch.Tensor: A 1-dimensional tensor of size [timesteps] containing the beta values for each timestep. These values are calculated using a sigmoid function, scaled to lie within the range specified by beta_start and beta_end.
    '''
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def add_noise(x, t,device = 'cuda'):
    """
    Adds noise to an image tensor at a given timestep.

    Parameters:
    - x (torch.Tensor): The original image tensor of shape [B, C, H, W].
    - t (torch.Tensor): The current timestep, a long tensor of shape [B].
    - device (str)

    Returns:
    - torch.Tensor: The noised image tensor of the same shape as x.
    """
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t])[:,None,None,None]
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1-alphas_cumprod[t])[:,None,None,None]
    noise = torch.randn_like(x)
    return sqrt_alpha_cumprod.to(device) * x.to(device) + sqrt_one_minus_alpha_cumprod.to(device) * noise.to(device), noise.to(device)

def denoise_step(x, t, model):
    """
    Performs a denoising step by estimating and subtracting the noise from the noised image.

    Parameters:
    - x (torch.Tensor): The noised image tensor of shape [B, C, H, W].
    - t (torch.Tensor): The current timestep, a long tensor of shape [B].
    - model (callable): The denoising model.

    Returns:
    - torch.Tensor: The denoised image tensor.
    """
    beta = betas[t][:,None,None,None]
    alpha = alphas[t][:,None,None,None]
    alpha_cumprod = alphas_cumprod[t][:,None,None,None]
    # Model prediction for noise
    noise_pred = model(x, t)
    # Calculate model mean
    model_mean = 1/torch.sqrt(alpha) * (x - noise_pred * ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))))

    if t.item() == 0:
        return model_mean
    else:
        return model_mean + torch.sqrt(beta) * torch.randn_like(x)



def show_tensor_image(image):
    '''
    Converts a PyTorch tensor to an image
    
    Parameters:
    - image (torch.Tensor): The image tensor to display. Can be a single image tensor of shape [C, H, W] or a batch of images [B, C, H, W]. If a batch is provided, only the first image is displayed.
    Returns:
    - None: 
    '''
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

@torch.no_grad()
def sample_plot_image(model, img_size):
    '''
    Generates and plots a sequence of images to visualize the denoising process over a predefined number of steps. 

    Parameters:
    - model (callable): The denoising model used to iteratively denoise the image.
    - img_size (int): The size of the square image to generate and denoise, specified as the height and width in pixels.
    Returns:
    - None. This function plots a sequence of images to visualize the denoising process.
    '''
    img = torch.randn((1, 3, img_size, img_size), device=device)  # [1, 3, img_size, img_size]
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = T // num_images

    for i in reversed(range(T)):
        t = torch.full((1,), i, device=device, dtype=torch.long)  # [1]
        img = denoise_step(img, t, model)  # denoise_step returns [1, 3, img_size, img_size]
        img = torch.clamp(img, -1.0, 1.0)  # Clamp to maintain [-1, 1] range

        if i % stepsize == 0:
            plt.subplot(1, num_images, num_images - i // stepsize)
            show_tensor_image(img.detach().cpu())  # Convert [1, 3, img_size, img_size] tensor to image

    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000  # Total number of timesteps
# Precompute terms for the diffusion process
#betas = linear_schedule(T).to(device)  # [T]
betas = sigmoid_beta_schedule(T).to(device)  # [T]
alphas = 1.0 - betas  # [T]
alphas_cumprod = torch.cumprod(alphas, dim=0)  # [T]