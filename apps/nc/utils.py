import numpy as np
import os
import torch
import cv2

import matplotlib.pyplot as plt

from torchvision import transforms

# Function to load all images from a directory and convert to PyTorch tensors
def load_images_from_directory(directory, num=10, size=(28,28)):
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    images = []
    for i, filename in enumerate(os.listdir(directory)):
        if i >= num:
            return torch.stack(images)
        img_path = os.path.join(directory, filename)
        image = cv2.resize(cv2.imread(img_path,0), size)
        image = transform(image).squeeze() # ToTensor() adds too much
        images.append(image)
    return torch.stack(images)

def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

def plot_images(imgs, labels=None, row_headers=None, col_headers=None, colmns=None, title=None):
    """
    imgs = [img1, img2]
    labels = ['label1', 'label2']
    colmns = 2 (so will be a 1x2 size display)
    """
    num = len(imgs)
    # Calculate the given number of subplots, or use colmns count to get a specific output
    if colmns is None:
        ay = np.ceil(np.sqrt(num)).astype(int) # this way it will prefer rows rather than columns
        ax = np.rint(np.sqrt(num)).astype(int)
    else:
        ax = np.ceil(num / colmns).astype(int)
        ay = colmns
        
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    for i in range(1, num+1):
        sub = fig.add_subplot(ax,ay,i)
        if labels is not None:
            sub.set_title(f'{labels[i-1]}')
            
        sub.axis('off')
        sub.imshow(imgs[i-1])
        
    add_headers(fig, row_headers=row_headers, col_headers=col_headers, rotate_row_headers=False)

def save_plot_imgs(image_list, labels=None, output_path='folder/', output_name='image.png', grid_size=None):
    """
    Plots all the images together in a grid layout and saves the plot as a single image.

    Parameters:
        image_list (list of numpy arrays): List of images as numpy arrays.
        labels (list of str): List of labels for each image (optional). Default is None.
        output_path (str): Path to save the output image. Default is 'output_image.png'.
        grid_size (tuple of int): Size of the grid (rows, columns) to arrange the images. If None, it is calculated based on the number of images. Default is None.
        square (bool): Whether to make the images square by padding if needed. Default is True.
    """

    # Calculate the grid size if not provided
    num_images = len(image_list)
    if grid_size is None:
        num_rows = np.ceil(np.sqrt(num_images)).astype(int)
        num_cols = np.ceil(num_images / num_rows).astype(int)
    else:
        num_rows, num_cols = grid_size

    # Create the figure and axis
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 30))  # Adjust the figsize if needed
    
    output_path = os.path.join(output_path, output_name)
    fig.suptitle(output_name, fontsize=16)

    # Iterate through each image
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_images:
                image = image_list[idx]

                # Show the image
                axs[i, j].imshow(image)
                axs[i, j].axis('off')

                # Add label if provided
                if labels is not None and idx < len(labels):
                    axs[i, j].set_title(labels[idx])

    # Remove any remaining empty subplots
    for idx in range(num_images, num_rows * num_cols):
        fig.delaxes(axs.flatten()[idx])

    # Save the plot to a file
    plt.tight_layout()  # Adjust spacing and layout
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)  # Adjust dpi if needed
    plt.close()
