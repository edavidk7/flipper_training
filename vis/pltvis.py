import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_grids_xyz(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """
    Plot the x, y, z grids.

    Parameters:
        - x: x-coordinates of the grid.
        - y: y-coordinates of the grid.
        - z: z-coordinates of the grid.
        - title: Title of the plot.
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 5), dpi=200)
    x_im = ax[0].contourf(x, y, x, cmap='gray', levels=100)
    ax[0].invert_yaxis()
    ax[0].set_title("X")
    ax[0].xaxis.tick_top()  # Move X-axis ticks to the top
    ax[0].tick_params(axis='x', labeltop=True, labelbottom=False)  # Show labels on top and hide them on bottom
    y_im = ax[1].contourf(x, y, y, cmap='gray', levels=100)
    ax[1].invert_yaxis()
    ax[1].set_title("Y")
    ax[1].xaxis.tick_top()  # Move X-axis ticks to the top
    ax[1].tick_params(axis='x', labeltop=True, labelbottom=False)  # Show labels on top and hide them on bottom
    z_im = ax[2].contourf(x, y, z, cmap='inferno', levels=100)
    ax[2].invert_yaxis()
    ax[2].set_title("Z")
    ax[2].xaxis.tick_top()  # Move X-axis ticks to the top
    ax[2].tick_params(axis='x', labeltop=True, labelbottom=False)  # Show labels on top and hide them on bottom
    for axis in ax:
        axis.set_aspect('equal')
    plt.colorbar(x_im, ax=ax[0])
    plt.colorbar(y_im, ax=ax[1])
    plt.colorbar(z_im, ax=ax[2])
    fig.tight_layout()
    plt.show()
