import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange


def get_distances(points, dictionary):
    points = rearrange(points, "n d -> n 1 d")
    dictionary = rearrange(dictionary, "m d -> 1 m d")

    distances = np.linalg.norm(points - dictionary, ord=2, axis=-1)
    return distances


def softmin_kernel(distances, temperature: float = 1.0):
    def softmin(x, axis):
        x = -x
        x = x - x.max(axis=axis, keepdims=True)
        x_exp = np.exp(x)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    return np.sum(softmin(distances / temperature, axis=1) * distances**2, axis=1)


def main():
    rng = np.random.default_rng(1)
    dictionary_size = 16
    grid_resolution = 128

    dictionary = rng.normal(size=(dictionary_size, 2))
    grid_x = np.linspace(-5, 5, grid_resolution)
    grid_y = np.linspace(-5, 5, grid_resolution)
    grid = np.stack(np.meshgrid(grid_x, grid_y), axis=2)

    distances = get_distances(grid.reshape(-1, 2), dictionary)
    values = softmin_kernel(distances, temperature=1.0)

    plt.figure()
    plt.scatter(*dictionary.T)
    plt.imshow(
        values.reshape(grid_resolution, grid_resolution),
        cmap="magma",
        extent=(grid_x[0], grid_x[-1], grid_y[-1], grid_y[0]),
        norm="log",
    )
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
