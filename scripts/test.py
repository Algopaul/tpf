import matplotlib.pyplot as plt
import numpy as np

arr = np.load("data/x_data_analytic.npy")
print(np.min(arr))
print(np.max(arr))

for i in range(512):
    fig = plt.figure()
    out = np.histogram2d(arr[:, i, 0], arr[:, i, 1], bins=128)[0]
    plt.imshow(out)
    plt.savefig(f"test_{i:04d}.png")
    plt.close(fig)
