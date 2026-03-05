import matplotlib
import numpy as np


def to_pixel(x, y, xlim, ylim, resolution):
    xmin, xmax = xlim
    ymin, ymax = ylim
    px = ((x - xmin) / (xmax - xmin) * (resolution - 1)).astype(int)
    py = ((y - ymin) / (ymax - ymin) * (resolution - 1)).astype(int)
    return px, py


def trace_video(
    data,
    *,
    resolution=512,
    xlim=(-1, 1),
    ylim=(-1, 1),
    trail_decay=0.92,  # closer to 1 = longer trails
    dot_intensity=1.0,  # brightness of current position
):
    """
    data: (n_time, n_particles, 2)
    """

    n_time, _, _ = data.shape

    # Frame buffer (float for accumulation)
    frame = np.zeros((resolution, resolution, 3), dtype=np.float32)

    out_data = []

    for t in range(n_time):
        # Fade previous frame
        frame *= trail_decay

        x = data[t, :, 0]
        y = data[t, :, 1]

        px, py = to_pixel(x, y, xlim, ylim, resolution)

        mask = (px >= 0) & (px < resolution) & (py >= 0) & (py < resolution)
        px = px[mask]
        py = py[mask]

        frame[py, px, 0] += 0.1 * dot_intensity
        frame[py, px, 1] += 0.8 * dot_intensity
        frame[py, px, 2] += 1.0 * dot_intensity

        img = np.clip(frame, 0, 1)
        rgb = (255 * img).astype(np.uint8)
        out_data.append(rgb)

    return out_data


def angle_color_coded(
    data,
    source_data,
    *,
    resolution=512,
    xlim=(-1, 1),
    ylim=(-1, 1),
):
    n_time, _, _ = data.shape
    cmap = matplotlib.colormaps["hsv"]  # cyclic colormap
    angles = np.arctan2(source_data[:, 1], source_data[:, 0])
    angles = (np.pi + angles) / (2 * np.pi)
    colors = cmap(angles)[:, :3]
    out_data = []

    for t in range(n_time):
        frame = np.zeros((resolution, resolution, 3), dtype=np.float32)
        x = data[t, :, 0]
        y = data[t, :, 1]
        px, py = to_pixel(x, y, xlim, ylim, resolution)
        mask = (px >= 0) & (px < resolution) & (py >= 0) & (py < resolution)
        px = px[mask]
        py = py[mask]
        frame[py, px] = colors[mask]
        img = np.clip(frame, 0, 1)
        rgb = (255 * img).astype(np.uint8)
        out_data.append(rgb)

    return out_data
