import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows

# Frame size
X = Y = 64
# Stride
stride = 32

img = plt.imread("earth_clean.png")

# Save memory space and computing power, precision isn't critical
# (float16 precision in [0, 1] is about .0005, values steps are .0039)
img = img.astype('float16')

# Take frames from the original image, shape it as an (N, X, Y) matrix with N frames of X by Y values
frames = view_as_windows(img, (Y, X), stride)
frames = frames.reshape((-1, Y, X))
img = None

size_before = len(frames)

# Remove all images in which the lowest and highest values are no more than 16 brightness values apart (out of 256 values)
diff = np.abs(frames.min(axis=(1, 2)) - frames.max(axis=(1, 2)))
frames = frames[np.where(diff > (16 / 256))]

# Normalize each frame
mins = frames.min(axis=(1, 2), keepdims=True)
maxs = frames.max(axis=(1, 2), keepdims=True)
frames = (frames - mins) / (maxs - mins)

# Remove the frames with a small standard deviation AFTER normalization, these are of low interest.
deviation = np.std(frames, axis=(1, 2))
frames = frames[np.where(deviation >= 0.1)]

print(f'Kept {len(frames)} out of {size_before} tiles ({(len(frames) / size_before):.2%}).')

# Add rotated and flipped version of the tiles
flipped_ud = np.flip(frames, 1)
flipped_lr = np.flip(frames, 2)
flipped_all = np.flip(frames, (1, 2))  # This is equivalent to rotate 180 degrees

rt_90 = np.rot90(frames, 1, (1, 2))
rt_270 = np.rot90(frames, 3, (1, 2))
rt_90_flip = np.flip(rt_90, 2)
rt_270_flip = np.flip(rt_270, 2)

all_frames = np.concatenate((frames, flipped_ud, flipped_lr, flipped_all, rt_90, rt_90_flip, rt_270, rt_270_flip))

n = len(frames)
print(f"Total frames: {n}.")

for i in range(n):
    print("\b"*6 +f'{(i / n) * 100 : >5.1f}%', end="")
    plt.imsave(f'data/earth/earth_clean_64-32/{i}.png', frames[i], cmap="gray")

print("Done.")
