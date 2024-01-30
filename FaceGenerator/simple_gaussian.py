import numpy as np
import random
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import os
SEED=1234
np.random.seed(SEED)
random.seed(SEED)

def load_dataset():
    X, y = fetch_lfw_people(return_X_y=True)
    X = X * 255
    X = X.round()
    # Reducing size of the dataset with the 500 most frequent labels
    unique, counts = np.unique(y, return_counts=True)
    labels_to_keep = np.argpartition(counts, -500)[-500:]
    X_ = []
    for label in labels_to_keep:
        indexes = (y == label)
        X_.append(X[indexes])
    X = np.concatenate(X_)
    np.random.shuffle(X)
    return X

def sample_from_normal(X_j_mu, X_j_std):
    return X_j_mu + X_j_std * np.random.randn(X_j_mu.size)

# INITIALIZATION
X = load_dataset()
image_shape = (62, 47)
os.makedirs('results/simple_gaussian', exist_ok=True)


# PLOTTING THE AVERAGE IMAGE
fig, ax = plt.subplots(1, 1)
ax.imshow(X.mean(0).reshape(image_shape), cmap='gray')
plt.savefig('results/simple_gaussian/average.png')



# PLOTTING 5 SAMPLES FROM THE NORMAL DISTRIBUTION
X_j_mu = np.average(X, axis=0)
X_j_std = np.std(X, axis=0)
samples = [sample_from_normal(X_j_mu, X_j_std) for _ in range(5)]

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(samples[i].clip(0, 256).reshape(image_shape), cmap='gray', vmin=0, vmax=255)
ax[2].set_title("Images sampled from learnt distribution", fontsize=16)
plt.savefig('results/simple_gaussian/normal_distribution.png')


