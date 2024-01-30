import numpy as np
import random
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

n_components = 250

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

def sample_from_normal_(X_j_mu_, X_j_std_, temperature=1):
    return X_j_mu_ + X_j_std_ * np.random.randn(X_j_mu_.size) * temperature


# INITIALIZATION
X = load_dataset()
image_shape = (62, 47)
os.makedirs('results/improved_gaussian', exist_ok=True)
pca = PCA(n_components, random_state=SEED)
X_ = pca.fit_transform(X)
X_j_mu_ = np.average(X_, axis=0)
X_j_std_ = np.std(X_, axis=0)

# USING EIGENVECTORS
eigenfaces = pca.components_[:n_components]
fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
for i in range(16):
    axes[i%4][i//4].imshow(eigenfaces[i].reshape(image_shape), cmap="gray")
plt.savefig('results/improved_gaussian/eigenfaces.png')


# PCA WITH TEMPERATURE
samples_ = pca.inverse_transform([sample_from_normal_(X_j_mu_, X_j_std_, 1) for _ in range(5)])

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(samples_[i].reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from learnt distribution (T=1)", fontsize=16)
plt.savefig('results/improved_gaussian/pca_with_temp1.png')

samples_ = pca.inverse_transform([sample_from_normal_(X_j_mu_, X_j_std_,0.25) for _ in range(5)])

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(samples_[i].reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from learnt distribution (T=0.25)", fontsize=16)
plt.savefig('results/improved_gaussian/pca_with_temp025.png')


samples_ = pca.inverse_transform([sample_from_normal_(X_j_mu_, X_j_std_,0) for _ in range(5)])

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(samples_[i].reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from learnt distribution (T=0)", fontsize=16)
plt.savefig('results/improved_gaussian/pca_with_temp0.png')
