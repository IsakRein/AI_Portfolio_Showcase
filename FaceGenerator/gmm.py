import numpy as np
import random
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

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
os.makedirs('results/GMM', exist_ok=True)
pca = PCA(n_components, random_state=SEED)
X_ = pca.fit_transform(X)
X_j_mu_ = np.average(X_, axis=0)
X_j_std_ = np.std(X_, axis=0)
gmm25_ = GaussianMixture(n_components=25, covariance_type='diag').fit(X_)
gmm100_ = GaussianMixture(n_components=100, covariance_type='diag').fit(X_)


def sample_from_normal_(X_j_mu_, X_j_std_, temperature=1):
    return X_j_mu_ + X_j_std_ * np.random.randn(X_j_mu_.size) * temperature


def samples_with_temperature(gmm, temperature=1, n_samples=5):
    """ Sklearn does not has a method/argument to do it so here is a hacky way of doing it"""
    
    assert 0 <= temperature <= 1, "The temperature parameter must be between 0 and 1."
    
    
    original_cov = gmm.covariances_.copy() # copying the original covariances
    gmm.covariances_ = gmm.covariances_ * temperature # Adjusting temperature
    samples, _ = gmm.sample(n_samples) # Sample
    gmm.covariances_ = original_cov # reset the temperature
    
    return samples


samples_ = [sample_from_normal_(X_j_mu_, X_j_std_, 0.25) for _ in range(5)]

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 1 component (T=0.25)", fontsize=16)
plt.savefig('results/GMM/GMM_1_component.png')


samples_ = samples_with_temperature(gmm25_, 0.25, 5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 25 components (approx. T=0.25)", fontsize=16)
plt.savefig('results/GMM/GMM_25_components.png')

samples_ = samples_with_temperature(gmm100_, 0.25, 5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 100 components (approx. T=0.25)", fontsize=16)
plt.savefig('results/GMM/GMM_100_components.png')



samples_ = [sample_from_normal_(X_j_mu_, X_j_std_,0) for _ in range(5)]

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 1 component (T=0)", fontsize=16)
plt.savefig('results/GMM/GMM_1_component_T0.png')

samples_ = samples_with_temperature(gmm25_, 0, 5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 25 components (approx. T=0)", fontsize=16)
plt.savefig('results/GMM/GMM_25_component_T0.png')

samples_ = samples_with_temperature(gmm100_, 0, 5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 100 components (approx. T=0)", fontsize=16)
plt.savefig('results/GMM/GMM_100_component_T0.png')
