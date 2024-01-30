# Python script generated from Jupyter Notebook

import numpy as np
import random
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from sklearn.mixture import GaussianMixture


SEED=1234
np.random.seed(SEED)
random.seed(SEED)








def samples_with_temperature(gmm, temperature=1, n_samples=5):    
    assert 0 <= temperature <= 1, "The temperature parameter must be between 0 and 1."
    
    original_cov = gmm.covariances_.copy() # copying the original covariances
    gmm.covariances_ = gmm.covariances_ * temperature # Adjusting temperature
    samples, _ = gmm.sample(n_samples) # Sample
    gmm.covariances_ = original_cov # reset the temperature
    
    return samples




gmm25_ = GaussianMixture(n_components=25, covariance_type='diag').fit(X_)
gmm100_ = GaussianMixture(n_components=100, covariance_type='diag').fit(X_)


samples_ = [sample_from_normal_(1) for _ in range(5)]

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 1 component", fontsize=16)
plt.show()

samples_, _ = gmm25_.sample(5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 25 components", fontsize=16)
plt.show()

samples_, _ = gmm100_.sample(5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 100 components", fontsize=16)
plt.show()








# GMM low temperature
samples_ = [sample_from_normal_(0.25) for _ in range(5)]

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 1 component (T=0.25)", fontsize=16)
plt.show()

samples_ = samples_with_temperature(gmm25_, 0.25, 5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 25 components (approx. T=0.25)", fontsize=16)
plt.show()

samples_ = samples_with_temperature(gmm100_, 0.25, 5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 100 components (approx. T=0.25)", fontsize=16)
plt.show()






# GMM high temeperature 

samples_ = [sample_from_normal_(0) for _ in range(5)]

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 1 component (T=0)", fontsize=16)
plt.show()

samples_ = samples_with_temperature(gmm25_, 0, 5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 25 components (approx. T=0)", fontsize=16)
plt.show()

samples_ = samples_with_temperature(gmm100_, 0, 5)

fig, ax = plt.subplots(1, 5, figsize=(10, 6))
for i in range(5):
    plotting_axis = ax[i]
    plotting_axis.imshow(pca.inverse_transform(samples_[i]).reshape(image_shape), cmap='gray')
ax[2].set_title("Images sampled from trained GMM with 100 components (approx. T=0)", fontsize=16)
plt.show()