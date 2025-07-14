# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:15:48 2025

@author: XiaoFan
"""
import numpy as np
import matplotlib.pyplot as plt

# Define ranges for the key parameters in stress fields
nu_Dol_min, nu_Dol_max = 0.20, 0.35       # dolomite Poisson′s ratio (ν_Dol)
nu_Ss_min, nu_Ss_max = 0.20, 0.30         # sandstone Poisson′s ratio (ν_Ss)
nu_Lst_min, nu_Lst_max = 0.20, 0.35       # limestone Poisson′s ratio (ν_Lst)
nu_Fz_min, nu_Fz_max = 0.30, 0.35         # fault zone Poisson′s ratio (ν_Fz)

# Number of experiments
n_experiments = 625

# Function to generate Latin Hypercube Samples
def latin_hypercube_sampling(n_samples, n_dimensions):
    # Create an array to hold the samples
    samples = np.zeros((n_samples, n_dimensions))
    
    # Generate LHS for each dimension
    for i in range(n_dimensions):
        # Create stratified random samples
        stratified_samples = np.random.rand(n_samples) + np.arange(n_samples)
        np.random.shuffle(stratified_samples)
        samples[:, i] = stratified_samples / n_samples
    return samples

# Generate LHS samples
lhs_samples = latin_hypercube_sampling(n_experiments, 4)

# Scale LHS samples to the defined ranges
nu_Dol_samples = nu_Dol_min + (nu_Dol_max - nu_Dol_min) * lhs_samples[:, 0]
nu_Ss_samples = nu_Ss_min + (nu_Ss_max - nu_Ss_min) * lhs_samples[:, 1]
nu_Lst_samples = nu_Lst_min + (nu_Lst_max - nu_Lst_min) * lhs_samples[:, 2]
nu_Fz_samples = nu_Fz_min + (nu_Fz_max - nu_Fz_min) * lhs_samples[:, 3]

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([6,6,6])
ax.view_init(elev=45, azim= 45) 
# Create scatter plot
scatter = ax.scatter(nu_Dol_samples, nu_Ss_samples, nu_Lst_samples, 
                     c=nu_Fz_samples, s=100, 
                     cmap='viridis', alpha=0.85, edgecolors='w')

# Add color bar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('ν_Fz', rotation=270, labelpad=15)
#Set the font size of the color bar scale
cbar.ax.tick_params(labelsize=15)  # font size =14

# Set labels and title
ax.set_xlabel('ν_Dol')
ax.set_ylabel('ν_Ss')
ax.set_zlabel('ν_Lst')

# Set the font size of the axes
ax.tick_params(axis='x', labelsize=15)  
ax.tick_params(axis='y', labelsize=15)  
ax.tick_params(axis='z', labelsize=15)  

# Save the figure as SVG
plt.savefig("LHS.pdf", format='pdf')

# Show the plot
plt.show()
