# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import models
import jax.numpy as jnp
import jax
from numpyro import optim
import numpyro
import inference
import visualizations

%load_ext autoreload
%autoreload 2

# %% Make a fake dataset
N = 10  # neurons
D = 3 # dynamics dimension
T = 1000 # time points
seed = 1
num_samples = 10

key = jax.random.PRNGKey(seed)

true_initial = models.InitialCondition(D,key=key)
true_lds = models.LDS(D=D,M=N,initial=true_initial,g=2,A=.9*jax.random.orthogonal(key,D),dt=.9)
true_emission = models.NeuralNetEmission(D,N)
true_likelihood = models.PoissonConditionalLikelihood(N)
true_joint = models.JointfLDS(true_lds,true_emission,true_likelihood)

x = true_lds.sample(T,key)
rate = true_emission.f(x,true_emission.params)
keys = jax.random.split(key,num_samples)
data = [true_likelihood.sample(rate,key=keys[i]) for i in range(num_samples)]
y = jnp.stack(data)

u = jnp.zeros((num_samples,T,N))

visualizations.plot_emissions_poisson(x,y[0])
visualizations.plot_emissions_poisson(x,y[1])
visualizations.plot_emissions_poisson(x,rate)

# %%
k1, key = jax.random.split(key,2)

initial = models.InitialCondition(D,key=k1)
lds = models.LDS(D=D,M=N,initial=initial,g=2,A=jax.random.orthogonal(k1,D),dt=.1)
emission = models.NeuralNetEmission(D,N)
likelihood = models.PoissonConditionalLikelihood(N)
joint = models.JointfLDS(lds,emission,likelihood)


k1, key = jax.random.split(key,2)

recognition = inference.AmortizedMeanField(
    D=D,N=N,M=N,key=k1
)

# %%

loss = inference.infer(
    joint, recognition, y,u,n_iter=1000,step_size=1e-3
)


# %% Visualization
import matplotlib.pyplot as plt
plt.plot(loss)


# %%
x_smooth, lp = recognition.f(y[0],u[0],recognition.params,key=key)
rate_smooth = emission.f(x_smooth,emission.params)
y_smooth = likelihood.sample(rate_smooth)

mu = recognition.sigma(recognition.mu_params,jnp.hstack((y[0],u[0])))
sigma = recognition.sigma(recognition.sigma_params,jnp.hstack((y[0],u[0])))


# visualizations.plot_emissions_poisson(x,y[0])
# visualizations.plot_emissions_poisson(x_smooth,y_smooth)
visualizations.plot_emissions_poisson(x,rate)
visualizations.plot_emissions_poisson(x_smooth,rate_smooth)

print(jnp.corrcoef(x_smooth.flatten(),x.flatten()))
print(jnp.corrcoef(rate.flatten(),rate_smooth.flatten()))

# %%
