# Interventional State Space Models

![Interventional State Space Models](https://github.com/user-attachments/assets/5117a1aa-7e42-4070-afeb-d24853744f01)


Causality is at the heart of neuroscience. Popular definitions of causality utilize interventions to capture the causal effect of one node on another. It has been shown that observational statistics falls short of determining causal directions in the absence of interventional data. Here, we introduce a new class of state space models (SSM) called Interventional SSM (iSSM) that explicitly model interventional inputs to build causal models of dynamical systems.
 
<!--
See **[our paper]()** for further details:

```
@inproceedings{
}
```
-->

**Note:** This research code remains a work-in-progress to some extent. It could use more documentation and examples. Please use at your own risk and reach out to us (anejatbakhsh@flatironinstitute.org) if you have questions.

## A short and preliminary guide

### Installation Instructions

1. Download and install [**anaconda**](https://docs.anaconda.com/anaconda/install/index.html).
2. Create a **virtual environment** using anaconda and activate it.

```
conda create -n issm
conda activate issm
```

3. Install [**JAX**](https://github.com/google/jax) package.

4. Install other requirements (matplotlib, scipy, sklearn, numpyro, flax).

5. Run the demo notebook under notebooks/demo_motor.ipynb or use the run script via the following commands.

```
python run.py -c configs/DynamicAttractor.yaml -o ../results/
```


Since the code is preliminary, you will be able to use `git pull` to get updates as we release them.

### Fit the model to some dataset.


```python
# Given
# -----
# y: Float[Array, "num_batches num_timesteps emission_dim"] (data).
# u: Float[Array, "num_batches num_timesteps stim_dim"] (interventional input).
# D: int (state dim),

from jax import vmap
import jax.numpy as jnp
import jax.random as jxr

import models
import inference
import visualizations
import simulations
import utils
import params

# N emission_dim, B: num_batches, T: num_timesteps, M: stim_dim
B, T, N = y.shape
_, _, M = u.shape

seed = 1 # Model's random seed

key = jxr.PRNGKey(seed)
k1,k2,k3,key = jxr.split(key,4)

# Initial parameters for InitialCondition
initial_params = params.ParamsNormal(
    mu = jnp.zeros(N),
    scale_tril = dt*jnp.eye(N)
)

# Initial parameters for the linear dynamical system
lds_params = params.ParamsLinearDynamics(
    scale_tril = dt*jnp.eye(N),
    A = jxr.normal(k1, shape=(N,N)),
    B = jnp.eye(N),
    initial = initial_params
)

# Initial likelihood parameter
likelihood_params = params.ParamsConditionalNormal(
    scale_tril = dt*jnp.eye(N)
)

# Create instances of model blocks (Dynamics, Emission, Likelihood, Joint, Recognition)
initial = models.InitialCondition(N, initial_params)

# The control matrix B can be made trainable by setting `train_B` to `True` 
lds = models.LinearDynamics(D=N,M=N,initial=initial,params=lds_params,dt=dt)
emission = models.NeuralNetEmission(N,N,key=k2,H=100)
likelihood = models.NormalConditionalLikelihood(N,params=likelihood_params)
joint = models.fLDS(lds,emission,likelihood)
recognition = inference.AmortizedLSTM(D=N,N=N,M=N,key=k3,T=y.shape[1])

# Fit the joint model and recognition model parameters
k1, key = jxr.split(key,2)

loss = inference.infer(
    k1,joint,recognition,y,u,
    n_iter=500,step_size=1e-2,
    gamma=0
)

# Infer the latents and denoise the data using trained models
k1, k2, key = jxr.split(key,3)

x_smooth = vmap(
    lambda y,u: recognition(recognition.params,k1,y,u)[0],
    in_axes=(0,0),out_axes=0
)(y,u)

mean = vmap(
    lambda x: emission(emission.params,x),
    in_axes=(0),out_axes=0
)(x_smooth)

y_smooth = vmap(
    lambda mu: likelihood.sample(likelihood_params,mu,key=k2),
    in_axes=(0),out_axes=0
)(mean)
```
