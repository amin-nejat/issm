# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""
from sklearn.linear_model import LinearRegression

import numpy as np

import jax.numpy as jnp
import jax
import jax.random as jxr

from scipy import interpolate
from scipy.stats import ortho_group

from jaxtyping import Array, Float, Int
from typing import List


# %%
def random_rotation(key: jxr.PRNGKey, n: int, theta: float):
    rot = jnp.array(
         [[jnp.cos(theta), -jnp.sin(theta)], 
          [jnp.sin(theta), jnp.cos(theta)]]
        )
    out = jnp.eye(n)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(jax.random.uniform(key, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)

# %%
def split_data(
          x: Float[Array, "C M"], 
          y: Float[Array, "K C N"], 
          train_trial_prop: float, 
          train_condition_prop: float, 
          seed: int
        ):
        N,M,_ = y.shape
        
        train_conditions = jax.random.choice(
            jax.random.PRNGKey(seed),
            shape=(int(train_condition_prop*M),),
            a=np.arange(M),
            replace=False
        ).sort()
        
        train_trials = jax.random.choice(
            jax.random.PRNGKey(seed),
            shape=(int(N*train_trial_prop),),
            a=np.arange(N),
            replace=False
        ).sort()

        
        test_conditions = jnp.setdiff1d(np.arange(M),train_conditions).tolist()
        test_trials = jnp.setdiff1d(np.arange(N),train_trials).tolist()
        
        y_test = {
            'x':y[test_trials,:,:][:,train_conditions],
            'x_test':y[:,test_conditions]
        }

        x_train = x[train_conditions,:]
        y_train = y[train_trials,:,:][:,train_conditions]
        x_test = x[test_conditions,:]

        
        return x_train,y_train,x_test,y_test

# %%
def rotation(n: int, theta: float, dims=[0,1]):
    rot = np.array(
         [[jnp.cos(theta), -jnp.sin(theta)], 
          [jnp.sin(theta),  jnp.cos(theta)]]
    )
    out = np.eye(n)
    out[dims][:,dims] = rot
    
    return out

# %%
def stimulation_protocol(
        key: jxr.PRNGKey,
        time_st: float,
        time_en: float,
        dt: float,
        N: int,
        stimulated: Int[Array, "1 N"],
        amplitude: Float[Array, "1 N"],
        stim_d: float,
        rest_d: float = 1.0,
        repetition: int = 1,
        sigma: float = 0
    ):
    """Create random stimulation protocol for nodes of a network given some input statistics

    Returns:
        [type]: [description]
    """

    t_eval = np.arange(time_st,time_en,dt)
    
    if stim_d == 0 or repetition == 0:
        return np.zeros((len(t_eval),N)),stimulated
    
    t_stim = jnp.linspace(
         time_st,time_en,
         len(stimulated)*repetition*int((stim_d+rest_d)/stim_d)
    )
    I_ = np.zeros((len(t_stim),N))

    for r in range(repetition):
        for idx,i in enumerate(stimulated):
            time_idx = r*len(stimulated)+idx
            d1,d2 = 1,int((stim_d+rest_d)/stim_d)

            k1,key = jax.random.split(key,2)
            # sgn = jax.random.choice(k1,np.array([-1.0,1.0]))
            
            sgn = np.random.choice([-1.0,1.0])
            I_[d2*time_idx+d2//2:d2*time_idx+d2//2+d1,i] = sgn*amplitude[i]
    
    inp = interpolate.interp1d(t_stim,I_.T,kind='nearest',bounds_error=False)
    k1,key = jax.random.split(key,2)
    

    I = inp(t_eval)
    I = I + (I != 0)*sigma*np.random.normal(size=I.shape)

    return I.T,stimulated

# %%
def reg(x_true, x_inferred):
    reg = LinearRegression().fit(x_inferred,x_true)
    return reg.coef_,reg.predict(x_inferred)

# %%
def diag_reg(x_true, x_inferred):
    aligned = np.array([
        LinearRegression().fit(
            x_inferred[:,i][:,None],x_true[:,i][:,None]
        ).predict(x_inferred[:,i][:,None])
        for i in range(x_inferred.shape[1])
    ]).squeeze().T
    return aligned

# %%
def lr_integrator_cnn(D: int):
    O = ortho_group.rvs(D)
    eig_vals = np.diag(np.exp(np.linspace(0,-1,D)))
    A = -np.eye(D)+O@eig_vals@O.T
    return A

# %%
def ff_integrator_cnn(D: int):
    O = ortho_group.rvs(D)
    T = np.eye(D,k=1) + .5*np.hstack((np.zeros((D,D-1)),np.ones((D,1))))
    A = -np.eye(D)+O@T@O.T
    return A