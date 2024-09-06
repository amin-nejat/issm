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
import utils

import numpy as np

# %%
class iPfLDSLoader():
    def __init__(self,params):
        # params: B (batch), T (time), N (neurons), D (latent dim) 
        # seed, stim_amplitude

        key = jax.random.PRNGKey(params['seed'])

        u = []

        for i in range(params['B']):
            k1, key = jax.random.split(key,2)
            u_,_ = utils.stimulation_protocol(
                k1,
                time_st=0,
                time_en=params['T'],
                dt=params['dt'],
                N=params['N'],
                stimulated=jnp.arange(params['N_stim']),
                amplitude=params['stim_amplitude']*jnp.ones(params['N']),
                stim_d=1.,repetition=params['repetition'],
                sigma=params['sigma']
            )
            
            u.append(u_)

        
        k1, k2, key = jax.random.split(key,3)

        true_initial = models.InitialCondition(params['D'])
        if params['B_sparsity'] == 'identity': 
            if params['D'] == params['N']:
                B,B_sparsity=jnp.eye(params['D']),None
            else:
                B = jnp.zeros((params['D'],params['N']))
                for d in range(params['D']):
                    for n in range(params['N']):
                        if d-n % params['D'] == 0:
                            B = B.at[d,n].set(1) 
                B_sparsity = None
        else:
            B,B_sparsity=None,params['B_sparsity']

        true_lds = models.iConstrainedLDS(
            key=k2,B_sparsity=B_sparsity,B=B,
            D=params['D'],M=params['N'],initial=true_initial,g=2,
            A=utils.random_rotation(k1, params['D'], .5),
            dt=params['dt'],
            scale_tril=jnp.eye(params['D'])
        )
        k1, key = jax.random.split(key,2)

        true_emission = eval('models.'+params['emission'])(params['D'],params['N'],key=k1)
        true_likelihood = eval('models.'+params['likelihood'])(params['N'])
        true_joint = models.JointfLDS(true_lds,true_emission,true_likelihood)

        x,y,r = [],[],[]

        keys = jax.random.split(key,params['B'])
        for i in range(params['B']):
            x_ = true_lds.sample(params['T'],keys[i],u=u[i])
            r_ = true_emission.f(x_,true_emission.params)
            y_ = true_likelihood.sample(r_,key=keys[i])
            x.append(x_); r.append(r_); y.append(y_)

        
        k1, key = jax.random.split(key,2)

        self.x = jnp.stack(x)
        self.r = jnp.stack(r)
        self.y = jnp.stack(y)
        self.u = jnp.stack(u)
        self.params = params
        self.lds = true_lds
        self.emission = true_emission
        self.likelihood = true_likelihood

        #  Unseen interventions
        if params['N_stim_test'] == 0: 
            stimulated = jnp.arange(params['N_stim'])
        else: 
            stimulated = jnp.arange(params['N_stim'],params['N_stim']+params['N_stim_test'])
        
        k1, key = jax.random.split(key,2)

        u_test,_ = utils.stimulation_protocol(
            k1,
            time_st=0,
            time_en=params['T_test'],
            dt=params['dt'],
            N=params['N'],
            stimulated=stimulated,
            amplitude=params['stim_amplitude_test']*jnp.ones(params['N']),
            stim_d=1.,repetition=params['repetition'],
            sigma=params['sigma']
        )

        k1, key = jax.random.split(key,2)

        x_test = true_lds.sample(params['T_test'],key,u=u_test)
        r_test = true_emission.f(x_test,true_emission.params)
        y_test = true_likelihood.sample(r_test,key=key)

        self.x_test = x_test
        self.r_test = r_test
        self.y_test = y_test
        self.u_test = u_test
        

    def load_data(self):
        return self.y, self.u

    def load_test_data(self):
        return self.y_test, self.u_test


# %%
class RateModel:
    '''Abstract class for a rate network
    '''
    def __init__(self,pm,discrete=True,B=None):
        assert 'D' in pm.keys()

        self.pm = pm
        self.discrete = discrete
        self.B = np.eye(pm['D']) if B is None else np.array(B)

        if 't_eval' not in self.pm.keys(): self.pm['t_eval'] = None
    
    
    def run(self,T,x0,u=None,dt=.1,b=None):
        # Assume u is interventional input
        t = np.arange(0,T,dt)
        x = np.zeros((len(t),x0.shape[0],x0.shape[1]))
        
        if u is None: du = np.zeros(x0.shape)

        x[0,:,:] = x0
        for i in range(1,len(t)):
            dx = dt*self.step(t[i],x[i-1])
            if b is not None: dx += dt*b(t[i])
            if u is not None: du = np.einsum('mn,km->kn',self.B,u[:,i])
            x[i] = (du==0).astype(float)*(x[i-1]+dx)+du

        return t,x
    
    def obs(self,x,t=None,u=None):
        raise NotImplementedError
    
    
    def step(self,t,x,u=None):
        raise NotImplementedError


# %%
class Linear(RateModel):
    '''Linear Dynamical System
    '''
    def __init__(self,pm,discrete=True,B=None):
        keys = pm.keys()
        assert 'sigma' in keys and 'A' in keys
        super(Linear, self).__init__(pm,discrete=discrete,B=B)
        self.eye = np.eye(pm['D'])

    def step(self,t,x,u=None):

        dx = np.einsum('mn,bn->bm',-self.eye+self.pm['A'],x)
        dw = self.pm['sigma']*np.random.randn(*x.shape)
        dxdt = dx+dw
        
        return dxdt
    

# %%
class RotationalDynamics(RateModel):
    def __init__(self,pm,discrete=True,B=None):
        keys = pm.keys()

        assert 'sigma' in keys and 'a' in keys
        super(RotationalDynamics, self).__init__(
            pm,discrete=discrete,B=B
        )

    def step(self,t,x,u=None):
        dw = self.pm['sigma']*np.random.randn(*x.shape)
        dx = np.stack([
            0*x[:,0],
            self.pm['a']*x[:,0]
            ]).T

        dxdt = dx+dw
        return dxdt

    def obs(self,x,t=None,u=None):
        y = np.stack((
            x[...,0] * np.cos(x[...,1]),
            x[...,0] * np.sin(x[...,1])
            ),axis=-1)
        return y

    def inv_obs(self,y,t=None,u=None):
        x = np.stack((
            np.sqrt(y[...,0]**2 + y[...,1]**2),
            np.arctan2(y[...,1], y[...,0])
        ),axis=-1)
        return x

# %%
class DynamicAttractor(RateModel):
    def __init__(self,pm,discrete=True,B=None):
        keys = pm.keys()
        assert 'sigma' in keys and 'a1' in keys and 'a2' in keys
        super(DynamicAttractor, self).__init__(
            pm,discrete=discrete,B=B
        )
    def step(self,t,x,u=None):
        dw = self.pm['sigma']*np.random.randn(*x.shape)
        dx = np.stack([
            self.pm['a1']*x[:,0],
            self.pm['a2']*(1-x[:,0])
            ]).T

        dxdt = dx+dw
        return dxdt

    def obs(self,x,t=None,u=None):
        y = np.stack((
            (1-x[...,0]) * np.cos(x[...,1]),
            (1-x[...,0]) * np.sin(x[...,1])
            ),axis=-1)
        return y

    def inv_obs(self,y,t=None,u=None):
        x = np.stack((
            1-np.sqrt(y[...,0]**2 + y[...,1]**2),
            np.arctan2(y[...,1], y[...,0])
            ),axis=-1)
        return x