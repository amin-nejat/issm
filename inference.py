# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import jax.random as jxr
import jax.numpy as jnp
import numpyro.distributions as dist

from jax.example_libraries import optimizers
from jax import jit, value_and_grad, vmap

from tqdm.auto import trange
import flax.linen as nn
import jax

from models import fLDS
from params import ParamsVariationalLSTM
from jaxtyping import Array, Float

# %%
class VariationalLSTM:
    def __init__(self,shape,interventional: bool):
        self.shape = shape
        self.f_mu = [nn.softmax]
        self.f_scale = [nn.softmax]
        self.interventional = interventional
    
    def init(self, key: jxr.PRNGKey, fs, shape):
        vars = []

        for i in range(len(fs)):
            if not hasattr(fs[i],'init'):
                continue
            k1, key = jxr.split(key,2)
            vars.append(fs[i].init(k1, jnp.ones(shape[i])))

        return tuple(vars)

    def _apply(self, params: ParamsVariationalLSTM, fs, y):
        for i in range(len(fs)):
            if hasattr(fs[i],'apply'):
                y = fs[i].apply(params[i],y)
            else:
                y = fs[i](y) 
        return y

    
    def __call__(
            self, 
            params: ParamsVariationalLSTM, 
            key: jxr.PRNGKey, 
            y: Float[Array, "num_timesteps emission_dim"],
            u: Float[Array, "num_timesteps stim_dim"]
        ):
        data = jnp.hstack((y,u))[None]

        mu = self._apply(params.theta_mu,self.f_mu,data).reshape(self.shape)
        sg = self._apply(params.theta_scale,self.f_scale,data).reshape(self.shape)

        Bu = jnp.einsum('tn,dn->td',u,params.B)
        if self.interventional:
            mu = (Bu==0).astype(float)*(mu)+Bu
        else:
            mu = mu+Bu

        sample = dist.Normal(mu,sg).sample(key)
        lp = dist.Normal(mu,sg).log_prob(sample)

        return sample[0],lp
    
    def posterior_mean(
            self, 
            params: ParamsVariationalLSTM, 
            y: Float[Array, "num_timesteps emission_dim"],
            u: Float[Array, "num_timesteps stim_dim"]
        ):
        data = jnp.hstack((y,u))[None]
        mu = self._apply(params.theta_mu,self.f_mu,data).reshape(self.shape)

        Bu = jnp.einsum('tn,dn->td',u,params.B)
        if self.interventional:
            mu = (Bu==0).astype(float)*(mu)+Bu
        else:
            mu = mu+Bu
        
        return mu[0]
    
    def posterior_scale(
            self, 
            params: ParamsVariationalLSTM,             
            y: Float[Array, "num_timesteps emission_dim"],
            u: Float[Array, "num_timesteps stim_dim"]
        ):
        data = jnp.hstack((y,u))[None]
        scale = self._apply(params.theta_scale,self.f_scale,data).reshape(self.shape)
        return scale[0]

# %%
class AmortizedLSTM(VariationalLSTM):
    """Differentiable representation of RNN for inference"""
    
    def __init__(self, D: int, N: int, M: int, key: jxr.PRNGKey, interventional: bool, H: int = 10, T: int = 10):
        '''initialize an instance
        '''        
        super(AmortizedLSTM, self).__init__(shape=(1,T,D),interventional=interventional)
        
        self.f_mu = [
            nn.RNN(nn.LSTMCell(H)),
            nn.Dense(D)
        ]

        self.f_scale = [
            nn.RNN(nn.LSTMCell(H)),
            nn.Dense(D),
            lambda x: nn.softplus(x)
        ]

        k1, key = jxr.split(key,2)
        theta_mu = self.init(k1,self.f_mu,[(1,T,M+N),(1,T,H)])

        k1, key = jxr.split(k1,2)
        theta_scale = self.init(key,self.f_scale,[(1,T,M+N),(1,T,H)])

        self.params = ParamsVariationalLSTM(
            theta_mu = theta_mu,
            theta_scale = theta_scale,
            B=jnp.zeros((D,M))
        )


# %%
def infer(
        key: jxr.PRNGKey,
        joint: fLDS,
        recognition: AmortizedLSTM,
        y: Float[Array, "num_batches num_timesteps emission_dim"],
        u: Float[Array, "num_batches num_timesteps stim_dim"],
        n_iter: int = 100,
        step_size: float = .1,
        b1: float = .8,
        b2: float = .9,
        gamma: float = 1.
    ):
    opt_init, opt_update, get_params = optimizers.adam(step_size,b1=b1,b2=b2)
    opt_state = opt_init((recognition.params, joint.params))

    # TODO: Fix batch size and log prior
    batch_size = len(y)
    
    def batch_elbo(params,key,y,u):
        def elbo(params,key,y,u):
            (recognition_params, joint_params) = params

            k1, key = jxr.split(key,2)
            # variational sample and log prob

            if joint_params.dynamics.B is None:
                recognition_params = recognition_params._replace(B=joint.dynamics.B)
            else:
                recognition_params = recognition_params._replace(B=joint_params.dynamics.B)
            x,lp = recognition(recognition_params,k1,y,u)
            log_prior = lp.sum()

            # Compute joint log probability over many samples of
            log_joint = joint.log_prob(joint_params,y,u,x)

            # ELBO is the expected joint log probability plus entropy
            return  -log_joint+gamma*log_prior, log_joint, log_prior

        fun = vmap(
            lambda y,u: elbo(params,key,y,u),
            in_axes=(0,0),
            out_axes=0
        )
        val = fun(y,u)
        elbo_loss, log_joint, log_prior = jnp.array(val).mean(1)
        
        return elbo_loss, (log_joint, log_prior)
    
    @jit
    def update(params,key,y,u,i,opt_state):
        ''' Perform a forward pass, calculate the MSE & perform a SGD step. '''

        (loss, (log_joint, log_prior)), grads = value_and_grad(
                batch_elbo,has_aux=True,allow_int=True
            )(params,key,y,u)
        
        opt_state = opt_update(i, grads, opt_state)
        params = get_params(opt_state)
        
        return loss, log_joint, log_prior, opt_state, params


    params = get_params(opt_state)

    pbar = trange(n_iter)
    pbar.set_description('jit compiling ...')

    losses = []

    for i in pbar:
        key, k1 = jxr.split(key,2)
        loss,log_joint,log_prior,opt_state,params = \
            update(params,k1,y,u,i,opt_state)
        
        losses.append(loss)
        if i % 10 == 0:
            pbar.set_description(
                'ELBO: {:.2f}, Log Joint: {:.2f}, Log Prior: {:.2f}'.format(loss, log_joint, log_prior)
            )

    
    (recognition_params, joint_params) = params

    if joint_params.dynamics.B is None:
        recognition_params = recognition_params._replace(B=joint.dynamics.B)
    else:
        recognition_params = recognition_params._replace(B=joint_params.dynamics.B)
    
    recognition.params = recognition_params
    joint.set_params(joint_params)
    
    return losses


# %%
def solve_lds(
        key: jxr.PRNGKey,
        joint: fLDS,
        recognition: AmortizedLSTM,
        y: Float[Array, "num_batches num_timesteps emission_dim"],
        u: Float[Array, "num_batches num_timesteps stim_dim"],
        n_iter: int = 100,
        step_size: float = .1,
    ):

    opt_init, opt_update, get_params = optimizers.adam(step_size,b1=.8,b2=.9)
    opt_state = opt_init(joint.dynamics.params)
    
    params = get_params(opt_state)    
    

    def batch_ll(params,key,y,u):
        def ll(params,key,y,u):
            k1, key = jxr.split(key,2)
            # x = recognition.posterior_mean(recognition.params,y,u)
            x,_ = recognition(recognition.params,k1,y,u)
            # negative log likelihood
            ll = joint.dynamics.log_prob(params,x,u)
            return  ll.sum()

        fun = vmap(
            lambda y,u: ll(params,key,y,u),
            in_axes=(0,0),
            out_axes=0
        )
        ll_ = fun(y,u)
        '''
        \\log p_{\\theta}(y) = \\log \\int log p_{\theta}(x,y)
            = \\log \\E_{x} p_{\\theta}(x,y)
            \\approx \\log \\sum_{k} p_{\\theta}(x_k,y) where x_k \\sim p(x_k)
            = logsumexp(\\log p_{\\theta}(x_k,y))
        '''
        nll = -jax.scipy.special.logsumexp(ll_)

        return nll

    @jit
    def update(params,key,y,u,i,opt_state):
        loss, grads = value_and_grad(batch_ll)(params,key,y,u)
        opt_state = opt_update(i, grads, opt_state)
        params = get_params(opt_state)
        return loss, opt_state, params


    pbar = trange(n_iter)
    pbar.set_description('jit compiling ...')

    losses = []
    for i in pbar:
        key, k1 = jxr.split(key,2)
        loss, opt_state, params = update(params,k1,y,u,i,opt_state)
        losses.append(loss)

        if i % 10 == 0:
            pbar.set_description('Log Likelihood: {:.2f}'.format(loss))

    joint.dynamics.set_params(params)
    
    return losses