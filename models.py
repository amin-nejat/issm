
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:41:04 2022

@author: Amin
"""

import jax.numpy as jnp
from jax import jit, lax
import jax.random as jxr

import numpyro.distributions as dist
from jax.example_libraries import stax

from params import ParamsfLDS, \
            ParamsLinearEmissions, \
            ParamsNNEmissions, \
            ParamsNormal, \
            ParamsConditionalNormal, \
            ParamsLinearDynamics, \
            ParamsMLPDynamics
            

# %%
class fLDS:
    def __init__(self, dynamics, emissions, likelihood):
        self.dynamics = dynamics
        self.emissions = emissions
        self.likelihood = likelihood

        self.params = ParamsfLDS(
            emissions = emissions.params,
            dynamics = dynamics.params,
            likelihood = likelihood.params
        )
        
    def sample(self, params: ParamsfLDS, y, u, key: jxr.PRNGKey):
        T, _ = y.shape
        k1, k2 = jxr.split(key,2)
        x = self.dynamics.sample(params.dynamics,x0=None,u=u,T=T,key=k1)
        suff_stats = self.emissions(params.emissions,x)
        y = self.likelihood.sample(params.likelihood,suff_stats,key=k2)
        
        return y

    def log_prob(self, params: ParamsfLDS, y, u, x):
        ld = self.dynamics.log_prob(params.dynamics,x,u)
        ld_prior = self.dynamics.log_prior(params.dynamics)
        suff_stats = self.emissions(params.emissions,x)
        le = self.likelihood.log_prob(params.likelihood,suff_stats,y=y)
        
        return ld.sum()+ld_prior.sum()+le.sum()
    
    def log_prob_dyn(self,params: ParamsfLDS, y, u, x):
        ld = self.dynamics.log_prob(params.dynamics,x,u)
        return ld.sum()
    
    def set_params(self, params: ParamsfLDS):
        self.dynamics.set_params(params.dynamics)
        self.emissions.params = params.emissions
        self.likelihood.params = params.likelihood

        self.params = params


# %%
class LinearEmission:
    def __init__(self,D: int, N: int):
        """Initialize and instance of `LinearEmission`

        Args:
            D (int): Dynamics dimension.
            N (int): Observation dimension.
        """
        self.D = D
        self.N = N

    def __call__(self, params: ParamsLinearEmissions, x):
        y = x@params.C
        return y


# %%
def _AdditiveCouplingLayer(dim, hidden_dim=64, reverse=False):
    """A stax-like Additive Coupling Layer."""
    dim1 = dim // 2
    dim2 = dim - dim1
    
    in_dim = dim2 if reverse else dim1
    out_dim = dim1 if reverse else dim2

    net_init, net_apply = stax.serial(
        stax.Dense(hidden_dim), stax.Relu,
        stax.Dense(hidden_dim), stax.Relu,
        stax.Dense(out_dim)
    )

    def init_fun(rng, input_shape):
        # input_shape is like (batch_size, dim)
        batch_size = input_shape[0] if input_shape and len(input_shape) > 1 else -1
        net_input_shape = (batch_size, in_dim)
        _, params = net_init(rng, net_input_shape)
        return input_shape, params

    def apply_fun(params, inputs, **kwargs):
        v1 = inputs[..., :dim1]
        v2 = inputs[..., dim1:]
        if reverse:
            y1 = v1 + net_apply(params, v2, **kwargs)
            y2 = v2
            return jnp.concatenate([y1, y2], axis=-1)
        else:
            y1 = v1
            y2 = v2 + net_apply(params, v1, **kwargs)
            return jnp.concatenate([y1, y2], axis=-1)
            
    return init_fun, apply_fun


class InjectiveFlowEmission:
    def __init__(self, D: int, N: int, key: jxr.PRNGKey, H: int=64, num_layers_g1: int = 3, num_layers_g2: int = 3):
        """Initialize an instance of `InjectiveFlowEmission`

        Args:
            D (int): Dynamics dimension.
            N (int): Observation dimension.
            key (jxr.PRNGKey): Random jax key.
            H (int, optional): Hidden dimension. Defaults to 64.
            num_layers_g1 (int, optional): Number of coupling layers for g1. Defaults to 3.
            num_layers_g2 (int, optional): Number of coupling layers for g2. Defaults to 3.
        """
        assert N >= D, "Observation dimension N must be >= Latent dimension D."
        self.D = D
        self.N = N

        g1_layers = [(_AdditiveCouplingLayer(dim=D, hidden_dim=H, reverse=(i % 2 != 0))) for i in range(num_layers_g1)]
        g2_layers = [(_AdditiveCouplingLayer(dim=N, hidden_dim=H, reverse=(i % 2 != 0))) for i in range(num_layers_g2)]

        g1_init, self.g1_apply = stax.serial(*g1_layers)
        g2_init, self.g2_apply = stax.serial(*g2_layers)

        k1, k2 = jxr.split(key)
        _, g1_params = g1_init(k1, (-1, D))
        _, g2_params = g2_init(k2, (-1, N))

        self.params = ParamsNNEmissions(theta=(g1_params, g2_params))

    def __call__(self, params: ParamsNNEmissions, x):
        g1_params, g2_params = params.theta
        z_warped = self.g1_apply(g1_params, x)
        padded_z = jnp.pad(z_warped, ((0, 0), (0, self.N - self.D))) if self.N > self.D else z_warped
        y = self.g2_apply(g2_params, padded_z)
        return y

# %%
class NeuralNetEmission:
    def __init__(self, D: int, N: int, key: jxr.PRNGKey, H: int=100, constraint=None):
        """Initialize and instance of `NeuralNetEmission`

        Args:
            D (int): Dynamics dimension.
            N (int): Observation dimension.
            key (jxr.PRNGKey): Random jax key.
            H (int, optional): Hidden dimension. Defaults to 100.
        """

        self.D = D
        self.N = N

        architecture = [
            stax.Dense(H),
            stax.Tanh,
            stax.Dense(H),
            stax.Tanh,
            stax.Dense(H),
            stax.Tanh,
            stax.Dense(N)
        ]

        if constraint is not None and constraint == 'positive':
            architecture += [stax.Softplus]

        initialize, self.f =  stax.serial(
            *architecture
        )
        _, theta = initialize(key, (self.D,))
        self.params = ParamsNNEmissions(
            theta=theta
        )

    def __call__(self, params: ParamsNNEmissions, x):
        y = self.f(params.theta,x)
        return y
    

# %%
class PoissonConditionalLikelihood:
    def __init__(self, D: int, params={}):
        self.D = D
        self.params = None
    
    def sample(self, params, suff_stats, key: jxr.PRNGKey):
        Y = dist.Poisson(suff_stats).to_event(1).sample(key)
        return Y
    
    def log_prob(self, params, suff_stats, y):
        return dist.Poisson(suff_stats).to_event(1).log_prob(y)

# %%
class NormalConditionalLikelihood:
    def __init__(self, D: int, params: ParamsConditionalNormal):
        self.D = D
        self.params = params

    def sample(self, params: ParamsConditionalNormal, suff_stats, key):
        y = dist.MultivariateNormal(suff_stats,scale_tril=params.scale_tril).sample(key)
        return y
    
    def log_prob(self, params: ParamsConditionalNormal, suff_stats, y):
        return dist.MultivariateNormal(suff_stats,scale_tril=params.scale_tril).log_prob(y)
    


#  %%
class InitialCondition:
    def __init__(self, D: int, params: ParamsNormal):
        self.D = D
        self.params = params
    
    def sample(self, params: ParamsNormal, key: jxr.PRNGKey):
        x0 = dist.MultivariateNormal(
            params.mu, scale_tril=params.scale_tril
        ).sample(key)
        return x0

    def log_prob(self, params: ParamsNormal, x0):
        lp = dist.MultivariateNormal(
            params.mu, scale_tril=params.scale_tril
        ).log_prob(x0)
        return lp

    
# %%
class LinearDynamics:
    """Differentiable representation of LDS for inference
    """
    def __init__(
            self, 
            D: int, 
            M: int, 
            initial: InitialCondition, 
            params: ParamsLinearDynamics,
            dt: float = .1, 
            sparsity: float = .1,
            train_B: bool = False,
            interventional: bool = True):
        """
        Args:
            D (int): Dynamics dimension.
            M (int): Input dimension.
            initial (InitialCondition): Initial distribution.
            dt (float, optional): Discretization. Defaults to .1.
            sparsity (float, optional): Sparsity of the B matrix. Defaults to .1.
            train_B (bool, optional): Makes B trainable.
        """
        
        self.D = D 
        self.M = M 
        self.dt = dt 
        self.initial = initial
        self.sparsity = sparsity
        self.interventional = interventional
        
        if not train_B: 
            self.B = params.B
            params = params._replace(B=None)

        self.set_params(params)
        
    def mean(self, params: ParamsLinearDynamics, T: int, key: jxr.PRNGKey, x0=None, u=None):
        A, B = params.A, params.B
        if B is None: B = self.B

        @jit
        def transition(carry, args):
            xs = carry
            u_new, _ = args
            
            inp = B@u_new

            if self.interventional:
                mu = (inp==0).astype(float)*((1-self.dt)*xs[-1]+(self.dt)*(A@xs[-1]))+inp
            else:
                mu = (1-self.dt)*xs[-1]+(self.dt)*(A@xs[-1])+inp
            
            xs = jnp.row_stack((xs[1:],mu))

            return xs, None
        
        
        if x0 is None:
            x0 = self.initial.sample(params.initial,key)

        if u is None:
            u = jnp.zeros((T,self.M))

        history = jnp.vstack((jnp.zeros((T-1,self.D)),x0[None]))
        
        xs, _ = lax.scan(
            transition, 
            history, 
            (u[1:],jnp.arange(T-1))
        )
        
        return xs
        
    
    def sample(self, params: ParamsLinearDynamics, T: int, key: jxr.PRNGKey, x0=None, u=None):
        A, B = params.A, params.B
        if B is None: B = self.B

        @jit
        def transition(carry, args):
            xs,k = carry
            u_new, _ = args

            k1, k = jxr.split(k,2)
            
            inp = B@u_new

            if self.interventional:
                mu = (inp==0).astype(float)*((1-self.dt)*xs[-1]+(self.dt)*(A@xs[-1]))+inp
            else:
                mu = (1-self.dt)*xs[-1]+(self.dt)*(A@xs[-1])+inp
            x_new = dist.MultivariateNormal(
                mu, scale_tril=params.scale_tril
                ).sample(k1)
            
            xs = jnp.row_stack((xs[1:],x_new))

            return (xs,k), None
        
        
        k1, key = jxr.split(key,2)
        if x0 is None:
            x0 = self.initial.sample(params.initial,k1)

        if u is None:
            u = jnp.zeros((T,self.M))

        history = jnp.vstack((jnp.zeros((T-1,self.D)),x0[None]))
        
        (xs, _), _ = lax.scan(
            transition, 
            (history,key), 
            (u[1:],jnp.arange(T-1))
        )
        
        return xs

    def log_prob(self, params: ParamsLinearDynamics, xs, us):
        A, B = params.A, params.B
        if B is None: B = self.B

        @jit
        def transition(carry, args):
            xs, lp = carry
            x_new, u_new, _ = args
            
            inp = B@u_new

            if self.interventional:
                mu = (inp==0).astype(float)*((1-self.dt)*xs[-1]+(self.dt)*(A@xs[-1]))+inp
            else:
                mu = (1-self.dt)*xs[-1]+(self.dt)*(A@xs[-1])+inp

            lp += dist.MultivariateNormal(
                mu,scale_tril=params.scale_tril
                ).log_prob(x_new)

            xs = jnp.row_stack((xs[1:],x_new))

            return (xs, lp), None
        
        T = len(xs)


        history = jnp.vstack((jnp.zeros((T-1,self.D)),xs[0][None]))

        (_, lp), _ = lax.scan(
            transition, 
            (history,self.initial.log_prob(params.initial,xs[0])), 
            (xs[1:],us[1:],jnp.arange(1,T))
        )

        return lp
    
    def log_prior(self, params: ParamsLinearDynamics):
        if params.B is None: return jnp.array([0])
        return -jnp.abs(params.B).sum()/self.sparsity

    def set_params(self, params: ParamsLinearDynamics):
        self.params = params
        self.initial.params = params.initial


# %%
class AnalyticMLPDynamics:
    """Differentiable representation of MLP dynamics for inference
    """
    def __init__(
            self, 
            D: int, 
            M: int, 
            initial: InitialCondition, 
            params: ParamsMLPDynamics,
            dt: float = .1, 
            train_B: bool = False,
            interventional: bool = True,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            activation: str = 'gelu',
            init_scale: float = 0.01,
            key: jxr.PRNGKey = None):
        """
        Args:
            D (int): Dynamics dimension.
            M (int): Input dimension.
            initial (InitialCondition): Initial distribution.
            params (ParamsMLPDynamics): Parameters for the MLP dynamics.
            dt (float, optional): Discretization. Defaults to .1.
            train_B (bool, optional): Makes B trainable.
            interventional (bool, optional): Whether to use hard interventions.
            hidden_dim (int): Hidden dimension of the MLP.
            num_hidden_layers (int): Number of hidden layers in the MLP.
            activation (str): Activation function ('gelu', 'softplus', 'tanh').
            init_scale (float): Scale for initializing the final layer weights.
            key (jxr.PRNGKey): JAX random key for initialization if params are not provided.
        """
        
        self.D = D 
        self.M = M 
        self.dt = dt 
        self.initial = initial
        self.interventional = interventional

        # Create MLP for drift
        _activations = {'gelu': stax.Gelu, 'softplus': stax.Softplus, 'tanh': stax.Tanh}
        if activation not in _activations:
            raise ValueError(f"activation must be one of {list(_activations.keys())}, got '{activation}'")
        act_fn = _activations[activation]

        layers = []
        for _ in range(num_hidden_layers):
            layers.extend([stax.Dense(hidden_dim), act_fn])
        layers.append(stax.Dense(D))
        
        initialize, self.mlp_apply = stax.serial(*layers)
        
        if params.mlp_params is None:
            if key is None:
                raise ValueError("A jax random key must be provided if mlp_params are not.")
            k1, key = jxr.split(key)
            _, mlp_params_list = initialize(k1, (-1, D))
            
            # Re-initialize last layer for small initial drift
            W, b = mlp_params_list[-1]
            k2, key = jxr.split(key)
            W_new = jxr.normal(k2, shape=W.shape) * init_scale
            b_new = jnp.zeros_like(b)
            mlp_params_list[-1] = (W_new, b_new)

            params = params._replace(mlp_params=tuple(mlp_params_list))

        if not train_B: 
            self.B = params.B
            params = params._replace(B=None)

        self.set_params(params)

    def _drift(self, params: ParamsMLPDynamics, x):
        return self.mlp_apply(params.mlp_params, x)
    
    def mean(self, params: ParamsMLPDynamics, T: int, key: jxr.PRNGKey, x0=None, u=None):
        B = params.B
        if B is None: B = self.B

        @jit
        def transition(carry, args):
            xs = carry
            u_new, _ = args
            
            inp = B@u_new

            drift = self._drift(params, xs[-1])

            if self.interventional:
                mu = (inp==0).astype(float)*((1-self.dt)*xs[-1]+(self.dt)*drift)+inp
            else:
                mu = (1-self.dt)*xs[-1]+(self.dt)*drift+inp
            
            xs = jnp.row_stack((xs[1:],mu))

            return xs, None
        
        
        if x0 is None:
            x0 = self.initial.sample(params.initial,key)

        if u is None:
            u = jnp.zeros((T,self.M))

        history = jnp.vstack((jnp.zeros((T-1,self.D)),x0[None]))
        
        xs, _ = lax.scan(
            transition, 
            history, 
            (u[1:],jnp.arange(T-1))
        )
        
        return xs
        
    
    def sample(self, params: ParamsMLPDynamics, T: int, key: jxr.PRNGKey, x0=None, u=None):
        B = params.B
        if B is None: B = self.B

        @jit
        def transition(carry, args):
            xs,k = carry
            u_new, _ = args

            k1, k = jxr.split(k,2)
            
            inp = B@u_new

            drift = self._drift(params, xs[-1])

            if self.interventional:
                mu = (inp==0).astype(float)*((1-self.dt)*xs[-1]+(self.dt)*drift)+inp
            else:
                mu = (1-self.dt)*xs[-1]+(self.dt)*drift+inp
            x_new = dist.MultivariateNormal(
                mu, scale_tril=params.scale_tril
                ).sample(k1)
            
            xs = jnp.row_stack((xs[1:],x_new))

            return (xs,k), None
        
        
        k1, key = jxr.split(key,2)
        if x0 is None:
            x0 = self.initial.sample(params.initial,k1)

        if u is None:
            u = jnp.zeros((T,self.M))

        history = jnp.vstack((jnp.zeros((T-1,self.D)),x0[None]))
        
        (xs, _), _ = lax.scan(
            transition, 
            (history,key), 
            (u[1:],jnp.arange(T-1))
        )
        
        return xs

    def log_prob(self, params: ParamsMLPDynamics, xs, us):
        B = params.B
        if B is None: B = self.B

        @jit
        def transition(carry, args):
            xs_carry, lp = carry
            x_new, u_new, _ = args
            
            inp = B@u_new

            drift = self._drift(params, xs_carry[-1])

            if self.interventional:
                mu = (inp==0).astype(float)*((1-self.dt)*xs_carry[-1]+(self.dt)*drift)+inp
            else:
                mu = (1-self.dt)*xs_carry[-1]+(self.dt)*drift+inp

            lp += dist.MultivariateNormal(
                mu,scale_tril=params.scale_tril
                ).log_prob(x_new)

            xs_carry = jnp.row_stack((xs_carry[1:],x_new))

            return (xs_carry, lp), None
        
        T = len(xs)


        history = jnp.vstack((jnp.zeros((T-1,self.D)),xs[0][None]))

        (_, lp), _ = lax.scan(
            transition, 
            (history,self.initial.log_prob(params.initial,xs[0])), 
            (xs[1:],us[1:],jnp.arange(1,T))
        )

        return lp
    
    def log_prior(self, params: ParamsMLPDynamics):
        if params.B is None: return jnp.array([0.])
        return jnp.array([0.])

    def set_params(self, params: ParamsMLPDynamics):
        self.params = params
        self.initial.params = params.initial