# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: Amin
"""

import numpy as np
import utils

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
        
        if u is None:
            du = np.zeros(x0.shape)

        x[0,:,:] = x0
        for i in range(1,len(t)):
            dx = dt*self.step(t[i],x[i-1])

            if b is not None:
                dx += dt*b(t[i])

            if u is not None:
                du = np.einsum('mn,km->kn',self.B,u[:,i])
            
            x[i] = (du==0).astype(float)*(x[i-1]+dx)+du

        return t,x
    
    def obs(self,x,t=None,u=None):
        raise NotImplementedError
    
    def linearize(self,x):
        t = np.array([0])
        step_ = lambda x: self.step(t,x[None,:])
        J = np.autograd.functional.jacobian(
            step_,x,create_graph=False,strict=False
        )
        return J
    
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
class Saddle(RateModel):
    def __init__(self,pm,discrete=False,B=None):
        '''simple saddle model
            cohs (list): list of the constant input drives (using -.1 and 0.1 as default)
            params (dict): dictionary of the parameters in the model (a,b,c) and system evolution
                (dt, euler timesep size), (ntrials, number of sample trials),
                (sigma, noise variance)
            time (int): number of 'units to run', time / dt is the number of steps
        '''
        keys = pm.keys()
        assert 'a' in keys and 'b' in keys and 'c' in keys
        assert 'sigma' in keys
        assert 'coh' in keys
        super(Saddle, self).__init__(pm,discrete=discrete,B=B)

    def step(self,t,x,u=None):
        dw = self.pm['sigma']*np.random.randn(*x.shape)
        dx = np.stack([
            self.pm['a']*x[:,0]**3 +\
                self.pm['b']*x[:,0] + self.pm['coh'],
                self.pm['c']*x[:,1] + self.pm['coh']
            ],
        ).T

        dxdt = dx+dw
        
        return dxdt

# %%
class LineAttractor(Linear):
    def __init__(self,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'l0' in keys and 'r0' in keys and 'eval1' in keys and 'sigma' in keys

        l0 = pm['l0']
        l0 /= np.linalg.norm(l0)
        evals = np.diag([0,pm['eval1']])

        R = np.array([pm['r0'],l0])
        L = np.linalg.inv(R)
        A = R @ evals @ L
        theta = np.radians(45)
        c,s = np.cos(theta), np.sin(theta)
        Mrot = np.array(((c, -s), (s, c)))
        self.pm['A'] = Mrot @ A

        super(LineAttractor, self).__init__(pm,discrete=discrete,B=B)

        

# %%
class PointAttractor(Linear):
    def __init__(self,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'a1' in keys and 'a2' in keys and 'sigma' in keys
        # make sure they're negative
        assert pm['a1'] < 0 and pm['a2'] < 0
        pm['A'] = np.diag([-np.abs(pm['a1']),-np.abs(pm['a2'])]) 
        super(PointAttractor, self).__init__(pm,discrete=discrete,B=B)
    

# %%
class FFIntegrator(Linear):
    def __init__(self,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'sigma' in keys
        pm['A'] = utils.ff_integrator_cnn(pm['D'])
        super(FFIntegrator, self).__init__(
            pm,discrete=discrete,B=B
        )

# %%
class LRIntegrator(Linear):
    def __init__(self,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'sigma' in keys
        pm['A'] = utils.lr_integrator_cnn(pm['D'])
        super(LRIntegrator, self).__init__(
            pm,discrete=discrete,B=B
        )

# %%
class RotationalDynamics(RateModel):
    def __init__(self,pm,discrete=False,B=None):
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
            ],
        ).T

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
    def __init__(self,pm,discrete=False,B=None):
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
            ],
        ).T

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