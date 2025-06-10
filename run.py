# %%
# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import models
import inference
import visualizations
import loader
import utils
import params
import jax.numpy as jnp
import jax.random as jxr
import jax
from jax import vmap

import argparse
import yaml

import os

# %%
def get_args():
    '''Parsing the arguments when this file is called from console
    '''
    parser = argparse.ArgumentParser(description='Runner for CCM',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c', metavar='Configuration',help='Configuration file address',default='/')
    parser.add_argument('--output', '-o', metavar='Output',help='Folder to save the output and results',default='/')
    
    return parser.parse_args()

# %%
if __name__ == '__main__':

    # Read parameters from config file
    args = get_args()
    with open(args.config, 'r') as stream: pm = yaml.safe_load(stream)

    stim_params, dataset_params, model_params, variational_params = pm['stim_params'], pm['dataset_params'], pm['model_params'], pm['variational_params']

    seed = pm['model_params']['seed']
    file = args.output

    if not os.path.exists(file): os.makedirs(file)


    key = jax.random.PRNGKey(seed)
    k1, key = jax.random.split(key,2)

    if "N" in dataset_params:
        C = jxr.orthogonal(k1, n=dataset_params['N'])[:dataset_params['D'],:] 
        # C = jxr.normal(k1,shape=(dataset_params['D'],dataset_params['N']))
        dataset_params['C'] = C


    dataloader = eval('loader.'+dataset_params['class'])(dataset_params)

    k1, key = jax.random.split(key,2)
    u = []
    for i in range(dataset_params['K']):
        u_,_ = utils.stimulation_protocol(
            k1,time_st=0,time_en=dataset_params['T'],
            dt=dataset_params['dt'],N=dataset_params['D'],
            stimulated=jnp.arange(dataset_params['D']),
            amplitude=1*jnp.ones(dataset_params['D']),
            stim_d=stim_params['stim_d'],
            repetition=stim_params['repetition'],
            sigma=stim_params['stim_sigma']
        )
        u.append(u_)

    u = jnp.array(u)

    t,x_ = dataloader.run(
        dataset_params['T'],dt=dataset_params['dt'],u=u,
        x0=1*jax.random.normal(
            k1,shape=(dataset_params['K'],dataset_params['D'])
        )
    )
    y_ = dataloader.obs(x_)

    y_ += dataset_params['obs_noise_scale']*jxr.normal(k1,shape=y_.shape)
    
    y = y_.transpose(1,0,2)
    x = x_.transpose(1,0,2)

    K,T,N = y.shape

    print(K,T,N)
    
    # Create model instances used for fitting
    k1, key = jax.random.split(key,2)

    initial_params = params.ParamsNormal(
        mu = jnp.zeros(model_params['D']),
        scale_tril = model_params['dt']*jnp.eye(model_params['D'])
    )

    initial = models.InitialCondition(model_params['D'],initial_params)

    lds_params = params.ParamsLinearDynamics(
        scale_tril = model_params['dt']*jnp.eye(model_params['D']),
        A = jxr.normal(k1, shape=(model_params['D'],model_params['D'])),
        B = jnp.eye(model_params['D']),
        initial = initial_params
    )

    lds_config = model_params['lds_params']

    lds = models.LinearDynamics(
        D=model_params['D'],
        M=model_params['D'],
        initial=initial,
        params=lds_params,
        dt=model_params['dt'],
        train_B=lds_config['train_B'],
        sparsity=lds_config['sparsity'] if lds_config['train_B'] else 0.,
        interventional=lds_config['interventional']
    )

        
    likelihood_params = params.ParamsConditionalNormal(
        scale_tril = model_params['likelihood_noise_scale']*jnp.eye(N)
    )
    likelihood = eval('models.'+model_params['likelihood'])(
        N,likelihood_params
    )


    k1, key = jax.random.split(key,2)
    emission = eval('models.'+model_params['emission'])(
        model_params['D'],
        N,
        key=k1,
        **model_params['emission_params']
    )
    
    joint = models.fLDS(lds,emission,likelihood)

    k1, key = jax.random.split(key,2)

    # Create recognition instance for inference
    recognition = inference.AmortizedLSTM(
        D=model_params['D'],N=N,M=model_params['D'],T=T,key=k1,
        interventional=lds_config['interventional']
    )

    # Run inference
    k1, key = jax.random.split(key,2)

    loss = inference.infer(
        k1,joint,recognition,y,u,
        n_iter=variational_params['n_iter'],
        step_size=variational_params['step_size'],
        gamma=variational_params['gamma']
    )

    # Visualize loss
    visualizations.plot_loss(
        loss,ylabel='ELBO',save=True,file=file+'loss'
    )   
    
    k1, k2, key = jxr.split(key,3)



    # # Test data
    # k1, key = jax.random.split(key,2)
    # u = []
    # for i in range(dataset_params['K']):
    #     u_,_ = utils.stimulation_protocol(
    #         k1,time_st=0,time_en=dataset_params['T'],
    #         dt=dataset_params['dt'],N=dataset_params['D'],
    #         stimulated=jnp.arange(dataset_params['D']),
    #         amplitude=1*jnp.ones(dataset_params['D']),
    #         stim_d=stim_params['stim_d'],
    #         repetition=stim_params['repetition'],
    #         sigma=stim_params['stim_sigma']
    #     )
    #     u.append(u_)

    # u = jnp.array(u)

    # t,x_ = dataloader.run(
    #     dataset_params['T'],dt=dataset_params['dt'],u=u,
    #     x0=.01*jax.random.normal(
    #         k1,shape=(dataset_params['K'],dataset_params['D'])
    #     )
    # )
    # y_ = dataloader.obs(x_)
    
    # y = y_.transpose(1,0,2)
    # x = x_.transpose(1,0,2)

    
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



    stim_frac = lambda u: jnp.count_nonzero(u)/len(u)
    stats = {
        'stim_d': stim_frac(u.flatten()),
        # 'y-corr-train': [jnp.corrcoef(mean[i].flatten(),y[i].flatten())[0,1] for i in range(len(y))],
        # 'x-corr-train':  [jnp.corrcoef(x_smooth[i].flatten(),x[i].flatten())[0,1] for i in range(len(y))],
        'x-corr-train': jnp.corrcoef(x[u==0].flatten(),x_smooth[u==0].flatten())[0,1],
        'y-corr-train': jnp.corrcoef(y.flatten(),y_smooth.flatten())[0,1]
    }

    jnp.save(file+'stats',stats)
# %%
