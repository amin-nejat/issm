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

import jax.numpy as jnp
import jax

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

    dataset_params, model_params, variational_params = pm['dataset_params'], pm['model_params'], pm['variational_params']

    seed = pm['model_params']['seed']
    file = args.output

    if not os.path.exists(file): os.makedirs(file)

    # Create a data loader and load training data
    # dataloader = loader.iPfLDSLoader(dataset_params)
    # y,u = dataloader.load_data()
    
    key = jax.random.PRNGKey(seed)
    k1, key = jax.random.split(key,2)

    dataloader = eval('loader.'+dataset_params['class'])(dataset_params)

    k1, key = jax.random.split(key,2)
    u = []
    for i in range(dataset_params['K']):
        u_,_ = utils.stimulation_protocol(
            k1,time_st=0,time_en=dataset_params['T'],
            dt=dataset_params['dt'],N=dataset_params['D'],
            stimulated=jnp.arange(dataset_params['D']),
            amplitude=1*jnp.ones(dataset_params['D']),
            stim_d=dataset_params['stim_d'],
            repetition=dataset_params['repetition'],
            sigma=dataset_params['stim_sigma']
        )
        u.append(u_)

    u = jnp.array(u)

    t,x_ = dataloader.run(
        dataset_params['T'],dt=dataset_params['dt'],u=u,
        x0=.01*jax.random.normal(
            k1,shape=(dataset_params['K'],dataset_params['D'])
        )
    )
    y_ = dataloader.obs(x_)
    y = jnp.array([y_[:,i] for i in range(y_.shape[1])])
    

    K,T,N = y.shape
    
    # Create model instances used for fitting
    k1, k2, key = jax.random.split(key,3)

    initial = models.InitialCondition(model_params['D'])
    lds = eval('models.'+model_params['lds'])(
        key=k1,D=model_params['D'],M=N,
        initial=initial,dt=model_params['dt'],
        B=dataloader.B,g=1,
        scale_tril=jnp.eye(model_params['D'])*model_params['dynamics_noise_scale']
    )
    
    k1, key = jax.random.split(key,2)
    emission = eval('models.'+model_params['emission'])(model_params['D'],N,key=k1)
    likelihood = eval('models.'+model_params['likelihood'])(
        N,
        scale_tril=jnp.eye(model_params['D'])*model_params['likelihood_noise_scale']
    )
    joint = models.JointfLDS(lds,emission,likelihood)

    k1, key = jax.random.split(key,2)

    # Create recognition instance for inference
    recognition = eval('inference.'+'AmortizedLSTM')(
        D=model_params['D'],N=N,M=N,T=T,key=k1
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

    k1, key = jax.random.split(key,2)


    x_corr, y_corr = [], []    
    for i in range(len(u)):
        k1, key = jax.random.split(key,2)
        x_smooth, lp = recognition.f(k1,y[i],u[i],recognition.params,lds.B)
        rate_smooth = emission.f(x_smooth,emission.params)
        
        k1, key = jax.random.split(key,2)
        y_smooth = likelihood.sample(rate_smooth,key=k1)

        # TODO: Is this comparable across different stimulation protocols
        y_corr.append(
            jnp.corrcoef(
            y_smooth[u[i]==0].flatten(),
            y[i][u[i]==0].flatten()
            )[0,1]
        )

        # TODO: Is this comparable across different stimulation protocols
        x_corr.append(
            jnp.corrcoef(
            x_smooth[u[i]==0].flatten(),
            x_[:,i][u[i]==0].flatten()
            )[0,1]
        )
        if i < 3 :
            visualizations.plot_signals(
                [x_[:,i],x_smooth],inp=u[i],
                colors=['k','r'],titlestr='$x$',labels=['True','Inferred'],
                save=True,file=file+'x_'+str(i)
            )
            visualizations.plot_signals(
                [y[i],rate_smooth],inp=u[i],
                colors=['k','r'],titlestr='$y$',labels=['True','Inferred'],
                save=True,file=file+'y_'+str(i)
            )

    stim_frac = lambda u: jnp.count_nonzero(u)/len(u)
    stats = {
        'stim_d': stim_frac(u.flatten()),
        'y-corr-train': y_corr,
        'x-corr-train': x_corr,
    }

    jnp.save(file+'stats',stats)
        

