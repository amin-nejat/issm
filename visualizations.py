# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 12:53:41 2022

@author: Amin
"""

import warnings
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib

from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

colors_ = np.concatenate((
    cm.get_cmap('tab20c', 20).colors,
    cm.get_cmap('tab20b', 20).colors
))

# %%
def plot_image(
        data, 
        inputs=None,
        t=None,
        clim=None,
        cmap='gray',
        xlabel='Time',
        ylabel='$y$',
        fontsize=20,
        xticks=None,
        yticks=None,
        titlestr='', 
        save=False, 
        file=None
    ):
    
    plt.figure(figsize=(6,10))
    plt.imshow(
        data.T, 
        cmap, 
        alpha=1, 
        aspect='auto', 
        interpolation='none'
    )
    plt.gca().invert_yaxis()

    def highlight_cell(x,y, ax=None, **kwargs):
        rect = plt.Rectangle(
            (x-.5,y-.5),width=1,height=1,
            fill=False, **kwargs
        )
        ax = ax or plt.gca()
        ax.add_patch(rect)
        return rect

    if inputs is not None:
        u_x,u_y = np.where(inputs!=0)
        for i in range(len(u_x)):
            highlight_cell(
                u_x[i],u_y[i],
                color='blue',linewidth=.1
            )

    plt.xlabel(xlabel,fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    if t is not None:
        plt.xticks(np.arange(len(t)),['{:.2f}'.format(t_) for t_ in t],fontsize=fontsize)
    else:
        plt.xticks(fontsize=fontsize)
        
    plt.yticks(fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    if xticks is None:
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
    else:
        plt.xticks(np.arange(len(xticks)), xticks, fontsize=fontsize)

    if yticks is None:
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
    else:
        plt.yticks(np.arange(len(yticks)), yticks, fontsize=fontsize)

    plt.grid(False)

    
    if clim is not None:
        plt.clim(clim)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
    



# %%
def compare_smoothened_predictions(Ey, Ey_true, Covy, data, xlim=None, save=False, file=None):
    data_dim = data.shape[-1]

    plt.figure(figsize=(15, 6))
    plt.plot(Ey_true + 10 * jnp.arange(data_dim))
    plt.plot(Ey + 10 * jnp.arange(data_dim), "--k")
    for i in range(data_dim):
        plt.fill_between(
            jnp.arange(len(data)),
            10 * i + Ey[:, i] - 2 * jnp.sqrt(Covy[:, i, i]),
            10 * i + Ey[:, i] + 2 * jnp.sqrt(Covy[:, i, i]),
            color="k",
            alpha=0.25,
        )
    plt.xlabel("time")
    plt.ylabel("data and predictions (for each neuron)")

    if xlim is not None:
        plt.xlim(xlim)

    plt.plot([0], "--k", label="Predicted")  # dummy trace for legend
    plt.plot([0], "-k", label="True")
    plt.legend(loc="upper right")
    

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()



# %%
def plot_states(
        states,input=None,labels=None,colors=None,new_fig=True,
        titlestr='',fontsize=15,save=False,file=None
    ):

    if new_fig:
        plt.figure(figsize=(5,5))

    
    if states[0].shape[1] == 2:
        states_pca = states
    else:
        pca = PCA(n_components=2)
        pca.fit(states[0])
        states_pca = [[]]*len(states)
        for i in range(len(states)):
            states_pca[i] = pca.transform(states[i])
            states_pca[i] = states_pca[i]-states_pca[i].mean(0)[None]
    
    for i in range(len(states_pca)):
        color = colors[i] if colors is not None else 'k'
        label = labels[i] if labels is not None else ''
        plt.plot(
            states_pca[i][:,0],states_pca[i][:,1],linewidth=.5,ls='--',
            c=color,label=label
        )
        plt.gca().scatter(
            states_pca[i][:,0],states_pca[i][:,1],lw=2,c=color,
            alpha=[jnp.linspace(0,1,len(states_pca[i]))]
        )

        if input is not None:
            plt.gca().scatter(
                states_pca[i][input.sum(1)!=0,0],
                states_pca[i][input.sum(1)!=0,1],
                lw=.1,c='b'
            )
    
    
        
    plt.xlabel('PC1',fontsize=fontsize)
    plt.ylabel('PC2',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
    
    plt.tight_layout()
    plt.grid(False)
    if label is not None:
        plt.legend(fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def plot_scatter(x,y,titlestr='',xlabel='',ylabel='',fontsize=15,save=False,file=None):
    plt.axline((0, 0), slope=1, linestyle='--', color='k')
    
    plt.scatter(x,y)

    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.title(titlestr,fontsize=fontsize)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))
    
    

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def plot_images(ims,titlestr='',label='y',fontsize=15,save=False,file=None):
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=len(ims),
        figsize=(3*len(ims)*(len(ims[0][0])/len(ims[0])), 3)
    )
    print(axes)
    
    vmin = np.array(ims).min()
    vmax = np.array(ims).max()

    if len(ims) == 1: axes = [axes]
    for i in range(len(ims)):
        im = axes[i].imshow(ims[i], vmin=vmin, vmax=vmax, cmap='seismic')

        axes[i].set_xticks(np.arange(len(ims[i][0])))
        axes[i].set_xticklabels(
            labels=['$'+label+'_{}$'.format(j+1) for j in range(len(ims[i][0]))],
            fontsize=fontsize
        )
        axes[i].set_yticks(np.arange(len(ims[i])))
        axes[i].set_yticklabels(
            labels=['$'+label+'_{}$'.format(j+1) for j in range(len(ims[i]))],
            fontsize=fontsize
        )

        axes[i].tick_params(axis=u'both', which=u'both',length=0)
        axes[i].grid(b=None)



    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    plt.suptitle(titlestr,fontsize=fontsize)

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

def plot_loss(loss,ylabel='',fontsize=15,save=False,file=None):
    plt.figure(figsize=(3,3))
    plt.plot(loss,'k')
    

    plt.xlabel('Iteration',fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))

    plt.grid(False)

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


def plot_performance(
        dims,
        performance,
        ylabel,
        legends=None,
        titlestr='',
        xlabel='Latent Dimension',
        error='std',
        fontsize=15,
        save=False,
        file=None
    ):

    plt.figure(figsize=(3,2))
    keys = list(performance.keys())

    for i,key in enumerate(keys):
        if error == 'std':
            yerr = np.nanstd(np.array(performance[key]),axis=1)
        if error == 'sem':
            yerr = stats.sem(np.array(performance[key]),axis=1,nan_policy='omit')

        plt.errorbar(
            dims,np.nanmean(performance[key],1),
            yerr=yerr,
            lw=3,label=legends[i] if legends is not None else None
        )
        

    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)

    plt.title(titlestr,fontsize=fontsize)

    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.ylim([-.2,1.2])

    plt.grid(False)

    if save:
        plt.savefig(file+key+'.png',format='png')
        plt.savefig(file+key+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def plot_states_3d(
        states,inputs,labels=None,angles=[30,60],
        titlestr='',fontsize=15,dotsize=5,linewidth=.1,
        save=False,file=None
    ):
    plt.figure(figsize=(10,10))

    ax = plt.figure().add_subplot(projection='3d')
    
    pca = PCA(n_components=3)
    pca.fit(states[0])
    states_pca = [[]]*len(states)
    for i in range(len(states)):
        states_pca[i] = pca.transform(states[i])
        states_pca[i] = states_pca[i]-states_pca[i].mean(0)[None]
    
    for i in range(len(states_pca)):
        ax.plot(
            states_pca[i][:,0],states_pca[i][:,1],states_pca[i][:,2],
            linewidth=linewidth,ls='--',
            c='k',
        )
        ax.scatter(
            states_pca[i][:,0],states_pca[i][:,1],states_pca[i][:,2],
            s=dotsize,
            lw=linewidth*5,c='k',
            alpha=[jnp.linspace(.5,1,len(states_pca[i]))],
        )

        if inputs is not None and inputs[i].sum() != 0:
            stimulated = states_pca[i][np.where(inputs[i].sum(1)!=0)]
            ax.scatter(
                stimulated[:,0],stimulated[:,1],stimulated[:,2],
                s=dotsize*10,facecolors='none',
                lw=linewidth*20,edgecolors=color_dict[labels[i]],
                label=label_dict[labels[i]]
            )
    
    plt.legend(fontsize=fontsize,loc='upper right',bbox_to_anchor=(0.5,-0.05),)
    ax.set_xlabel('PC1',fontsize=fontsize)
    ax.set_ylabel('PC2',fontsize=fontsize)
    ax.set_zlabel('PC3',fontsize=fontsize)

    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='z', labelsize=fontsize)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.gca().zaxis.set_major_locator(plt.MaxNLocator(5))

    plt.title(titlestr,fontsize=fontsize)
    plt.tight_layout()

    ax.view_init(
        elev=angles[0], azim=-angles[1]
    )
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def plot_violin(
        performance: dict,
        titlestr: str = '',
        colors: list = ['k','r'],
        labels: list = ['Train', 'Test'],
        fontsize: int = 15,
        save=False,
        file=None
    ):
    plt.figure(figsize=(3,5))

    
    violin = plt.violinplot(performance.values(),showmedians=True)

    
    for l,pc in enumerate(violin['bodies']):
        pc.set_facecolor(colors[l])
        pc.set_alpha(0.4)
        pc.set_edgecolor(colors[l])

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violin[partname]
        vp.set_alpha(0.4)
        vp.set_edgecolor(colors)

    plt.xticks(
        np.arange(len(performance))+1,
        labels, 
        fontsize=fontsize, rotation=0
    )

    plt.yticks(fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    plt.grid(False)

    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def plot_signals(
        X,
        inp=None,
        t=None,
        colors=None,
        labels=None,
        ylabel='',
        titlestr='',
        fontsize=15,
        linewidth=2,
        margin=1.,
        save=False,
        file=None
    ):
    plt.figure(figsize=(3,1*len(X[0].T)))
    
    
    offset = np.append(0.0, np.nanmax(X[0][:,0:-1,],0)-np.nanmin(X[0][:,0:-1],0))
    shifts = np.cumsum(offset+margin)
    for i,x in enumerate(X):
        color = colors[i] if colors is not None else 'k'
        label = labels[i] if labels is not None else ''
        s = (x-np.nanmin(X[0],0)[None,:]+shifts[None,:])
        
        if t is not None: plt.plot(t,s,linewidth=linewidth,color=color,label=label)
        else: plt.plot(s,linewidth=linewidth,color=color,label=label)
        
        plt.grid('off')

        if i == 0: plt.ylim([s.min()-margin,s.max()+margin])

    
    if inp is not None:
        for j in range(inp.shape[1]):
            t_ = np.arange(len(inp)) if t is None else t
            plt.vlines(
                x=t_[inp[:,j]!=0],
                ymin=(shifts[j]),
                ymax=(shifts[j]+np.diff(shifts).min()), 
                color='b',
                alpha=.1,
                lw=1.
            )
    
    
    plt.xlabel('Time',fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))

    plt.grid(False)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    
    if labels is not None:
        plt.legend(handles, labels, loc='center left', fontsize=fontsize, bbox_to_anchor=(1, 0.5))


    plt.title(titlestr,fontsize=fontsize)
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def flow_2d(
        funs,
        colors=['k'],
        data=None,
        inputs=None,
        xlim=[-1,1],
        ylim=[-1,1],
        n_points=10,
        titlestr='',
        fontsize=15,
        xlabel='$x^{(1)}$',
        ylabel='$x^{(2)}$',
        scale=1,
        show=True,
        save=False,
        file=None
    ):

    if data is not None:
        xlim = (data[...,0].min()-.1, data[...,0].max()+.1)
        ylim = (data[...,1].min()-.1, data[...,1].max()+.1)

    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)

    # Create a quiver plot
    fig, ax = plt.subplots()

    for i in range(len(funs)):
        fun = funs[i]
        color = colors[i]

        Q = ax.quiver(
            X, 
            Y, 
            np.zeros_like(X), 
            np.zeros_like(Y), 
            pivot='mid', 
            scale=1,
            color=color
        )

        matrix = fun(np.stack((X.flatten(),Y.flatten())).T).T
        Q.set_UVC(scale*matrix[0], scale*matrix[1])

    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))

    plt.title(titlestr,fontsize=fontsize)

    if data is not None:
        for i in range(len(data)):
            plt.plot(data[i,:,0],data[i,:,1], 'k', lw=.1)
            if inputs is not None:
                plt.scatter(
                    data[i][inputs[i].sum(1)!=0,0],
                    data[i][inputs[i].sum(1)!=0,1],
                    lw=.1,c='b',
                )
    
    
    plt.tight_layout()

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        if show:
            plt.show()
    
