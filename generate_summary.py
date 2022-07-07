#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition, manifold
from matplotlib import cm
import scipy.io
import os
from datetime import datetime
import pytz
from scipy.stats import ortho_group
import argparse
import json
import time
import shutil

import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.ops.rnn_cell_impl import RNNCell
import tensorflow.keras.backend as K

from numpy import diag
from numpy import dot
from scipy.linalg import svd
import seaborn as sns
import math


# In[2]:


n_eachring = 32
n_input, n_output = 1+n_eachring, 1+n_eachring
batch_size_train = 64
hp = {
        # batch size for training
        'batch_size_train': batch_size_train,
        # batch size for validation
        'batch_size_val': batch_size_train,
        # Type of loss functions
        'loss_type': 'lsq',
        # Optimizer
        'optimizer': 'adam',
        # Type of activation runctions, relu, softplus, tanh, elu
        'activation': 'relu',
        # Time constant (ms)
        'tau': 100,
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'alpha': 0.2,
        # recurrent noise
#         'sigma_rec': 0.05,
        'sigma_rec': 0,
        # input noise
#         'sigma_x': 0.01,
        'sigma_x': 0,
        # leaky_rec weight initialization, diag, randortho, randgauss
        'w_rec_init': 'randortho',
        # a default weak regularization prevents instability
        'l1_h': 0,
        # l2 regularization on activity
        'l2_h': 0,
        # l2 regularization on weight
        'l1_weight': 0,
        # l2 regularization on weight
        'l2_weight': 0,
        # l2 regularization on deviation from initialization
        'l2_weight_init': 0,
        # Stopping performance
        'target_perf': 1.,
        # number of units each ring
        'n_eachring': n_eachring,
        # first input index for rule units
        'rule_start': 1+n_eachring,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of recurrent units
        'n_rnn': 256,
        # learning rate
        'learning_rate': 0.0001,
        # number of display epochs
        'steps_per_epoch': 1,
        # number of fixed locations for isomap
        'n_loc': 128,
        # accuracy threshold to stop training
        'accuracy_threshold': 0.9,
        }


# In[3]:


n_loc = 128
n_stim_loc1, n_stim_loc2, repeat = stim_loc_shape = n_loc, n_loc, 1
stim_loc_size = np.prod(stim_loc_shape)
ind_stim_loc1, ind_stim_loc2, ind_repeat = np.unravel_index(range(stim_loc_size),stim_loc_shape)
stim1_locs = 2*np.pi*ind_stim_loc1/n_stim_loc1
stim2_locs = 2*np.pi*ind_stim_loc2/n_stim_loc2

palette1 = cm.get_cmap('autumn',n_loc+15)
palette1 = [palette1(i)[:3] for i in range(n_loc)]
palette2 = cm.get_cmap('summer',n_loc+15)
palette2 = [palette2(i)[:3] for i in range(n_loc)]

color1=np.array(palette1)[ind_stim_loc1]
color2=np.array(palette2)[ind_stim_loc2]


# In[4]:


class LeakyRNNCell2(tf.keras.layers.Layer):

    def __init__(self,hp,**kwargs):

        self.units = hp['n_rnn']
        self.state_size = hp['n_rnn']
        
        activation = hp['activation']
        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'power':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self._activation = lambda x: tf.tanh(tf.nn.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')
            

        self.rng = hp['rng']
        self.seed = hp['seed']
        self._alpha = hp['alpha']
        self._sigma = np.sqrt(2 / self._alpha) * hp['sigma_rec']
        
        super(LeakyRNNCell2, self).__init__(**kwargs)

    def build(self, input_shape):
        
        w_in0 = (self.rng.randn(input_shape[-1], self.units) /
                 np.sqrt(input_shape[-1]) * self._w_in_start)
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), dtype=tf.float32,
                                      initializer=tf.constant_initializer(w_in0),
                                      name='kernel')
        w_rec0 = self._w_rec_start*ortho_group.rvs(dim=self.units, random_state=self.rng)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), dtype=tf.float32,
                                                initializer=tf.constant_initializer(w_rec0),
                                                name='recurrent_kernel')
        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)
    
#         self.kernel = self.add_weight(
#                 name='kernel',
#                 shape=[input_shape[-1] + self.units, self.units], 
#                 dtype=tf.float32,
#                 initializer=tf.constant_initializer(matrix0))
        
        self._bias = self.add_weight(
                name='bias',
                shape=[self.units],
                dtype=tf.float32,
                initializer=tf.zeros_initializer())
        
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
#         h = tf.keras.backend.dot(array_ops.concat([inputs, prev_output], 1), self.kernel)
        h = tf.keras.backend.dot(inputs, self.kernel)
        h = h + tf.keras.backend.dot(prev_output, self.recurrent_kernel)
        h = tf.nn.bias_add(h, self._bias)
        noise = tf.random.normal(tf.shape(prev_output), mean=0, stddev=self._sigma, seed=self.seed)
        h = h + noise
        output = self._activation(h)
        output = (1-self._alpha) * prev_output + self._alpha * output
        return output, [output]


# In[5]:


def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))

class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, tdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            tdim: int, number of time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32' # This should be the default
        self.config = config
        self.dt = self.config['dt']

        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.pref  = np.arange(0,2*np.pi,2*np.pi/self.n_eachring) # preferences

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)
        if self.config['loss_type'] == 'lsq':
            self.y[:,:,:] = 0.05
        # y_loc is the stimulus location of the output, -1 for fixation, (0,2 pi) for response
        self.y_loc = -np.ones((tdim, batch_size)      , dtype=self.float_type)

        self._sigma_x = config['sigma_x']*np.sqrt(2/config['alpha'])

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        """Add an input or stimulus output.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added, only for loc_type=stim or out
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float or list, strength of input or target output
            mods: int or list, modalities of input or target output
        """

        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]: offs[i], i, 0] = 1
            elif loc_type == 'stim':
                # Assuming that mods[i] starts from 1
                self.x[ons[i]: offs[i], i, 1+(mods[i]-1)*self.n_eachring:1+mods[i]*self.n_eachring]                     += self.add_x_loc(locs[i])*strengths[i]
            elif loc_type == 'fix_out':
                # Notice this shouldn't be set at 1, because the output is logistic and saturates at 1
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 0] = 0.8
                else:
                    self.y[ons[i]: offs[i], i, 0] = 1.0
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 1:] += self.add_y_loc(locs[i])*strengths[i]  #target response
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]: offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i]: offs[i], i] = locs[i] #location
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape)*self._sigma_x

    def add_c_mask(self, pre_offs, post_ons):
        """Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """

        pre_on   = int(100/self.dt) # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)

        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                c_mask[post_ons[i]:, i, :] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i, :] = 1.

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 2. # Fixation is important

            self.c_mask = c_mask.reshape((self.tdim*self.batch_size, self.n_output))
        else:
            c_mask = np.zeros((self.tdim, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                # Having it larger than 1 encourages the network to achieve higher performance
                c_mask[post_ons[i]:, i] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i] = 1.

            self.c_mask = c_mask.reshape((self.tdim*self.batch_size,))
            self.c_mask /= self.c_mask.mean()

    def add_rule(self, rule, on=None, off=None, strength=1.):
        """Add rule input."""
        if isinstance(rule, int):
            self.x[on:off, :, self.config['rule_start']+rule] = strength
        else:
            ind_rule = get_rule_index(rule, self.config)
            self.x[on:off, :, ind_rule] = strength

    def add_x_loc(self, x_loc):
        """Input activity given location."""
        dist = get_dist(x_loc-self.pref)  # periodic boundary
        dist /= np.pi/8
        return 0.8*np.exp(-dist**2/2)

    def add_y_loc(self, y_loc):
        """Target response given location."""
        dist = get_dist(y_loc-self.pref)  # periodic boundary
        if self.config['loss_type'] == 'lsq':
            dist /= np.pi/8
            y = 0.8*np.exp(-dist**2/2)
        else:
            # One-hot output
            y = np.zeros_like(dist)
            ind = np.argmin(dist)
            y[ind] = 1.
        return y


# In[6]:



def plot_loss_over_epochs(history, foldername=''):
    plt.figure(figsize=(10,8))
    plt.plot(history.history['loss'],label="Training set loss")
    # plt.plot(history.history['val_loss'],label="Validation set loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('%sloss_over_epochs.png'%foldername)

def get_delay_bins(delay):
    dt=20
    stim1_ons = int(500/dt)
    stim1_offs = stim1_ons + int(300/dt)
    stim2_ons =stim1_offs + int(1000/dt)
    stim2_offs = stim2_ons + int(300/dt)
    fix_offs  = stim2_offs + int(1000/dt)

    baseline = (0,stim1_ons)

    if delay == 1:
            delay_bins = (stim2_ons - int(500/dt),stim2_ons)

    elif delay == 2:
            delay_bins = (fix_offs - int(500/dt),fix_offs)

    return delay_bins

def plot_tuning_curves(Z,foldername='',filename=''):
    input
    fig,ax = plt.subplots(min(Z.shape[1],10),1,figsize=(5,min(Z.shape[1],10)*3))
    for i in range(min(Z.shape[1],10)):
        if Z.shape[1]<10:
            neuron = i
        else:
            neuron = i*int(Z.shape[1]/10)
        for loc1 in [j*10 for j in range(int(128/10))]:
            df = pd.DataFrame({'first_stim':ind_stim_loc1[ind_stim_loc1==loc1],'second_stim':ind_stim_loc2[ind_stim_loc1==loc1],'activity':Z[:,neuron][ind_stim_loc1==loc1]})
            x = df['second_stim']
            y = df['activity']
            ax[i].scatter(x,y,s=1, color=palette1[loc1], label=loc1)
        ax[i].set_ylabel('%dth neuron'%neuron, fontsize=13)
    ax[0].legend(loc='upper left', bbox_to_anchor= (1.05, 1.05), title='Stim 1')
    ax[0].set_xlabel('Stim 2', fontsize=13)
    ax[0].xaxis.set_label_position('top') 
    if filename != '':
        fig.tight_layout()
        fig.savefig(foldername+filename+'_stim2')
    plt.show()
    plt.close()
        
    fig,ax = plt.subplots(min(Z.shape[1],10),1,figsize=(5,min(Z.shape[1],10)*3))
    for i in range(min(Z.shape[1],10)):
        if Z.shape[1]<10:
            neuron = i
        else:
            neuron = i*int(Z.shape[1]/10)
        for loc2 in [j*10 for j in range(int(128/10))]:
            df = pd.DataFrame({'first_stim':ind_stim_loc1[ind_stim_loc2==loc2],'second_stim':ind_stim_loc2[ind_stim_loc2==loc2],'activity':Z[:,neuron][ind_stim_loc2==loc2]})
            x = df['first_stim']
            y = df['activity']
            ax[i].scatter(x,y,s=1, color=palette2[loc2], label=loc2)
        ax[i].set_ylabel('%dth neuron'%neuron, fontsize=13)
    ax[0].legend(loc='upper left', bbox_to_anchor= (1.05, 1.05), title='Stim 2')
    ax[0].set_xlabel('Stim 1', fontsize=13)
    ax[0].xaxis.set_label_position('top') 
    if filename != '':
        fig.tight_layout()
        fig.savefig(foldername+filename+'_stim1')
    plt.show()
    plt.close()
    
def fit_isomap(data_to_use, n_neighbors = 15, target_dim = 3):
    iso_instance = manifold.Isomap(n_neighbors = n_neighbors, n_components = target_dim)
    proj = iso_instance.fit_transform(data_to_use)
    return proj

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_isomap(data_plot, color, annotate=False):
    fig = plt.figure(figsize=(16,16),dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    if annotate:
        ax.scatter(data_plot[:,0], data_plot[:,1], data_plot[:,2], 
            s=5, alpha=1, edgecolor='face',c=color)
        label = 0
        for xyz in zip(data_plot[:,0], data_plot[:,1], data_plot[:,2]):
            x, y, z = xyz
            ax.text(x, y, z, '%s' % (label), size=5, zorder=1, color='k')
            label += 1
    else:
        ax.scatter(data_plot[:,0], data_plot[:,1], data_plot[:,2], 
            s=20, alpha=1, edgecolor='face',c=color)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    return fig, ax

def plot_single_distractor_or_target(palette, xlim, ylim, zlim, label_plot, proj_plot, annotate=False, filename=''):

    color=np.array(palette)[label_plot]

    # h0_longest,h1_longest,h2_longest = run_ripser(proj_plot,figure_dir+'ripser'+figure_subscript)
    fig, ax = plot_isomap(data_plot=proj_plot, color=color, annotate=annotate)
    plt.setp(ax, xlim=xlim, ylim=ylim, zlim=zlim)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    plt.show()
    plt.close(fig) 


def plot_all_isomap_figures(proj,foldername='',filename=''):
    fig,ax = plot_isomap(data_plot=proj, color=color1)
    set_axes_equal(ax)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(foldername+filename+'_target_isomap.png')
    plt.show()
    plt.close(fig)

    fig,ax = plot_isomap(data_plot=proj, color=color2)
    set_axes_equal(ax)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(foldername+filename+'_distractor_isomap.png')
    plt.show()
    plt.close(fig)

    xlim=fig.gca().get_xlim()
    ylim=fig.gca().get_ylim()
    zlim=fig.gca().get_zlim()

    num=0
    indices = ind_stim_loc1==num
    label_plot = ind_stim_loc2[indices]
    proj_plot = proj[indices,:]
    plot_single_distractor_or_target(palette = palette2, xlim = xlim, ylim = ylim, zlim = zlim, label_plot=label_plot, proj_plot = proj_plot, filename = foldername+filename+'_single_target.png')

    num=0
    indices = ind_stim_loc2==num
    label_plot = ind_stim_loc1[indices]
    proj_plot = proj[indices,:]
    plot_single_distractor_or_target(palette = palette1, xlim = xlim, ylim = ylim, zlim = zlim, label_plot=label_plot, proj_plot = proj_plot, filename = foldername+filename+'_single_distractor.png')


# In[109]:


def get_c_mask(batch_dim):

    dt = hp['dt']
    pre_on   = int(100/dt)
    
    stim1_strengths = 1
    stim2_strengths = 1
    stim1_ons = int(500/dt)
    stim1_offs = stim1_ons + int(300/dt)
    stim2_ons =stim1_offs + int(1000/dt)
    stim2_offs = stim2_ons + int(300/dt)
    fix_offs  = stim2_offs + int(1000/dt)
    output_2_on = fix_offs + int(500/dt)
    tdim = output_2_on + int(500/dt)
    check_ons = fix_offs + int(100/dt)
    
    pre_offs=fix_offs
    post_ons=check_ons
    
    c_mask = np.zeros(batch_dim, dtype='float32')
    for i in range(batch_dim[0]):
        c_mask[i, post_ons:, :] = 5.
        c_mask[i, pre_on:pre_offs, :] = 1.
    c_mask[:, :, 0] *= 2. # Fixation is important
    c_mask = c_mask.reshape((batch_dim[0]*batch_dim[1], batch_dim[2]))
    
    return c_mask
    

def custom_mse(y_true, y_hat):

    n_output = hp['n_output']
    y_true_shaped = tf.reshape(y_true, (-1, n_output))
    y_hat_shaped = tf.reshape(y_hat, (-1, n_output))
    cost_lsq = K.mean(K.square((y_true_shaped - y_hat_shaped) * c_mask_shaped_tf))
    
#     cost = self.cost_lsq + self.cost_reg
    cost = cost_lsq
    
    return cost

def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)

def custom_response_accuracy(y_true, y_hat):
    if type(y_true) is not np.ndarray:
        y_true = y_true.numpy()
    if type(y_hat) is not np.ndarray:
        y_hat = y_hat.numpy()
    y_hat_loc1 = popvec(np.mean(y_hat[:, 155:180, 1:], axis=1))
    y_true_loc1 = popvec(np.mean(y_true[:, 155:180, 1:], axis=1))
    original_dist1 = y_true_loc1 - y_hat_loc1
    dist1 = np.minimum(abs(original_dist1), 2*np.pi-abs(original_dist1))
    corr_loc1 = dist1 < 2*np.pi/hp['n_loc']

    y_hat_loc2 = popvec(np.mean(y_hat[:, 180:, 1:], axis=1))
    y_true_loc2 = popvec(np.mean(y_true[:, 180:, 1:], axis=1))
    original_dist2 = y_true_loc2 - y_hat_loc2
    dist2 = np.minimum(abs(original_dist2), 2*np.pi-abs(original_dist2))
    corr_loc2 = dist2 < 2*np.pi/hp['n_loc']
    
    return np.sum(corr_loc1*0.5+corr_loc2*0.5)/corr_loc1.shape[0]

def custom_response_loss(y_true, y_hat):
    if type(y_true) is not np.ndarray:
        y_true = y_true.numpy()
    if type(y_hat) is not np.ndarray:
        y_hat = y_hat.numpy()
    y_hat_loc1 = popvec(np.mean(y_hat[:, 155:180, 1:], axis=1))
    y_true_loc1 = popvec(np.mean(y_true[:, 155:180, 1:], axis=1))
    original_dist1 = y_true_loc1 - y_hat_loc1
    dist1 = np.minimum(abs(original_dist1), 2*np.pi-abs(original_dist1))

    y_hat_loc2 = popvec(np.mean(y_hat[:, 180:, 1:], axis=1))
    y_true_loc2 = popvec(np.mean(y_true[:, 180:, 1:], axis=1))
    original_dist2 = y_true_loc2 - y_hat_loc2
    dist2 = np.minimum(abs(original_dist2), 2*np.pi-abs(original_dist2))
    
    return np.sum(dist1*0.5+dist2*0.5)/dist1.shape[0]*(360/(2*np.pi))

def custom_perf(y_true, y_hat):

    if type(y_true) is not np.ndarray:
        y_true = y_true.numpy()
    if type(y_hat) is not np.ndarray:
        y_hat = y_hat.numpy()

    y_true = y_true[:,-1,:]
    y_hat = y_hat[:,-1,:]

    y_hat_loc = popvec(y_hat[..., 1:])
    y_true_loc = popvec(y_true[..., 1:])
    y_hat_fix = y_hat[..., 0]
    fixating = y_hat_fix > 0.5 

    original_dist = y_true_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 2*np.pi/hp['n_loc']


    # Should fixate?
    should_fix = y_true_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating) 
    return np.mean(perf)
# In[83]:

def generate_trial(mode='random',batch_size=hp['batch_size_train'],**kwargs):
    dt = hp['dt']
    if mode == 'random':
        rng = hp['rng']
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = rng.uniform(0, 2*np.pi, (batch_size,))

        stims_mean = rng.uniform(0.8,1.2,(batch_size,))
        stims_coh  = rng.choice([0.,0.08,0.16,0.32],(batch_size,))
        stims_sign = rng.choice([1,-1], (batch_size,))

        stim1_strengths = stims_mean + stims_coh*stims_sign
        stim2_strengths = stims_mean - stims_coh*stims_sign

    elif mode == 'fixed':
        n_loc = kwargs['n_loc']
        batch_size = n_loc*n_loc
        n_stim_loc1, n_stim_loc2, repeat = stim_loc_shape = n_loc, n_loc, 1
        stim_loc_size = np.prod(stim_loc_shape)
        ind_stim_loc1, ind_stim_loc2, ind_repeat = np.unravel_index(range(stim_loc_size),stim_loc_shape)
        stim1_locs = 2*np.pi*ind_stim_loc1/n_stim_loc1
        stim2_locs = 2*np.pi*ind_stim_loc2/n_stim_loc2
        
        stim1_strengths = 1
        stim2_strengths = 1
        
    else:
        raise ValueError('Unknown mode: ' + str(mode))
        
    stim1_ons = int(500/dt)
    stim1_offs = stim1_ons + int(300/dt)
    stim2_ons =stim1_offs + int(1000/dt)
    stim2_offs = stim2_ons + int(300/dt)
    fix_offs  = stim2_offs + int(1000/dt)
    output_2_on = fix_offs + int(500/dt)
    tdim = output_2_on + int(500/dt)

    check_ons = fix_offs + int(100/dt)

    trial = Trial(hp, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_strengths, mods=1)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_strengths, mods=1)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] for i in range(batch_size)]
    stim_locs2 = [stim2_locs[i] for i in range(batch_size)]
    trial.add('out',stim_locs,ons=fix_offs,offs=output_2_on)
    trial.add('out',stim_locs2,ons=output_2_on)

    trial.add_c_mask(pre_offs=fix_offs,post_ons=check_ons)
    trial.epochs = {'fix1':(None,stim1_ons),
                    'stim1':(stim1_ons,stim1_offs),
                    'delay1':(stim1_offs,stim2_ons),
                    'stim2':(stim2_ons,stim2_offs),
                    'delay2':(stim2_offs,fix_offs),
                    'go1':(fix_offs,output_2_on),
                    'go2':(output_2_on,None)}
    return trial

def train_generator():
    for i in range(hp['steps_per_epoch']*hp['n_epochs']):
        trial = generate_trial(mode='random',batch_size=hp['batch_size_train'])
        x = trial.x.swapaxes(0,1)
        y = trial.y.swapaxes(0,1)
        yield x, y

def val_generator():
    for i in range(hp['steps_per_epoch']*hp['n_epochs']):
        trial = generate_trial(mode='random',batch_size=hp['batch_size_val'])
        x = trial.x.swapaxes(0,1)
        y = trial.y.swapaxes(0,1)
        yield x, y


# In[84]:


def create_model(rnn_layer):
    model = Sequential()
    model.add(rnn_layer)
    model.add(TimeDistributed(Dense(33, activation='sigmoid')))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']),
        loss=custom_mse,
        metrics=[custom_perf, custom_response_accuracy,custom_response_loss],
        run_eagerly=True)
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']),
    #     loss=custom_mse,
    #     metrics=[custom_perf],
    #     run_eagerly=True)

    return model


# In[85]:


def generate_hidden_layer_plots(rnn_layer, superscript):
    hidden = rnn_layer(x)

    for delay in range(1,3):
        print('delay:%s'%delay)
        delay_bins = get_delay_bins(delay=delay)
        #extract mean firing rates for delay bins
        delay_hidden = np.mean(hidden[:, delay_bins[0]:delay_bins[1], :], axis=1)
    #     delay_hidden = hidden[:, delay_bins[1], :]
        plot_tuning_curves(delay_hidden,tuning_curve_folder,'%s_delay%d_hidden'%(superscript,delay))

        proj = fit_isomap(data_to_use=delay_hidden)
        plot_all_isomap_figures(proj,isomap_folder,'%s_delay%d_hidden'%(superscript,delay))

def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng', None)
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)

class NBatchLogger(tf.keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}
        self.t_start = time.time()
        

    def on_batch_end(self, batch, logs={}):
        keys = list(logs.keys())
        for k in keys:
            self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('time: {} | trial: {} | batch: {} ... {}'.format(time.time()-self.t_start, self.step * hp['batch_size_train'], self.step,
                                          metrics_log))
            self.metric_cache.clear()
        self.step += 1

class AccuracyThresholdCallback(tf.keras.callbacks.Callback): 
    def on_batch_end(self, batch, logs={}):
        acc = logs.get('custom_response_accuracy')
        if(acc > hp['accuracy_threshold']):
            print("\nReached %2.2f%% accuracy!" %(acc*100))   
            self.model.stop_training = True

class printeverybatch(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        tf.print('train batch:')
        tf.print(x[0,90:105,0])
        tf.print(y[0,180:,0])
        tf.print()
        tf.print(x[0,90:105,1])
        tf.print(y[0,180:,1])
        return super().train_step(data)

    def test_step(self, data):
        x, y = data
        tf.print('val batch:')
        tf.print(x[0,90:105,0])
        tf.print(y[0,180:,0])
        tf.print()
        tf.print(x[0,90:105,1])
        tf.print(y[0,180:,1])
        return super().test_step(data)

def get_diag_meas(block_dist):

    d = block_dist.shape[0]
    j = np.ones(d)
    r = np.array(range(1,d+1))
    r2 = np.array(list(map(lambda x: x*x, range(1,d+1))))

    block_dist = block_dist + np.abs(np.min(block_dist))
    highest = np.nanmax(np.abs(block_dist))
    block_dist = block_dist/highest

    n = j.dot(block_dist.dot(j.T))
    sum_x = r.dot(block_dist.dot(j.T))
    sum_y = j.dot(block_dist.dot(r.T))
    sum_x2 = r2.dot(block_dist.dot(j.T))
    sum_y2 = j.dot(block_dist.dot(r2.T))
    sum_xy = r.dot(block_dist.dot(r.T))

    corr = (n*sum_xy - sum_x*sum_y)/((math.sqrt(n*sum_x2 - (sum_x)**2))*(math.sqrt(n*sum_y2 - (sum_y)**2)))
    return corr

def plot_weight_selectivity_matrix(target,Z,W,n_loc,foldername,filename,stim,delay):
    
    # mean firing rate of each neuron for each location. shape: (n_rnn, n_loc)
    mean_delay_FR = np.array([[np.mean(Z[target==i,j]) for i in range(n_loc)] for j in range(Z.shape[1])])
    
    # the most sensitive location (location with highest mean firing rate) of each neuron. shape: n_rnn
    label = np.argmax(mean_delay_FR,1)
    
    # indices of neurons, sorted according to most sensitive location. shape: n_rnn
    ind = np.argsort(label)
    
    # weight matrix sorted according to most sensitive location. shape: (n_rnn, n_rnn)
    sorted_weight_matrix = W[:,ind]
    sorted_weight_matrix = sorted_weight_matrix[ind,:]
    
    # block matrix, mean weights connecting neurons most sensitive to 1 loc to neurons most sensitive to another loc. shape: (n_loc, n_loc)
    block_dist = np.zeros([n_loc,n_loc])
    for i in range(n_loc):
        for j in range(n_loc):
            from_neurons = np.where(label[ind] == i)[0]
            to_neurons = np.where(label[ind] == j)[0]
            block_dist[i,j] = np.nanmean(sorted_weight_matrix[from_neurons,:][:,to_neurons])

    highest = np.nanmax(np.abs(block_dist))
    fig = plt.figure()
    diag=get_diag_meas(block_dist)
    plt.title(f'Weight Selectivity Matrix Stim {stim} Delay {delay}')
    plt.legend(loc='upper left', bbox_to_anchor= (1.3, 1.03),
               title=f'diag: {round(diag,3)}')
    sns.heatmap(block_dist,cmap='bwr',vmin=-highest, vmax=highest, )

    if filename != '':
        fig.savefig(foldername+filename,bbox_inches='tight')
    return diag

def plot_activity_heatmap(A, neuron, separability_index, linear_mixed_selectivity_index, foldername='', filename=''):
    fig, ax = plt.subplots(1,1)
    sns.heatmap(A)
    ax.set_xlabel('Stim2')
    ax.set_ylabel('Stim1')
    ax.set_title(f'Activity Heatmap (Neuron {neuron})')
    plt.legend(loc='upper left', bbox_to_anchor= (1.3, 1.03),
               title=f'linear separability: {round(separability_index,3)}\nlinear selectivity: {round(linear_mixed_selectivity_index,3)}')
    if filename != '':
        fig.savefig(foldername+filename,bbox_inches='tight')
    plt.show()
    plt.close()

def get_separability_index(A):
    U, s, VT = svd(A)
    sum_of_squares_s = np.array([s[i]**2 for i in range(len(s))]).sum()
    separability_index = s[0]**2/sum_of_squares_s
    return separability_index

def get_linear_mixed_selectivity_index(aov):
    linV = aov.iloc[0,1]+aov.iloc[1,1]
    nonlinV = aov.iloc[2,1]
    linear_mixed_selectivity_index = (linV)/(nonlinV+linV)
    return linear_mixed_selectivity_index

n_rnn = 256
hp['n_rnn']=n_rnn
hp['sigma_rec']=0
hp['sigma_x']=0


# generate 128*128 input
trial = generate_trial(mode='fixed',n_loc=hp['n_loc'])
x = trial.x.swapaxes(0,1)
print('x shape: ' + str(x.shape))
stim_loc_shape = hp['n_loc'],hp['n_loc'],1
stim_loc_size = np.prod(stim_loc_shape)
ind_stim_loc1, ind_stim_loc2, ind_repeat = np.unravel_index(range(stim_loc_size),stim_loc_shape)

# generate 8*8 input
n_loc=8
trial_ = generate_trial(mode='fixed',n_loc=n_loc)
x_ = trial_.x.swapaxes(0,1)
print('x_ shape: ' + str(x_.shape))
stim_loc_shape_ = n_loc,n_loc,1
stim_loc_size_ = np.prod(stim_loc_shape_)
ind_stim_loc1_, ind_stim_loc2_, ind_repeat_ = np.unravel_index(range(stim_loc_size_),stim_loc_shape_)

# generate 10 noisy samples for each 8*8 input (for anova)
X_n = []
for i in range(x_.shape[0]):
    for j in range(10):
        X_n.append(x_[i] + np.random.RandomState(seed).randn(205, 33)*0.01)
X_n = np.array(X_n)
print('X_n shape: ' + str(X_n.shape))

        
for acc in [95,80,60]:
    df_summary = pd.DataFrame()
    df_indices = []
    for seed in range(1,21):

        tf.random.set_seed(seed)
        hp['seed']=seed
        hp['rng'] = np.random.RandomState(seed)

        model_folder = '/hpctmp/e0309561/2_stim_batch_size_512/n_hidden_128/2_stim_batch_size_512_n_hidden_128_acc_%s_seed_%s_with_noise'%(acc,seed)
        print('model folder: ' + model_folder)
        checkpoint_path = model_folder+"/checkpoint/cp.ckpt"
        isomap_folder = model_folder+'/isomap'
        tuning_curve_folder = model_folder+'/tuning_curves/'
        weight_matrix_folder = model_folder+'/weight_matrices/'

        if not os.path.exists(weight_matrix_folder):
            os.makedirs(weight_matrix_folder)
            
        cell = LeakyRNNCell2(hp)
        rnn_layer = tf.keras.layers.RNN(cell,input_shape=((x.shape[1],x.shape[2])),
                                        return_sequences=True)
        model = create_model(rnn_layer)
        model.load_weights(checkpoint_path)
        rnn_layer = tf.keras.layers.RNN(cell,input_shape=((x.shape[1],x.shape[2])),
                                        return_sequences=True, weights=model.layers[0].get_weights())
        weights=model.layers[0].get_weights()
        W = weights[1]

        print('generating_hidden layer')
        hidden = rnn_layer(x) #for linear separability
        hidden_ = rnn_layer(x_)
        hidden_n = rnn_layer(X_n) #for anova linear/ non-linear mixed selectivity
        print('done')

        results_dict = {}
        for delay in range(1,3):
            print('delay:%s'%delay)
            delay_bins = get_delay_bins(delay=delay)

            #extract mean firing rates for delay bins
            delay_hidden = np.mean(hidden[:, delay_bins[0]:delay_bins[1], :], axis=1)
            delay_hidden_ = np.mean(hidden_[:, delay_bins[0]:delay_bins[1], :], axis=1)
            delay_hidden_n = np.mean(hidden_n[:, delay_bins[0]:delay_bins[1], :], axis=1)

            #get weight matrices
            ind_stim_locs = {1:ind_stim_loc1_, 2:ind_stim_loc2_}
            for stim in [1,2]:
                weight_matrix_file = f'delay_{delay}_stim_{stim}_weight_matrix.png'
                diag = plot_weight_selectivity_matrix(ind_stim_locs[stim],delay_hidden_,W,n_loc,weight_matrix_folder,weight_matrix_file,stim,delay)
                results_dict[f'delay {delay} diag {stim}'] = diag

            #get separability and selectivity indices
            separability_index_list = []
            linear_mixed_selectivity_index_list = []
            for neuron in range(256):
                A = delay_hidden[:,neuron].reshape(hp['n_loc'],hp['n_loc'])
                separability_index = get_separability_index(A)
                separability_index_list.append(separability_index)

                Z = delay_hidden_n[:,neuron]
                df = pd.DataFrame({'first_stim':np.array([[i]*10 for i in ind_stim_loc1_]).flatten(),'second_stim':np.array([[i]*10 for i in ind_stim_loc2_]).flatten(),'activity':Z})
                aov = pg.anova(dv='activity', between=['first_stim', 'second_stim'], data=df,
                         detailed=True)
                linear_mixed_selectivity_index = get_linear_mixed_selectivity_index(aov)
                linear_mixed_selectivity_index_list.append(linear_mixed_selectivity_index)

                #activity heatmap
                plot_activity_heatmap(A,neuron, separability_index, linear_mixed_selectivity_index,tuning_curve_folder,f'delay_{delay}_heatmap_neuron_{neuron}')

            df = pd.DataFrame(np.array([separability_index_list,linear_mixed_selectivity_index_list]).T, columns=['linear separability','linear selectivity'])
            df.to_csv(tuning_curve_folder+f'delay_{delay}_separability_and_selectivity.csv')
            df = df.dropna()

            results_dict[f'delay {delay} separability mean'] = df['linear separability'].mean()
            results_dict[f'delay {delay} separability std'] = df['linear separability'].std()
            results_dict[f'delay {delay} selectivity mean'] = df['linear selectivity'].mean()
            results_dict[f'delay {delay} selectivity std'] = df['linear separability'].std()

            fig, ax = plt.subplots()
            plt.plot(df['linear separability'], label="Linear Separability Index")
            plt.plot(df['linear selectivity'], label="Linear Mixed Selectivity Index")
            correlation = df['linear selectivity'].corr(df['linear separability'])
            plt.legend(loc='upper left', bbox_to_anchor= (1.05, 1.03), title=f'Correlation = {round(correlation,3)}')
            ax.set_xlabel('Neurons')
            ax.set_ylabel('Index')
            ax.set_title('Linear Separability and Linear Mixed Selectivity of Neurons')
            fig.savefig(tuning_curve_folder+f'delay_{delay}_separability_and_selectivity_plot.png',bbox_inches='tight')

            results_dict[f'delay {delay} selectivity/separability correlation'] = correlation
            with open(os.path.join(model_folder, 'results.json'), 'w') as f:
                json.dump(results_dict, f)

        df_summary = df_summary.append(results_dict, ignore_index=True)
        df_indices.append(seed)
    df_summary.index = df_indices
    df_summary.to_csv(f'{os.path.dirname(model_folder)}/summary_acc_{acc}.csv')

