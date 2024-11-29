#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:46:44 2021

@author: ari
"""

import os
import numpy as np
import math
import time
import re
import sys
os.environ["MODEL_CNN"] = "NN_WallRecon";
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE";
#%% Configuration import
import config_deep as config
from vit_keras import vit, utils
import tensorflow as tf
from tensorflow.keras import layers, Model
os.environ["MODEL_CNN"] = "NN_WallRecon";
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE";
from tensorflow.keras.layers import LayerNormalization, BatchNormalization
prb_def = os.environ.get('MODEL_CNN', None)

if  prb_def == 'NN_WallRecon':
    app = config.NN_WallRecon
    print('Newtonian Wall-Recon is used')
else:
    raise ValueError('"MODEL_CNN" enviroment variable must be defined as "NN_WallRecon". Otherwise, use different train script.')

os.environ["CUDA_VISIBLE_DEVICES"]=str(app.WHICH_GPU_TEST);


#%% Tensorflow imports

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, \
                                       ModelCheckpoint, LearningRateScheduler
#device_name = tf.test.gpu_device_name()
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 
print('Using TensorFlow version:', tf.__version__, ', GPU:', availale_GPUs)

if physical_devices:
  try:
    for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

on_GPU = app.ON_GPU
n_gpus = app.N_GPU
initial_learning_rate = 1e-3
distributed_training = on_GPU == True and n_gpus>1

if distributed_training:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    print('Number of devices for distributed training: {}'.format(strategy.num_replicas_in_sync))

#%% Function definition

@tf.function
def periodic_padding(tensor, padding):
    """
    Tensorflow function to pad periodically a 2D tensor

    Parameters
    ----------
    tensor : 2D tf.Tensor
        Tensor to be padded
    padding : integer values
        Padding value, same in all directions

    Returns
    -------
    Padded tensor

    """
    lower_pad = tensor[:padding[0][0],:]
    upper_pad = tensor[-padding[0][1]:,:]
    
    partial_tensor = tf.concat([upper_pad, tensor, lower_pad], axis=0)
    
    left_pad = partial_tensor[:,-padding[1][0]:]
    right_pad = partial_tensor[:,:padding[1][1]]
    
    padded_tensor = tf.concat([left_pad, partial_tensor, right_pad], axis=1)
    
    return padded_tensor
    
@tf.function
def periodic_padding_z(tensor, padding):
    """
    Tensorflow function to pad periodically a 2D tensor

    Parameters
    ----------
    tensor : 2D tf.Tensor
        Tensor to be padded
    padding : integer values
        Padding value, same in all directions

    Returns
    -------
    Padded tensor

    """
    lower_pad = tensor[:padding[0][0],:]
    upper_pad = tensor[-padding[0][1]:,:]
    
    padded_tensor = tf.concat([upper_pad, tensor, lower_pad], axis=0)
    
    return padded_tensor

@tf.function
def output_parser(rec):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''        
    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),
        'nx': tf.io.FixedLenFeature([], tf.int64),
        'ny': tf.io.FixedLenFeature([], tf.int64),
        'nz': tf.io.FixedLenFeature([], tf.int64),
        'comp_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
#        'comp_out_raw4': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
        
    parsed_rec = tf.io.parse_single_example(rec, features)
    
    # i_sample = parsed_rec['i_sample']
    nx = tf.cast(parsed_rec['nx'], tf.int32)
    # ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = tf.cast(parsed_rec['nz'], tf.int32)
    nx = 432
    nz = 432
#    print('nx:'+str(nx))
#    print('nz:'+str(nz))

    output1 = tf.reshape(parsed_rec['comp_out_raw1'],(1,nz, nx))

    if app.N_VARS_OUT == 1:
        return output1
    else:
        output2 = tf.reshape(parsed_rec['comp_out_raw2'],(1,nz, nx))
        if app.N_VARS_OUT == 2:
            return (output1, output2)
        else:
            output3 = tf.reshape(parsed_rec['comp_out_raw3'],(1,nz, nx))
            return (output1, output2, output3)

@tf.function
def input_parser(rec):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''        
    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),
        'nx': tf.io.FixedLenFeature([], tf.int64),
        'ny': tf.io.FixedLenFeature([], tf.int64),
        'nz': tf.io.FixedLenFeature([], tf.int64),
        'comp_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
#        'comp_raw4': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
    
    parsed_rec = tf.io.parse_single_example(rec, features)
    
    # i_sample = parsed_rec['i_sample']
    nx = tf.cast(parsed_rec['nx'], tf.int32)
    # ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = tf.cast(parsed_rec['nz'], tf.int32)
    nx = 432
    nz = 432
    padding = tf.cast(pad/2, tf.int32)

    nxd = nx + pad
    nzd = nz + pad
    
#    # Input processing --------------------------------------------------------
    inputs = periodic_padding(tf.reshape(parsed_rec['comp_raw1'],(nz, nx)),((padding,padding),(padding,padding)))
    inputs = tf.reshape(inputs,(1,nzd,nxd))

    for i_comp in range(1,app.N_VARS_IN):
        new_input = tf.reshape(parsed_rec[f'comp_raw{i_comp+1}'],(nz,nx))
        inputs = tf.concat((inputs, tf.reshape(periodic_padding(new_input,((padding,padding),(padding,padding))),(1,nzd,nxd))),0)
        
    return inputs

@tf.function
def eval_parser(rec):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''        
    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),
        'nx': tf.io.FixedLenFeature([], tf.int64),
        'ny': tf.io.FixedLenFeature([], tf.int64),
        'nz': tf.io.FixedLenFeature([], tf.int64),
        'comp_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
    
    parsed_rec = tf.io.parse_single_example(rec, features)
    
    # i_sample = parsed_rec['i_sample']
    nx = tf.cast(parsed_rec['nx'], tf.int32)
    # ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = tf.cast(parsed_rec['nz'], tf.int32)
    
#    print('nx:'+str(nx))
#    print('nz:'+str(nz))
    
    padding = tf.cast(pad/2, tf.int32)

#    nxd = nx + pad
    nzd = nz + pad
    
    # Input processing --------------------------------------------------------
    if norm_input == True:
        inputs = periodic_padding(tf.reshape((parsed_rec['comp_raw1']-avgs_in[0])/std_in[0],(nx, nz)),((padding,padding),(padding,padding)))
    else:
        inputs = periodic_padding(tf.reshape(parsed_rec['comp_raw1'],(nx, nz)),((padding,padding),(padding,padding)))
    inputs = tf.reshape(inputs,(1,nxd,nzd))
    
    for i_comp in range(1,app.N_VARS_IN):
        new_input = parsed_rec[f'comp_raw{i_comp+1}']
        if norm_input == True:
            new_input = (new_input-avgs_in[i_comp])/std_in[i_comp]
        inputs = tf.concat((inputs, tf.reshape(periodic_padding(tf.reshape(new_input,(nx, nz)),((padding,padding),(padding,padding))),(1,nxd,nzd))),0)
    
    # Output processing
    nx_out = nx
    nz_out = nz
    
    output1 = tf.reshape(parsed_rec['comp_out_raw1'],(1,nx_out, nz_out))
    
    if pred_fluct == True:    
        output1 = output1 - avgs[0][ypos_Ret[str(target_yp)]]
    
    if app.N_VARS_OUT == 1:
        return inputs, output1
    else:
        output2 = tf.reshape(parsed_rec['comp_out_raw2'],(1,nx_out, nz_out))
        if pred_fluct == True:    
            output2 = output2 - avgs[1][ypos_Ret[str(target_yp)]]
        
        if scale_output == True:
            scaling_coeff2 = tf.cast(rms[0][ypos_Ret[str(target_yp)]] / rms[1][ypos_Ret[str(target_yp)]], tf.float32)
            output2 = output2 * scaling_coeff2
        if app.N_VARS_OUT == 2:
            return inputs, (output1, output2)
        else:
            output3 = tf.reshape(parsed_rec['comp_out_raw3'],(1,nx_out, nz_out))
            if pred_fluct == True:    
                output3 = output3 - avgs[2][ypos_Ret[str(target_yp)]]
            
            if scale_output == True:
                scaling_coeff3 = tf.cast(rms[0][ypos_Ret[str(target_yp)]] / rms[2][ypos_Ret[str(target_yp)]], tf.float32)
                output3 = output3 * scaling_coeff3
            return inputs, (output1, output2, output3)

#%% Functions for the NN
def vit_model():
    input_shape = (app.N_VARS_IN, nz, nx)
    pred_fluct = app.FLUCTUATIONS_PRED
    padding = 'same'

    inputs = layers.Input(shape=input_shape, name='input_data')

    x = layers.Permute((2, 3, 1))(inputs)  # (batch_size, channels nz, nx, )

    image_size_vit = 224
    x = layers.Lambda(lambda image: tf.image.resize(image, (image_size_vit, image_size_vit)))(x)

    
    if app.N_VARS_IN != 3:
        x = layers.Conv2D(3, (1, 1), padding='same')(x)

    vit_model_base = vit.vit_b16(
        image_size=image_size_vit,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    vit_model_base.trainable = True

    vit_inputs = vit_model_base.input
    vit_outputs = vit_model_base.get_layer('Transformer/encoder_norm').output  
    vit_submodel = Model(inputs=vit_inputs, outputs=vit_outputs)

    x = vit_submodel(x)  # 出力形状: (batch_size, num_patches + 1, hidden_size)

    x = layers.Lambda(lambda v: v[:, 1:, :])(x)  # (batch_size, num_patches, hidden_size)

    num_patches = x.shape[1]
    patch_dim = int(num_patches ** 0.5)
    x = layers.Reshape((patch_dim, patch_dim, -1))(x)  # (batch_size, patch_dim, patch_dim, hidden_size)

    x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding=padding, activation='relu')(x)
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding=padding, activation='relu')(x)
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding=padding, activation='relu')(x)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding=padding, activation='relu')(x)

    x = layers.Lambda(lambda image: tf.image.resize(image, (nz_, nx_)))(x)
    act_30 = layers.Permute((3, 1, 2))(x)  # (batch_size, channels, nz_, nx_)

    
    outputs_model = []
    losses = {}

    # Branch 1
    cnv_b1 = layers.Conv2D(1, (3, 3), padding=padding, data_format='channels_first')(act_30)
    if pred_fluct:
        act_b1 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b1')(cnv_b1)
    else:
        act_b1 = layers.Activation('relu')(cnv_b1)
    output_b1 = act_b1  # Cropping2D は不要
    outputs_model.append(output_b1)
    losses['output_b1'] = 'mse'

    # Branch 2
    cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding, data_format='channels_first')(act_30)
    if pred_fluct:
        act_b2 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b2')(cnv_b2)
    else:
        act_b2 = layers.Activation('relu')(cnv_b2)
    output_b2 = act_b2
    outputs_model.append(output_b2)
    losses['output_b2'] = 'mse'

    # Branch 3
    cnv_b3 = layers.Conv2D(1, (3, 3), padding=padding, data_format='channels_first')(act_30)
    if pred_fluct:
        act_b3 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b3')(cnv_b3)
    else:
        act_b3 = layers.Activation('relu')(cnv_b3)
    output_b3 = act_b3
    outputs_model.append(output_b3)
    losses['output_b3'] = 'mse'

    ViT_model = Model(inputs=inputs, outputs=outputs_model)
    return ViT_model, losses
'''
def vit_model():
    input_shape = (app.N_VARS_IN, nz, nx)
    pred_fluct = app.FLUCTUATIONS_PRED
    padding = 'same'

    inputs = layers.Input(shape=input_shape, name='input_data')

    x = layers.Permute((2, 3, 1))(inputs)  # (batch_size, channels nz, nx, )

    image_size_vit = 224
    x = layers.Lambda(lambda image: tf.image.resize(image, (image_size_vit, image_size_vit)))(x)

    
    if app.N_VARS_IN != 3:
        x = layers.Conv2D(3, (1, 1), padding='same')(x)

    vit_model_base = vit.vit_b16(
        image_size=image_size_vit,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    vit_model_base.trainable = True

    vit_inputs = vit_model_base.input
    vit_outputs = vit_model_base.get_layer('Transformer/encoder_norm').output  
    vit_submodel = Model(inputs=vit_inputs, outputs=vit_outputs)

    x = vit_submodel(x)  # 出力形状: (batch_size, num_patches + 1, hidden_size)

    x = layers.Lambda(lambda v: v[:, 1:, :])(x)  # (batch_size, num_patches, hidden_size)

    num_patches = x.shape[1]
    patch_dim = int(num_patches ** 0.5)
    x = layers.Reshape((patch_dim, patch_dim, -1))(x)  # (batch_size, patch_dim, patch_dim, hidden_size)

    x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding=padding, activation='relu')(x)
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding=padding, activation='relu')(x)
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding=padding, activation='relu')(x)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding=padding, activation='relu')(x)

    x = layers.Lambda(lambda image: tf.image.resize(image, (nz_, nx_)))(x)
    act_30 = layers.Permute((3, 1, 2))(x)  # (batch_size, channels, nz_, nx_)

    
    outputs_model = []
    losses = {}

    # Branch 1
    cnv_b1 = layers.Conv2D(1, (3, 3), padding=padding, data_format='channels_first')(act_30)
    if pred_fluct:
        act_b1 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b1')(cnv_b1)
    else:
        act_b1 = layers.Activation('relu')(cnv_b1)
    output_b1 = act_b1  # Cropping2D は不要
    outputs_model.append(output_b1)
    losses['output_b1'] = 'mse'

    # Branch 2
    cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding, data_format='channels_first')(act_30)
    if pred_fluct:
        act_b2 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b2')(cnv_b2)
    else:
        act_b2 = layers.Activation('relu')(cnv_b2)
    output_b2 = act_b2
    outputs_model.append(output_b2)
    losses['output_b2'] = 'mse'

    # Branch 3
    cnv_b3 = layers.Conv2D(1, (3, 3), padding=padding, data_format='channels_first')(act_30)
    if pred_fluct:
        act_b3 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b3')(cnv_b3)
    else:
        act_b3 = layers.Activation('relu')(cnv_b3)
    output_b3 = act_b3
    outputs_model.append(output_b3)
    losses['output_b3'] = 'mse'

    ViT_model = Model(inputs=inputs, outputs=outputs_model)
    return ViT_model, losses
'''
def step_decay_schedule(epoch, initial_lr=1e-3, drop=0.5, epochs_drop=10):
    """
    Step decay function for learning rate scheduling.
    """
    lrate = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

# Define your parameters

lrate_scheduler = LearningRateScheduler(lambda epoch: step_decay_schedule(epoch, initial_learning_rate, drop, epochs_drop))

# Final ReLu function for fluctuations

#%% Reading configuration

cur_path = app.CUR_PATH
ds_path = app.DS_PATH_TEST
# Average profiles folder
avg_path = ds_path +'/.avg/'

train_yp = app.TRAIN_YP
target_yp = app.TARGET_YP
if not(type(target_yp) is int):
    target_yp = target_yp[0]

n_samples = np.array(app.N_SAMPLES_TEST)
n_samples_tot = np.sum(n_samples)
n_files_test = n_samples.shape[0]

interv = app.INTERV_TEST
tfr_path = ds_path+f'/.tfrecords_singlefile_test_dt{int(11.57*100*interv)}_f32/'
# epochs = app.N_EPOCHS
batch_size = app.BATCH_SIZE
if distributed_training:
    print('WARNING: The provided batch size is used in each device of the distributed training')
    batch_size *= strategy.num_replicas_in_sync
# validation_split = app.VAL_SPLIT
# Learning rate config
init_lr = app.INIT_LR
# lr_drop = app.LR_DROP
# lr_epdrop = app.LR_EPDROP

if app.NET_MODEL == 1:
    pad = tf.constant(64)
    pad_out = 2
    padding_in = 64
    padding_out = 0
else:
    pad = tf.constant(0)
    raise ValueError('NET_MODEL = 1 is the only one implentated so far')

#%% Settings for TFRecords
tfr_files_output = [os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))]

regex = re.compile(f'yp{target_yp:03d}')
regex_tr = re.compile(f'test')
regex_t = re.compile(f'yp{train_yp:03d}')
regex_tb = re.compile('Ret180')
regex_p = re.compile('velocityn25')
regex_q = re.compile('001-of-002')
print(tfr_files_output,tfr_path)
tfr_files_output = [string for string in tfr_files_output if re.search(regex,string) and re.search(regex_tr,string) and re.search(regex_tb,string) and re.search(regex_t,string) and re.search(regex_p,string) and re.search(regex_q,string)]
tfr_files_output = [string for string in tfr_files_output if int((string.split('_')[-3].split('-')[-1])[2:]) == target_yp and int((string.split('_')[-3].split('-')[0])[2:]) == train_yp]

tfr_files_output = [string for string in tfr_files_output if int(string.split('_')[-2][4:7])<n_files_test]

tfr_files_output = sorted(tfr_files_output)

Ret = (tfr_files_output[0].split('/')[-1]).split('_')[0][3:]

(nx_, ny_, nz_) = [int(val) for val in tfr_files_output[0].split('/')[-1].split('_')[1].split('x')]
nx_ = 432
nz_ = 432

#print('nx_in:'+str(nx_in))
#print('nz_in:'+str(nz_in))
#print('nx_out:'+str(nx_out))
#print('nz_out:'+str(nz_out))

# Dictionary for the statistics files from Simson
ypos_Ret180_576 = {'0':0, '1':1, '15':2, '30':3, '50':4, '310':5, '330':6, '345':7, '359':8, '360':9}
#print('WARNING: the y+ indices are computed only at Re_tau = 180')
print(Ret, ny_)
if Ret == str(180):
    if ny_ == 576:
        ypos_Ret = ypos_Ret180_576
    else:
        raise ValueError('Wall-normal resolution not supported')

# Check whether we are predicting the fluctuations
try:
    pred_fluct = app.FLUCTUATIONS_PRED
    if not(str(target_yp) in ypos_Ret):
        raise ValueError("The selected target does not have a corresponding y-index in simulation")
except NameError:
    print('Setting the prediction to full flow fields (default value)')
    pred_fluct = False

# Check whether inputs are normalized as input Gaussian
try:
    norm_input = app.NORMALIZE_INPUT
except NameError:
    norm_input = False

# Checking whether the outputs are scaled with the ratio of RMS values
try:
    scale_output = app.SCALE_OUTPUT
except NameError:
    scale_output = False

# Test-specific parameters
timestamp = app.TIMESTAMP

pred_fld = os.listdir(cur_path+'/.logs/')
print(pred_fld)
for fld in pred_fld:
    if timestamp in fld and fld.split("_")[-1] != "log":
        NAME = fld
        break

try:
    print('[MODEL]')
    print(NAME)
except NameError:
    print('WARNING: Model not found in the logs folder')

pred_fld = os.listdir(cur_path+'/.saved_models/')

for fld in pred_fld:
    if timestamp in fld:
        NAME = fld
        break
        
print(NAME)

nx = nx_ + padding_in
nz = nz_ + padding_in

input_shape = (app.N_VARS_IN, nz, nx)


# Loading the mean profile and the fluctuations intensity if needed
if pred_fluct == True:
    print('The model outputs are the velocity fluctuations')
    avgs = tf.reshape(tf.constant(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_OUT[0]+'.m').astype(np.float32)[:]),(1,8))
    for i in range(1,app.N_VARS_OUT):
        avgs = tf.concat((avgs, tf.reshape(tf.constant(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_OUT[i]+'.m').astype(np.float32)[:]),(1,8))),0)

if scale_output == True:
    rms = tf.reshape(tf.constant(np.loadtxt(avg_path+app.VARS_NAME_OUT[0]+'_rms.m').astype(np.float32)[:]),(1,8))
    if prb_def == 'WallRecon':
        print('The outputs are scaled with the ratio of the RMS values, taking the first input as reference')
    for i in range(1,app.N_VARS_OUT):
        rms = tf.concat((rms, tf.reshape(tf.constant(np.loadtxt(avg_path+app.VARS_NAME_OUT[i]+'_rms.m').astype(np.float32)[:]),(1,8))),0)

if norm_input == True:
    print('The inputs are normalized to have a unit Gaussian distribution')
    avgs_in = tf.reshape(tf.constant(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_IN[0]+'.m').astype(np.float32)[:]),(1,8))
    for i in range(1,app.N_VARS_IN):
        avgs_in = tf.concat((avgs_in, tf.reshape(tf.constant(np.loadtxt(avg_path+'mean_'+app.VARS_NAME_IN[i]+'.m').astype(np.float32)[:]),(1,8))),0)

    rms_in = tf.reshape(tf.constant(np.loadtxt(avg_path+app.VARS_NAME_IN[0]+'_rms.m').astype(np.float32)[:]),(1,8))
    for i in range(1,app.N_VARS_IN):
        rms_in = tf.concat((rms_in, tf.reshape(tf.constant(np.loadtxt(avg_path+app.VARS_NAME_IN[i]+'_rms.m').astype(np.float32)[:]),(1,8))),0)
    std_in = rms_in

print('RMS')
print(std_in)

#%% Data preprocessing with tf.data.Dataset

tfr_files_output_test_ds = tf.data.Dataset.list_files(tfr_files_output, shuffle=False)

tfr_files_output_test_ds = tfr_files_output_test_ds.interleave(lambda x : tf.data.TFRecordDataset(x).take(tf.gather(n_samples, tf.strings.to_number(tf.strings.substr(tf.strings.split(\
                       x,sep='_')[-2],4,3),tf.int64))), cycle_length=1)

dataset_test = tfr_files_output_test_ds.map(output_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
X_test = tfr_files_output_test_ds.map(input_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# Datasets size check ---------------------------------------------------------
#itr = iter(dataset_test)
#j = 0
#for i in range(n_samples_tot):
#    example = next(itr)
#    j += 1
#
#try:
#    example = next(itr)
#except StopIteration:
#    print(f'Train set over: {j}')
#
#sys.exit(0)

# Datasets for evaluation -----------------------------------------------------
#dataset_eval = tfr_files_output_test_ds.map(eval_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#dataset_eval = dataset_eval.batch(batch_size=batch_size)

# Iterating over ground truth datasets ----------------------------------------
itr = iter(dataset_test)
itrX = iter(X_test)

print('Files used for testing')
for fl in tfr_files_output:
    print(fl)
print('')

X_test = np.ndarray((n_samples_tot,app.N_VARS_IN,nz_+padding_in,nx_+padding_in),dtype='float')
Y_test = np.ndarray((n_samples_tot,app.N_VARS_OUT,nz_,nx_),dtype='float')
ii = 0
for i in range(n_samples_tot):
    X_test[i] = next(itrX)
    if app.N_VARS_OUT == 1 :
        Y_test[i,0] = next(itr)
    elif app.N_VARS_OUT == 2 :
        (Y_test[i,0], Y_test[i,1]) = next(itr)
    else:
        (Y_test[i,0], Y_test[i,1], Y_test[i,2]) = next(itr)
    ii += 1
    # print(i+1)
print(f'Iterated over {ii} samples')
print('')
# sys.exit(0)
# Configuration summary -------------------------------------------------------

print('')
print('# ====================================================================')
print('#     Summary of the options for the model                            ')
print('# ====================================================================')
print('')
print(f'Number of samples for training: {int(n_samples_tot)}')
# print(f'Number of samples for validation: {int(n_samp_valid)}')
print(f'Total number of samples: {n_samples_tot}')
# print(f'Batch size: {batch_size}')
print('')
print(f'Data augmentation: {app.DATA_AUG} (not implemented in this model)')
print(f'Initial distribution of parameters: {app.INIT}')
if app.INIT == 'random':
    print('')
    print('')
if app.INIT == 'model':
    print(f'    Timestamp: {app.INIT_MODEL[-10]}')
    print(f'    Transfer learning: {app.TRANSFER_LEARNING} (not implemented in this model)')
print(f'Prediction of fluctuation only: {app.FLUCTUATIONS_PRED}')
print(f'y- and z-output scaling with the ratio of RMS values : {app.SCALE_OUTPUT}')
print(f'Normalized input: {app.NORMALIZE_INPUT}')
print('')
print('# ====================================================================')

# =============================================================================
#   Neural network loading 
# =============================================================================

# PREPARATION FOR SAVING THE RESULTS

pred_path = cur_path+'/.predictions/'
if not os.path.exists(pred_path):
    os.mkdir(pred_path)

#pred_path = cur_path+'/CNN-'+timestamp+'-ckpt/'
pred_path = pred_path+NAME+'/'
if not os.path.exists(pred_path):
    os.mkdir(pred_path)

# Loading model trained
if app.FROM_CKPT == True:
    model_path = cur_path+'/.logs/'+NAME+'/'
    ckpt = app.CKPT
    init_model = tf.keras.models.load_model(
            model_path+f'model.ckpt.{ckpt:04d}.hdf5'
            )
    print('[MODEL LOADING]')
    print('Loading model '+str(app.NET_MODEL)+' from checkpoint '+str(ckpt))    
    pred_path = pred_path+f'ckpt_{ckpt:04d}/'
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
else:
    model_path = cur_path+'/.saved_models/'
    init_model = tf.keras.models.load_model(
            model_path+NAME
            )
    print('[MODEL LOADING]')
    print('Loading model '+str(app.NET_MODEL)+' from saved model')
    pred_path = pred_path+'saved_model/'
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)

# If distributed training is used, we need to load only the weights

if distributed_training:
   padding = 'valid'

   print('Compiling and training the model for multiple GPU')

   with strategy.scope():

       model, losses = vit_model()
       print("vit is surely working")
       model.compile(loss='mse',
                     optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr))

           
       init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
       model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
       os.remove('/tmp/model_weights-CNN_keras_model.h5')
       del init_model
           

else:
    model = init_model

    model.compile(loss=['mse','mse','mse'],
                     optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate))

        
print(model.summary())

#%% Model evaluation
#print('Evaluating model performance')
#loss_values = CNN_model.evaluate(dataset_eval, batch_size=None)
#
#sys.exit(0)

#%% 
Y_pred = np.ndarray((n_samples_tot,app.N_VARS_OUT,nz_,nx_),dtype='float')

if app.N_VARS_OUT == 1:
    Y_pred[:,0,np.newaxis] = model.predict(X_test, batch_size=batch_size)
if app.N_VARS_OUT == 2:
    (Y_pred[:,0,np.newaxis], Y_pred[:,1,np.newaxis]) = model.predict(X_test, batch_size=batch_size)
if app.N_VARS_OUT == 3:
    (Y_pred[:,0,np.newaxis], Y_pred[:,1,np.newaxis], Y_pred[:,2,np.newaxis]) = model.predict(X_test, batch_size=batch_size)

print(type(Y_pred))
print(np.shape(Y_pred))

# Revert back to the flow field
# if scale_output == True:
#    for i in range(app.N_VARS_OUT):
#        print('Rescale back component '+str(i))
#    Y_pred[:,1] *= (1/17.785)
#    Y_pred[:,2] *= (1/5.5428)

# if pred_fluct == True:
#     for i in range(app.N_VARS_OUT):
#         print('Adding back mean of the component '+str(i))
#         Y_pred[:,i] = Y_pred[:,i] + avgs[i][ypos_Ret[str(target_yp)]]
#         Y_test[:,i] = Y_test[:,i] + avgs[i][ypos_Ret[str(target_yp)]]

print(Y_pred.shape)

i_set_pred = 0
while os.path.exists(pred_path+f'pred_fluct{i_set_pred:04d}.npz'):
    i_set_pred = i_set_pred + 1
print('[SAVING PREDICTIONS]')
print('Saving predictions in '+f'pred_fluct{i_set_pred:04d}')    
np.savez(pred_path+f'pred_fluct{i_set_pred:04d}', Y_test=Y_test, Y_pred=Y_pred,X_test = X_test)