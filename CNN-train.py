#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 1 12:14:06 2021

@author: ari
"""

import os
# import scipy.io as sio
import numpy as np
import math
import time
import re
import sys

os.environ["MODEL_CNN"] = "NN_WallRecon";
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE";
#%% Configuration import
import config_deep as config

prb_def = os.environ.get('MODEL_CNN', None)

if  prb_def == 'NN_WallRecon':
    app = config.NN_WallRecon
    print('Newtonian Wall-Recon is used')
else:
    raise ValueError('"MODEL_CNN" enviroment variable must be defined as "NN_WallRecon". Otherwise, use different train script.')

# =============================================================================
#   IMPLEMENTATION WARNINGS
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3";
# Data augmentation not implemented in this model for now
app.DATA_AUG = False
# Transfer learning not implemented in this model for now
app.TRANSFER_LEARNING = False

#%% Tensorflow imports

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, \
                                       ModelCheckpoint, LearningRateScheduler

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

#%% Class definition

# Credit to Martin Holub for the Class definition
class SubTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(SubTensorBoard, self).__init__(*args, **kwargs)

    def lr_getter(self):
        # Get learning rate from the optimizer
        lr = self.model.optimizer.learning_rate
        iters = self.model.optimizer.iterations
        beta_1 = self.model.optimizer.beta_1
        beta_2 = self.model.optimizer.beta_2
        # Calculate the learning rate
        t = tf.cast(iters, tf.float32) + 1
        lr_t = lr * (tf.math.sqrt(1. - tf.math.pow(beta_2, t)) / (1. - tf.math.pow(beta_1, t)))
        return np.float32(lr_t)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        logs.update({'learning_rate': self.lr_getter()})
        super(SubTensorBoard, self).on_epoch_end(epoch, logs)


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
def parser(rec):
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
#        'comp_out_raw4': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
    
    parsed_rec = tf.io.parse_single_example(rec, features)
    
    # i_sample = parsed_rec['i_sample']
    nx = 432 #tf.cast(parsed_rec['nx'], tf.int32)
    # ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = 432 #tf.cast(parsed_rec['nz'], tf.int32)
    
#    print('nx:'+str(nx))
#    print('nz:'+str(nz))
    
    padding = tf.cast(pad/2, tf.int32)

    nxd = nx + pad
    nzd = nz + pad
    
#    # Input processing --------------------------------------------------------
    inputs = periodic_padding(tf.reshape(parsed_rec['comp_raw1'],(nz, nx)),((padding,padding),(padding,padding)))
    inputs = tf.reshape(inputs,(1,nzd,nxd))

    for i_comp in range(1,app.N_VARS_IN):
        new_input = tf.reshape(parsed_rec[f'comp_raw{i_comp+1}'],(nz,nx))
        inputs = tf.concat((inputs, tf.reshape(periodic_padding(new_input,((padding,padding),(padding,padding))),(1,nzd,nxd))),0)


    output1 = tf.reshape(parsed_rec['comp_out_raw1'],(1,nz, nx))

    if app.N_VARS_OUT == 1:
        return inputs, output1
    else:
        output2 = tf.reshape(parsed_rec['comp_out_raw2'],(1,nz, nx))
        if app.N_VARS_OUT == 2:
            return inputs, (output1, output2)
        else:
            output3 = tf.reshape(parsed_rec['comp_out_raw3'],(1,nz, nx))
            return inputs, (output1, output2, output3)

#%% Functions for the NN
def cnn_model():
    input_data = layers.Input(shape=input_shape, name='input_data')
    ini = 30
    dx = 4
    # ------------------------------------------------------------------
    cnv_1 = layers.Conv2D(ini, (3, 3), padding=padding,
                                data_format='channels_first')(input_data)
    bch_1 = layers.BatchNormalization(axis=1)(cnv_1)
    act_1 = layers.Activation('relu')(bch_1)
    # ------------------------------------------------------------------
    cnv_2 = layers.Conv2D(ini+dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_1)
    bch_2 = layers.BatchNormalization(axis=1)(cnv_2)
    act_2 = layers.Activation('relu')(bch_2)
    # ------------------------------------------------------------------
    cnv_3 = layers.Conv2D(ini+2*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_2)
    bch_3 = layers.BatchNormalization(axis=1)(cnv_3)
    act_3 = layers.Activation('relu')(bch_3)
    # ------------------------------------------------------------------
    cnv_4 = layers.Conv2D(ini+3*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_3)
    bch_4 = layers.BatchNormalization(axis=1)(cnv_4)
    act_4 = layers.Activation('relu')(bch_4)
    # ------------------------------------------------------------------
    cnv_5 = layers.Conv2D(ini+4*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_4)
    bch_5 = layers.BatchNormalization(axis=1)(cnv_5)
    act_5 = layers.Activation('relu')(bch_5)
    # ------------------------------------------------------------------
    cnv_6 = layers.Conv2D(ini+5*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
    bch_6 = layers.BatchNormalization(axis=1)(cnv_6)
    act_6 = layers.Activation('relu')(bch_6)
    # ------------------------------------------------------------------
    cnv_7 = layers.Conv2D(ini+6*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_6)
    bch_7 = layers.BatchNormalization(axis=1)(cnv_7)
    act_7 = layers.Activation('relu')(bch_7)
    # ------------------------------------------------------------------
    cnv_8 = layers.Conv2D(ini+7*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_7)
    bch_8 = layers.BatchNormalization(axis=1)(cnv_8)
    act_8 = layers.Activation('relu')(bch_8)
    # ------------------------------------------------------------------
    cnv_9 = layers.Conv2D(ini+8*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_8)
    bch_9 = layers.BatchNormalization(axis=1)(cnv_9)
    act_9 = layers.Activation('relu')(bch_9)
    # ------------------------------------------------------------------
    cnv_10 = layers.Conv2D(ini+9*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_9)
    bch_10 = layers.BatchNormalization(axis=1)(cnv_10)
    act_10 = layers.Activation('relu')(bch_10)
    # ------------------------------------------------------------------
    cnv_11 = layers.Conv2D(ini+10*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_10)
    bch_11 = layers.BatchNormalization(axis=1)(cnv_11)
    act_11 = layers.Activation('relu')(bch_11)
    # ------------------------------------------------------------------
    cnv_12 = layers.Conv2D(ini+11*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_11)
    bch_12 = layers.BatchNormalization(axis=1)(cnv_12)
    act_12 = layers.Activation('relu')(bch_12)
    # ------------------------------------------------------------------
    cnv_13 = layers.Conv2D(ini+12*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_12)
    bch_13 = layers.BatchNormalization(axis=1)(cnv_13)
    act_13 = layers.Activation('relu')(bch_13)
    # ------------------------------------------------------------------
    cnv_14 = layers.Conv2D(ini+13*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_13)
    bch_14 = layers.BatchNormalization(axis=1)(cnv_14)
    act_14 = layers.Activation('relu')(bch_14)
    # ------------------------------------------------------------------
    cnv_15 = layers.Conv2D(ini+14*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_14)
    bch_15 = layers.BatchNormalization(axis=1)(cnv_15)
    act_15 = layers.Activation('relu')(bch_15)
    # ------------------------------------------------------------------
    cnv_16 = layers.Conv2D(ini+14*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_15)
    bch_16 = layers.BatchNormalization(axis=1)(cnv_16)
    act_16 = layers.Activation('relu')(bch_16)
    # ------------------------------------------------------------------
    cnv_17 = layers.Conv2D(ini+13*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_16)
    bch_17 = layers.BatchNormalization(axis=1)(cnv_17)
    act_17 = layers.Activation('relu')(bch_17)
    # ------------------------------------------------------------------
    cnv_18 = layers.Conv2D(ini+12*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_17)
    bch_18 = layers.BatchNormalization(axis=1)(cnv_18)
    act_18 = layers.Activation('relu')(bch_18)
    # ------------------------------------------------------------------
    cnv_19 = layers.Conv2D(ini+11*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_18)
    bch_19 = layers.BatchNormalization(axis=1)(cnv_19)
    act_19 = layers.Activation('relu')(bch_19)
    # ------------------------------------------------------------------
    cnv_20 = layers.Conv2D(ini+10*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_19)
    bch_20 = layers.BatchNormalization(axis=1)(cnv_20)
    act_20 = layers.Activation('relu')(bch_20)
    # ------------------------------------------------------------------
    cnv_21 = layers.Conv2D(ini+9*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_20)
    bch_21 = layers.BatchNormalization(axis=1)(cnv_21)
    act_21 = layers.Activation('relu')(bch_21)
    # ------------------------------------------------------------------
    cnv_22 = layers.Conv2D(ini+8*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_21)
    bch_22 = layers.BatchNormalization(axis=1)(cnv_22)
    act_22 = layers.Activation('relu')(bch_22)
    # ------------------------------------------------------------------
    cnv_23 = layers.Conv2D(ini+7*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_22)
    bch_23 = layers.BatchNormalization(axis=1)(cnv_23)
    act_23 = layers.Activation('relu')(bch_23)
    # ------------------------------------------------------------------
    cnv_24 = layers.Conv2D(ini+6*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_23)
    bch_24 = layers.BatchNormalization(axis=1)(cnv_24)
    act_24 = layers.Activation('relu')(bch_24)
    # ------------------------------------------------------------------
    cnv_25 = layers.Conv2D(ini+5*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_24)
    bch_25 = layers.BatchNormalization(axis=1)(cnv_25)
    act_25 = layers.Activation('relu')(bch_25)
    # ------------------------------------------------------------------
    cnv_26 = layers.Conv2D(ini+4*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_25)
    bch_26 = layers.BatchNormalization(axis=1)(cnv_26)
    act_26 = layers.Activation('relu')(bch_26)
    # ------------------------------------------------------------------
    cnv_27 = layers.Conv2D(ini+3*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_26)
    bch_27 = layers.BatchNormalization(axis=1)(cnv_27)
    act_27 = layers.Activation('relu')(bch_27)
    # ------------------------------------------------------------------
    cnv_28 = layers.Conv2D(ini+2*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_27)
    bch_28 = layers.BatchNormalization(axis=1)(cnv_28)
    act_28 = layers.Activation('relu')(bch_28)
    # ------------------------------------------------------------------
    cnv_29 = layers.Conv2D(ini+dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_28)
    bch_29 = layers.BatchNormalization(axis=1)(cnv_29)
    act_29 = layers.Activation('relu')(bch_29)
    # ------------------------------------------------------------------
    cnv_30 = layers.Conv2D(ini, (3, 3), padding=padding,
                                data_format='channels_first')(act_29)
    bch_30 = layers.BatchNormalization(axis=1)(cnv_30)
    act_30 = layers.Activation('relu')(bch_30)


    # Different branches for different components
    
    # Branch 1
    cnv_b1 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
    if pred_fluct == True:
        act_b1 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b1')(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)
    else:
        act_b1 = layers.Activation('relu')(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)

    losses = {'output_b1':'mse'}
    
    if app.N_VARS_OUT == 2:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
        if pred_fluct == True:
            act_b2 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b2')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)

        outputs_model = [output_b1, output_b2]

        losses['output_b2']='mse'
    
    elif app.N_VARS_OUT == 3:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
        if pred_fluct == True:
            act_b2 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b2')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
    
        losses['output_b2']='mse'

        # Branch 3
        cnv_b3 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
        if pred_fluct == True:
            act_b3 = layers.Lambda(lambda x: tf.maximum(x, -1.0), name='act_b3')(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)
        else:
            act_b3 = layers.Activation('relu')(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)
    
        outputs_model = [output_b1, output_b2, output_b3]

        losses['output_b3']='mse'
    
    else:
        outputs_model = output_b1
    
    CNN_model = tf.keras.models.Model(inputs=input_data, outputs=outputs_model)
    return CNN_model, losses

    
def step_decay_schedule(epoch, initial_lr=1e-3, drop=0.5, epochs_drop=10):
    """
    Step decay function for learning rate scheduling.
    """
    lrate = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

# Define your parameters

lrate_scheduler = LearningRateScheduler(lambda epoch: step_decay_schedule(epoch, initial_learning_rate, drop, epochs_drop))

# Final ReLu function for fluctuations

# Credit to Marcin Możejko for the Callback definition
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
#%% Reading configuration

cur_path = app.CUR_PATH
ds_path_train = app.DS_PATH_TRAIN
ds_path_validation = app.DS_PATH_VALIDATION
# Average profiles folder
avg_path = ds_path_train +'/.avg/'

train_yp = app.TRAIN_YP
target_yp = app.TARGET_YP
if not(type(target_yp) is int):
    target_yp = target_yp[0]

interv = app.INTERV_TRAIN
tfr_path_train = ds_path_train+f'/.tfrecords_singlefile_train_dt{int(11.57*100*interv)}_f32/'
tfr_path_validation = ds_path_validation+f'/.tfrecords_singlefile_validation_dt{int(11.57*100*interv)}_f32/'
epochs = app.N_EPOCHS
batch_size = app.BATCH_SIZE

if distributed_training:
    print('WARNING: The provided batch size is used in each device of the distributed training')
    batch_size *= strategy.num_replicas_in_sync
validation_split = app.VAL_SPLIT
# Learning rate config
init_lr = app.INIT_LR
lr_drop = app.LR_DROP
lr_epdrop = app.LR_EPDROP

if app.NET_MODEL == 1:
    pad = tf.constant(64)
    pad_out = 2
    padding_in = 64
    padding_out = 0
else:
    pad = tf.constant(0)
    raise ValueError('NET_MODEL = 1 is the only one implentated so far')

# in_vars = ''
# for i in range(app.N_VARS_IN):
#     in_vars = in_vars + app.VARS_NAME_IN[i]

tstamp = int(time.time())

NAME = prb_def

n_samp_train = np.array(app.N_SAMPLES_TRAIN)
n_samp_valid = np.array(app.N_SAMPLES_VALIDATION)

n_samples_train = n_samp_train # np.repeat(n_samp_train,2)
n_samples_validation = n_samp_valid # np.repeat(n_samp_valid,2)

n_files_train = int(0.5*n_samples_train.shape[0])
n_files_validation = int(0.5*n_samples_validation.shape[0])

print('n_files_train:'+str(n_files_train))
print('n_files_validation:'+str(n_files_validation))

n_samples_train = np.array(app.N_SAMPLES_TRAIN)
n_samples_tot_train = np.sum(n_samples_train)

n_samples_validation = np.array(app.N_SAMPLES_VALIDATION)
n_samples_tot_validation = np.sum(n_samples_validation)
n_samples_tot = n_samples_tot_train + n_samples_tot_validation

print('n_samp_train:'+str(n_samples_tot_train))
print('n_samp_validation:'+str(n_samples_tot_validation))

#%% Settings for TFRecords
tfr_files_train = [os.path.join(tfr_path_train,f) for f in os.listdir(tfr_path_train) if os.path.isfile(os.path.join(tfr_path_train,f))]
tfr_files_validation = [os.path.join(tfr_path_validation,f) for f in os.listdir(tfr_path_validation) if os.path.isfile(os.path.join(tfr_path_validation,f))]

regex = re.compile(f'yp{target_yp:03d}')
regex_t = re.compile(f'yp{train_yp:03d}')
regex_tr = re.compile(f'train')
regex_ts = re.compile(f'validation')
regex_tb = re.compile('Ret180')
regex_p = re.compile('velocityn25')
#regex_q = re.compile('of-002')

tfr_files_train = [string for string in tfr_files_train if re.search(regex,string) and re.search(regex_tr,string) and re.search(regex_tb,string) and re.search(regex_t,string) and re.search(regex_p,string)] # and re.search(regex_q,string)]
tfr_files_train = [string for string in tfr_files_train if int((string.split('_')[-3].split('-')[-1])[2:]) == target_yp and int((string.split('_')[-3].split('-')[0])[2:]) == train_yp]
tfr_files_validation = [string for string in tfr_files_validation if re.search(regex,string) and re.search(regex_ts,string) and re.search(regex_tb,string) and re.search(regex_t,string) and re.search(regex_p,string)] # and re.search(regex_q,string)]
tfr_files_validation = [string for string in tfr_files_validation if int((string.split('_')[-3].split('-')[-1])[2:]) == target_yp and int((string.split('_')[-3].split('-')[0])[2:]) == train_yp]

tfr_files_train = [string for string in tfr_files_train if int(string.split('_')[-2][4:7])<n_files_train]
tfr_files_train = sorted(tfr_files_train)
tfr_files_validation = sorted([string for string in tfr_files_validation if int(string.split('_')[-2][4:7])<n_files_validation])
tfr_files_validation = sorted(tfr_files_validation)

print('tfr train:')
for i in tfr_files_train:
    print(i)
print('tfr valid:')
for i in tfr_files_validation:
    print(i)

# Separating files for training and validation
Ret = (tfr_files_train[0].split('/')[-1]).split('_')[0][3:]

(nx_, ny_, nz_) = [int(val) for val in tfr_files_train[0].split('/')[-1].split('_')[1].split('x')]
nx_ = 432
nz_ = 432

#%% Old setting again

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
    if pred_fluct == True:
        NAME = NAME + 'fluct'
    if not(str(target_yp) in ypos_Ret):
        raise ValueError("The selected target does not have a corresponding y-index in simulation")
except NameError:
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

NAME += str(app.NET_MODEL) + f'TF2_{app.N_VARS_IN}'

if norm_input:
    NAME += 'Norm'
NAME += f'In-{app.N_VARS_OUT}'
if scale_output:
    NAME += 'Scaled'
NAME += f'Out'

# TODO: Update the way the model name is registered in the new name
# TODO: Add correct naming when the model is loaded from a checkpoint
if app.INIT == 'model':
    model_path = app.INIT_MODEL
    NAME = NAME + '_init' + model_path[-10:]
    if app.TRANSFER_LEARNING == True:
        NAME = NAME + 'tr' + str(app.N_TRAINABLE_LAYERS)

if prb_def == 'NN_WallRecon':
    NAME = NAME+f'_{train_yp}-'+ \
        str(target_yp)+'_'+str(nx_)+ \
        'x'+str(nz_)+'_Ret'+str(Ret)+'_lr'+str(init_lr)+'_decay'+ \
        str(int(lr_epdrop))+ 'drop'+str(lr_drop)+ \
        '_relu-'+str(tstamp)

nx = nx_ + padding_in
nz = nz_ + padding_in

input_shape = (app.N_VARS_IN, nz, nx)

        
#elif norm_input == True and prb_def == 'OuterRecon': #GB
#    avgs_in = tf.constant(np.array([avgs[i][ypos_Ret[str(train_yp)]] for i in range(0,app.N_VARS_IN)],dtype=np.float32))
#    std_in = tf.constant(np.array([rms[i][ypos_Ret[str(train_yp)]] for i in range(0,app.N_VARS_IN)],dtype=np.float32))
    

# Callbacks
tensorboard = SubTensorBoard(log_dir='.logs/{}'.format(NAME),
                          histogram_freq=app.TB_HIST_FREQ
)

checkpoint = ModelCheckpoint('.logs/'+NAME+'/model.ckpt.{epoch:04d}.hdf5', \
                             verbose=2, monitor='val_output_b1_loss',save_best_only=True,mode='min')

lrate = tf.keras.callbacks.LearningRateScheduler(step_decay_schedule)
time_callback = TimeHistory()


# tfr_files_input_train_ds = tf.data.Dataset.list_files(tfr_files_input_train, seed=666)
tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)

# tfr_files_input_val_ds = tf.data.Dataset.list_files(tfr_files_input_valid, seed=686)
tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_validation, seed=686)



tfr_files_train_ds = tfr_files_train_ds.interleave(lambda x : tf.data.TFRecordDataset(x).take(tf.gather(n_samples_train, (tf.strings.to_number(tf.strings.substr(tf.strings.split(x,sep='_')[-1],0,3),tf.int32)-1)*n_files_train +  tf.strings.to_number(tf.strings.substr(tf.strings.split(x,sep='_')[-2],4,3),tf.int32))), cycle_length=16,num_parallel_calls=tf.data.experimental.AUTOTUNE)
                       
#tfr_files_train_ds = tfr_files_train_ds.interleave(lambda x : tf.data.TFRecordDataset(x).take(tf.cast(tf.gather(\
#                       n_samples_train,tf.gather(n_samples_train,tf.strings.to_number(tf.strings.substr(tf.strings.split(x,sep='_')[-2],4,3)\
#                       ,tf.int64))+tf.strings.to_number(tf.strings.split(tf.strings.split(x,sep='_')[-1],sep='-')[0],tf.int64)-1)\
#                       ,dtype=tf.int64)), cycle_length=16,num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#tfr_files_val_ds = tfr_files_val_ds.interleave(lambda x : tf.data.TFRecordDataset(x).take(tf.cast(tf.gather(\
#                       n_samples_validation,tf.gather(n_samples_validation,tf.strings.to_number(tf.strings.substr(tf.strings.split(x,sep='_')[-2],4,3)\
#                       ,tf.int64))+tf.strings.to_number(tf.strings.split(tf.strings.split(x,sep='_')[-1],sep='-')[0],tf.int64)-1)\
#                       ,dtype=tf.int64)), cycle_length=16,num_parallel_calls=tf.data.experimental.AUTOTUNE)


tfr_files_val_ds = tfr_files_val_ds.interleave(lambda x : tf.data.TFRecordDataset(x).take(tf.gather(n_samples_validation, (tf.strings.to_number(tf.strings.substr(tf.strings.split(x,sep='_')[-1],0,3),tf.int32)-1)*n_files_validation +  tf.strings.to_number(tf.strings.substr(tf.strings.split(x,sep='_')[-2],4,3),tf.int32))), cycle_length=16,num_parallel_calls=tf.data.experimental.AUTOTUNE) # Can I change the cycle length? GB

#print(tfr_files_train_ds)
#print(tfr_files_val_ds)
#tfr_files_train_ds = tfr_files_train_ds.interleave(lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) \
#    if tf.math.equal(x,shared_tfr_out) \
#    else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-1],sep='-')[0], tf.int32)-1)),
#    cycle_length=16,
#    num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#tfr_files_val_ds = tfr_files_val_ds.interleave(lambda x : tf.data.TFRecordDataset(x).skip(samples_train_shared).take(n_samples_tfr_shared - samples_train_shared) \
#    if tf.math.equal(x,shared_tfr_out) \
#    else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-1],sep='-')[0], tf.int32)-1)),
#    cycle_length=16,
#    num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Parsing datasets ------------------------------------------------------------
# parsed_ds_input_train = tfr_files_input_train_ds.map(parse_features)#, num_parallel_calls=tf.data.experimental.AUTOTUNE)    # Add number_of_parallel_calls?
dataset_train = tfr_files_train_ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# parsed_ds_input_val = tfr_files_input_val_ds.map(parse_features)#, num_parallel_calls=tf.data.experimental.AUTOTUNE)        # Add number_of_parallel_calls?
dataset_val = tfr_files_val_ds.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Zipping input and output together -------------------------------------------
# dataset_train = tf.data.Dataset.zip((parsed_ds_input_train, parsed_ds_output_train))
# dataset_val = tf.data.Dataset.zip((parsed_ds_input_val, parsed_ds_output_val))

# Datasets size check ---------------------------------------------------------
#itr = iter(dataset_train)
#j = 0
#for i in range(n_samp_train):
#    example = next(itr)
#    j += 1
#
#try:
#    example = next(itr)
#except StopIteration:
#    print(f'Train set over: {j}')
#
#itr1 = iter(dataset_val)
#jj = 0
#for i in range(n_samp_valid):
#    example1 = next(itr1)
##    if np.any(np.isnan(example1[0].numpy())):
##         sys.exit(1)
##     elif np.any(np.isnan(example1[1][0].numpy())):
##         sys.exit(2)
##     elif np.any(np.isnan(example1[1][1].numpy())):
##         sys.exit(3)
##     elif np.any(np.isnan(example1[1][2].numpy())):
##         sys.exit(4)
#
#    jj += 1
#
#try:
#    example1 = next(itr1)
#except StopIteration:
#    print(f'Valid set over: {jj}')
#print(NAME)
#sys.exit(0)

# Dataset shuffling -----------------------------------------------------------

if Ret == str(180):
    shuffle_buffer = 2500
    n_prefetch = 1

dataset_train = dataset_train.shuffle(shuffle_buffer)
# dataset_train = dataset_train.filter(interval_filter_fn)
dataset_train = dataset_train.repeat(epochs)
dataset_train = dataset_train.batch(batch_size=batch_size)
dataset_train = dataset_train.prefetch(n_prefetch)

#dataset_val = dataset_val.shuffle(shuffle_buffer)
# dataset_val = dataset_val.filter(interval_filter_fn)
#dataset_val = dataset_val.cache()
dataset_val = dataset_val.repeat(epochs)
dataset_val = dataset_val.batch(batch_size=batch_size)
dataset_val = dataset_val.prefetch(n_prefetch)

print('')
print('# ====================================================================')
print('#     Summary of the options for the model                            ')
print('# ====================================================================')
print('')
print(f'Model name: {NAME}')
print(f'Number of samples for training: {int(n_samples_tot_train)}')
print(f'Number of samples for validation: {int(n_samples_tot_validation)}')
print(f'Total number of samples: {n_samples_tot}')
print(f'Batch size: {batch_size}')
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

#%% Keras model

padding = 'valid'

if distributed_training:
   print('Compiling and training the model for multiple GPU')
   if app.INIT == 'model':
       init_model = tf.keras.models.load_model(model_path)
       init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
   
   with strategy.scope():
       CNN_model, losses = cnn_model()
       
       if app.INIT == 'model':
           print('Weights of the model initialized with another trained model')
           # init_model = tf.keras.models.load_model(model_path)
           # init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
           CNN_model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
           os.remove('/tmp/model_weights-CNN_keras_model.h5')

           # A smaller learning rate is used in this case
           init_lr = init_lr/2
       
       CNN_model.compile(loss=['mse','mse','mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate))
else:
   CNN_model, losses = cnn_model()
   # Initialization of the model for transfer learning, if required
   if app.INIT == 'model':
       print('Weights of the model initialized with another trained model')
       # TODO: check if this condition is still valid for the models that were
       # added later
#       if int(model_path[-67]) != app.NET_MODEL:
#           raise ValueError('The model for initialization is different from the model to be initialized')
           
       init_model = tf.keras.models.load_model(model_path)
       init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
       CNN_model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
       os.remove('/tmp/model_weights-CNN_keras_model.h5')
       
       # A smaller learning rate is used in this case
       init_lr = init_lr/2
       
       # TODO: Modify this implementation of transfer learning to account for cropping layers
       # if app.TRANSFER_LEARNING == True:
       #     lyrs = CNN_model.layers
       #     n_lyrs = len(lyrs)
       #     for i_l in range(n_lyrs):
       #         print(CNN_model.layers[i_l].name, CNN_model.layers[i_l].trainable)
       #         if i_l <= n_lyrs - (2+3*(app.N_TRAINABLE_LAYERS-1)) - 1:  # Every layer has 3 sublayers (conv+batch_norm+activ), except the last one (no batch_norm)
       #             CNN_model.layers[i_l].trainable = False
       #         print(CNN_model.layers[i_l].name, CNN_model.layers[i_l].trainable)

   elif app.INIT == 'random':
       print('Weights of the model initialized from random distributions')

   print('Compiling and training the model for single GPU')
   CNN_model.compile(loss=['mse','mse','mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate))
    
print(CNN_model.summary())

#print('Evaluating model performance before training')
#
#ds_eval_fake = dataset_train.take(256)
#
#loss_values = CNN_model.evaluate(ds_eval_fake)
#print(loss_values)
#sys.exit(0)

# Training and evaluation
train_history = CNN_model.fit(dataset_train,
                              epochs=epochs,
                              steps_per_epoch=int(np.ceil(n_samples_tot_train/batch_size)),
                              validation_data=dataset_val,
                              validation_steps=int(np.ceil(n_samples_tot_validation/batch_size)),
                              verbose=2,
                              callbacks=[tensorboard, checkpoint, lrate, time_callback])
# validation_data=dataset_train,
#                               validation_steps=int(np.ceil(n_samp_train/batch_size)),
#                               verbose=2,
#                               callbacks=[tensorboard, checkpoint, lrate, time_callback])
# Saving model

save_path = cur_path+'/.saved_models/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
epoch_path = cur_path+'/.epoch_log/'
if not os.path.exists(epoch_path):
    os.mkdir(epoch_path)

tf.keras.models.save_model(
    CNN_model,
    save_path+NAME,
    overwrite=True,
    include_optimizer=True,
    save_format='h5'
)

# Saving history

tLoss = train_history.history['loss']
vLoss = train_history.history['val_loss']
tTrain = time_callback.times

np.savez(epoch_path+NAME+'_log', tLoss=tLoss, vLoss=vLoss, tTrain=tTrain)

