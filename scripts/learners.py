#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, absolute_import
from six import iteritems

import sys
import os
import numpy
import logging
import random
import warnings
import importlib
import copy
import scipy

from sklearn.metrics import mean_absolute_error
from math import sqrt
from datetime import datetime
from .files import DataFile
from .containers import ContainerMixin, DottedDict
from .features import FeatureContainer
from .utils import SuppressStdoutAndStderr
from .metadata import MetaDataContainer, MetaDataItem, EventRoll
from .utils import Timer
from keras.models import load_model

from tqdm import tqdm
from IPython import embed

import math
import tensorflow as tf
################################################## Import CNN below
import numpy as np
import os
from recurrentshop import LSTMCell, RecurrentSequential
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import keras.backend as K

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
#from keras.initializations import norRemal, identity
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop, Adadelta
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.models import model_from_json

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras import backend as K

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

maxToAdd = 1
DatDim = 40
AggreNum = 10
ClassNum = 6


def vae_loss(x, x_hat):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    xent_loss = 40 * objectives.binary_crossentropy(x, x_hat)
    mse_loss = 40 * objectives.mse(x, x_hat) 
    if use_loss == 'xent':
        return xent_loss + kl_loss
    if use_loss == 'mse':
        return mse_loss + kl_loss

def CalculateDist(data,mean,var):
    Dis = 0
    for c in range(DatDim):
        Dis = Dis + (data[c]-mean[c])*(data[c]-mean[c])/(var[c])
    Dis = Dis/DatDim
    return Dis
def findMax(feature):
    ma = -100000;
    for i in range(feature.shape[0]):
        if(feature[i] >= ma):
            index = i
            ma = feature[i]
    return index
def findMin(feature):
    mi = 100000;
    for i in range(feature.shape[0]):
        if(feature[i] <= mi):
            index = i
            mi = feature[i]
    return index    
def GetTcen(feature):
    fvecs = np.zeros((1,DatDim))
    for dim in range(feature.shape[1]):
        summ = 0
        summm = 0
        for j in range(feature.shape[0]):
            summ = summ + feature[j,dim]*j/feature.shape[0]
            summm = summm + feature[j,dim]
        fvecs[1,dim] = summ/summm;
    fvecs_np = np.array(fvecs).astype(np.float32)   
    return fvecs_np
def AlignProbability(frame_probabilities,frame_probabilities_CQT):
    fvecs = np.zeros((frame_probabilities.shape[0],frame_probabilities.shape[1]))
    fvecs = frame_probabilities
    cnt = 0
    cnt_cqt = 0
    while(cnt < frame_probabilities_CQT.shape[1]*5):        
        for c in range(frame_probabilities.shape[0]):
            fvecs[c][cnt] = 0.5*frame_probabilities[c][cnt] + frame_probabilities_CQT[c][cnt_cqt]*0.5           
        cnt = cnt + 1;
        if(cnt % 5 == 0):
            cnt_cqt = cnt_cqt + 1
    fvecs_np = np.array(fvecs).astype(np.float32)  
    return fvecs_np
def StoreSimilarity():
    fvecs = np.mat(np.ones((ClassNum,ClassNum)));
    ma = 11.369
    
    fvecs[0,1] = 1-2.5158/ma
    fvecs[0,2] = 1-11.1369/ma
    fvecs[0,3] = 1-0.8432/ma
    fvecs[0,4] = 1-5.6038/ma
    fvecs[0,5] = 1-2.9298/ma

    fvecs[1,0] = fvecs[0,1]
    fvecs[1,2] = 1-9.2338/ma
    fvecs[1,3] = 1-2.1359/ma
    fvecs[1,4] = 1-3.9037/ma
    fvecs[1,5] = 1-1.5912/ma

    fvecs[2,0] = fvecs[0,2]
    fvecs[2,1] = fvecs[1,2]
    fvecs[2,3] = 1 - 10.9896/ma
    fvecs[2,4] = 1 - 5.5521/ma
    fvecs[2,5] = 1 - 8.2964/ma

    fvecs[3,0] = fvecs[0,3]
    fvecs[3,1] = fvecs[1,3]
    fvecs[3,2] = fvecs[2,3]
    fvecs[3,4] = 1 - 5.4650/ma
    fvecs[3,5] = 1 - 2.7516/ma

    fvecs[4,0] = fvecs[0,4]
    fvecs[4,1] = fvecs[1,4]
    fvecs[4,2] = fvecs[2,4]
    fvecs[4,3] = fvecs[3,4]
    fvecs[4,5] = 1 - 2.7789/ma
    
    return fvecs

def AggreData(feature,AggreNum = AggreNum):
    fvecs = np.ones((feature.shape[0],DatDim*AggreNum))    
    for i in range(feature.shape[0]-AggreNum+1):
        for j in range(AggreNum):
            fvecs[i,j*DatDim:(j+1)*DatDim] = feature[i+j,0:DatDim]    
    for a in range(AggreNum-1):
        i = i + 1
        z = np.zeros((1,DatDim))       
        k =0        
        fvecs_temp = np.zeros((1,AggreNum*DatDim))     
        fvecs_temp[0,0:DatDim] = feature[i+k,0:DatDim]
        k = k + 1
        while k<AggreNum-a-1:
            fvecs_temp[0,k*DatDim:(k+1)*DatDim] = feature[i+k,0:DatDim]            
            k = k+ 1
        fvecs[i,:] = fvecs_temp
    fvecs_np = np.array(fvecs).astype(np.float32)   
    return fvecs_np    
def extract_data(filename_feature):

    # Arrays to hold the labels and feature vectors.    
    labels = []
    fvecs = []
    fvecl = []
    file = open(filename_feature)
    i = 0
    for line in file:
        row = line.split(",")
        i = i + 1
        fvecs.append([float(x) for x in row[0:DatDim]])
        fvecl.append([float(x) for x in row[DatDim:DatDim+ClassNum]])        
    fvecs_np = np.array(fvecs).astype(np.float32)    
    fvecs_label = np.array(fvecl).astype(np.int)    
    return fvecs_np,fvecs_label,i  

def extract_data_raw_test(filename_feature):

    # Arrays to hold the labels and feature vectors.    
    fvecs = []
    file = open(filename_feature)
    i = 0
    for line in file:
        row = line.split(",")
        i = i + 1
        fvecs.append([float(x) for x in row[0:DatDim]])      
    fvecs_np = np.array(fvecs).astype(np.float32)       
    return fvecs_np      
def GetNormalizeData(data,mean,var):
    #print(data.shape)
    #print(mean.shape)
    fvecs = np.ones((data.shape[0],data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            fvecs[i][j] = (data[i][j] - mean[0][j])/var[0][j]            
    fvecs_np = np.array(fvecs).astype(np.float32)       
    
    return fvecs_np
def ReverseData(feature):
    fvecs = np.ones((feature.shape[0],DatDim*AggreNum))
    j = 0
    print(feature.shape[0])
    for num in range (feature.shape[0]):
        flag_small = 0
        flag_large = AggreNum
        #print('---------------')
        for aggre in range (AggreNum):
            if((aggre % 2) != 0):
                fvecs[num,aggre*DatDim:(aggre+1)*DatDim] = feature[num,(flag_large-1)*DatDim:flag_large*DatDim]
                flag_large = flag_large - 1
            if((aggre % 2) == 0):
                fvecs[num,aggre*DatDim:(aggre+1)*DatDim] = feature[num,flag_small*DatDim:(flag_small+1)*DatDim]
                flag_small = flag_small + 1
    fvecs_np = np.array(fvecs).astype(np.float32)
    return fvecs_np    
def CombineData(featurea, featureb):
    fvecs = np.ones((featurea.shape[0],DatDim*2))
    cnt = 0   
    for i in range(featurea.shape[0]):
        fvecs[i,0:DatDim] = featurea[i,:]
        fvecs[i,DatDim:DatDim*2] = featureb[i,:]
    fvecs_np = np.array(fvecs).astype(np.float32)   
    return fvecs_np     
#########################################################################################Import CRNN above
def multilayer_perception(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, 0.8)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, 0.8)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
def scene_classifier_factory(*args, **kwargs):
    if kwargs.get('method', None) == 'gmm':
        return SceneClassifierGMM(*args, **kwargs)
    elif kwargs.get('method', None) == 'mlp':
        return SceneClassifierMLP(*args, **kwargs)
    else:
        raise ValueError('{name}: Invalid SegmentClassifier method [{method}]'.format(
            name='segment_classifier_factory',
            method=kwargs.get('method', None))
        )


def event_detector_factory(*args, **kwargs):
    if kwargs.get('method', None) == 'gmm':
        return EventDetectorGMM(*args, **kwargs)
    elif kwargs.get('method', None) == 'mlp':
        return EventDetectorMLP(*args, **kwargs)
    else:
        raise ValueError('{name}: Invalid EventDetector method [{method}]'.format(
            name='event_detector_factory',
            method=kwargs.get('method', None))
        )


class LearnerContainer(DataFile, ContainerMixin):
    valid_formats = ['cpickle']

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        method : str
            Method label
            Default value "None"
        class_labels : list of strings
            List of class labels
            Default value "[]"
        params : dict or DottedDict
            Parameters
        feature_masker : FeatureMasker or class inherited from FeatureMasker
            Feature masker instance
            Default value "None"
        feature_normalizer : FeatureNormalizer or class inherited from FeatureNormalizer
            Feature normalizer instance
            Default value "None"
        feature_stacker : FeatureStacker or class inherited from FeatureStacker
            Feature stacker instance
            Default value "None"
        feature_aggregator : FeatureAggregator or class inherited from FeatureAggregator
            Feature aggregator instance
            Default value "None"
        logger : logging
            Instance of logging
            Default value "None"
        disable_progress_bar : bool
            Disable progress bar in console
            Default value "False"
        log_progress : bool
            Show progress in log.
            Default value "False"
        show_extra_debug : bool
            Show extra debug information
            Default value "True"

        """

        super(LearnerContainer, self).__init__({
            'method': kwargs.get('method', None),
            'class_labels': kwargs.get('class_labels', []),
            'params': DottedDict(kwargs.get('params', {})),
            'feature_masker': kwargs.get('feature_masker', None),
            'feature_normalizer': kwargs.get('feature_normalizer', None),
            'feature_stacker': kwargs.get('feature_stacker', None),
            'feature_aggregator': kwargs.get('feature_aggregator', None),
            'model': kwargs.get('model', {}),
            'learning_history': kwargs.get('learning_history', {}),
        }, *args, **kwargs)

        # Set randomization seed
        if self.params.get_path('seed') is not None:
            self.seed = self.params.get_path('seed')
        elif self.params.get_path('parameters.seed') is not None:
            self.seed = self.params.get_path('parameters.seed')
        elif kwargs.get('seed', None):
            self.seed = kwargs.get('seed')
        else:
            bigint, mod = divmod(int(datetime.now().strftime("%s")) * 1000, 2**32)
            self.seed = mod

        self.logger = kwargs.get('logger',  logging.getLogger(__name__))
        self.disable_progress_bar = kwargs.get('disable_progress_bar',  False)
        self.log_progress = kwargs.get('log_progress',  False)
        self.show_extra_debug = kwargs.get('show_extra_debug', True)

    @property
    def class_labels(self):
        """Class labels

        Returns
        -------
        list of strings
            List of class labels in the model
        """
        return self.get('class_labels', None)

    @class_labels.setter
    def class_labels(self, value):
        self['class_labels'] = value

    @property
    def method(self):
        """Learner method label

        Returns
        -------
        str
            Learner method label
        """

        return self.get('method', None)

    @method.setter
    def method(self, value):
        self['method'] = value

    @property
    def params(self):
        """Parameters

        Returns
        -------
        DottedDict
            Parameters
        """
        return self.get('params', None)

    @params.setter
    def params(self, value):
        self['params'] = value

    @property
    def feature_masker(self):
        """Feature masker instance

        Returns
        -------
        FeatureMasker

        """

        return self.get('feature_masker', None)

    @feature_masker.setter
    def feature_masker(self, value):
        self['feature_masker'] = value

    @property
    def feature_normalizer(self):
        """Feature normalizer instance

        Returns
        -------
        FeatureNormalizer

        """

        return self.get('feature_normalizer', None)

    @feature_normalizer.setter
    def feature_normalizer(self, value):
        self['feature_normalizer'] = value

    @property
    def feature_stacker(self):
        """Feature stacker instance

        Returns
        -------
        FeatureStacker

        """

        return self.get('feature_stacker', None)

    @feature_stacker.setter
    def feature_stacker(self, value):
        self['feature_stacker'] = value

    @property
    def feature_aggregator(self):
        """Feature aggregator instance

        Returns
        -------
        FeatureAggregator

        """

        return self.get('feature_aggregator', None)

    @feature_aggregator.setter
    def feature_aggregator(self, value):
        self['feature_aggregator'] = value

    @property
    def model(self):
        """Acoustic model

        Returns
        -------
        model

        """

        return self.get('model', None)

    @model.setter
    def model(self, value):
        self['model'] = value

    def set_seed(self, seed=None):
        """Set randomization seeds

        Returns
        -------
        nothing

        """

        if seed is None:
            seed = self.seed

        numpy.random.seed(seed)
        random.seed(seed)

    @property
    def learner_params(self):
        """Get learner parameters from parameter container

        Returns
        -------
        DottedDict
            Learner parameters

        """

        if 'parameters' in self['params']:
            parameters = self['params']['parameters']
        else:
            parameters = self['params']

        return DottedDict({k: v for k, v in parameters.items() if not k.startswith('_')})

    def _get_input_size(self, data):
        input_shape = None
        for audio_filename in data:
            if not input_shape:
                input_shape = data[audio_filename].feat[0].shape[1]
            elif input_shape != data[audio_filename].feat[0].shape[1]:
                message = '{name}: Input size not coherent.'.format(
                    name=self.__class__.__name__
                )
                self.logger.exception(message)
                raise ValueError(message)

        return input_shape


class KerasMixin(object):

    def exists(self):
        keras_model_filename = os.path.splitext(self.filename)[0] + '.model.hdf5'
        return os.path.isfile(self.filename) and os.path.isfile(keras_model_filename)

    def log_model_summary(self):
        layer_name_map = {
            'BatchNormalization': 'BatchNorm',
        }

        from keras.utils.layer_utils import count_total_params
        self.logger.debug('  ')
        self.logger.debug('  Model summary')
        self.logger.debug('  {type:<12s} | {out:10s} | {param:6s}  | {name:20s}  | {conn:27s} | {act:10s} | {init:9s}'.format(
            type='Layer type',
            out='Output',
            param='Param',
            name='Name',
            conn='Connected to',
            act='Activation',
            init='Init')
        )

        self.logger.debug('  {name:<12s} + {out:10s} + {param:6s}  + {name:20s}  + {conn:27s} + {act:10s} + {init:9s}'.format(
            type='-'*12,
            out='-'*10,
            param='-'*6,
            name='-'*20,
            conn='-'*27,
            act='-'*10,
            init='-'*9)
        )

        for layer in self.model.layers:
            connections = []
            for node_index, node in enumerate(layer.inbound_nodes):
                for i in range(len(node.inbound_layers)):
                    inbound_layer = node.inbound_layers[i].name
                    inbound_node_index = node.node_indices[i]
                    inbound_tensor_index = node.tensor_indices[i]
                    connections.append(inbound_layer + '[' + str(inbound_node_index) +
                                       '][' + str(inbound_tensor_index) + ']')

            config = layer.get_config()
            layer_name = layer.__class__.__name__
            if layer_name in layer_name_map:
                layer_name = layer_name_map[layer_name]

            self.logger.debug(
                '  {type:<12s} | {shape:10s} | {params:6s}  | {name:20s}  | {connected:27s} | {activation:10s} | {init:9s}'.format(
                    type=layer_name,
                    shape=str(layer.output_shape),
                    params=str(layer.count_params()),
                    name=str(layer.name),
                    connected=str(connections[0]),
                    activation=str(config.get('activation', '---')),
                    init=str(config.get('init', '---'))
                )
            )
        trainable_count, non_trainable_count = count_total_params(self.model.layers, layer_set=None)
        self.logger.debug('  Total params         : {:,}'.format(trainable_count + non_trainable_count))
        self.logger.debug('  Trainable params     : {:,}'.format(trainable_count))
        self.logger.debug('  Non-trainable params : {:,}'.format(non_trainable_count))

    def plot_model(self, filename='model.png', show_shapes=True, show_layer_names=True):
        from keras.utils.visualize_util import plot
        plot(self.model, to_file=filename, show_shapes=show_shapes, show_layer_names=show_layer_names)

    def create_model(self, input_shape):
        from keras.models import Sequential
        from keras.layers.normalization import BatchNormalization
        self.model = Sequential()
        model_params = copy.deepcopy(self.learner_params.get_path('model.config'))
        for layer_id, layer_setup in enumerate(model_params):
            layer_setup = DottedDict(layer_setup)
            try:
                LayerClass = getattr(importlib.import_module("keras.layers"), layer_setup['class_name'])
            except AttributeError:
                message = '{name}: Invalid Keras layer type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=layer_setup['class_name']
                )
                self.logger.exception(message)
                raise AttributeError(message)

            if 'config' not in layer_setup:
                layer_setup['config'] = {}

            # Set layer input
            if layer_id == 0 and layer_setup.get_path('config.input_shape') is None:
                # Set input layer dimension for the first layer if not set
                if layer_setup.get('class_name') == 'Dropout':
                    layer_setup['config']['input_shape'] = (input_shape,)
                else:
                    layer_setup['config']['input_dim'] = input_shape

            elif layer_setup.get_path('config.input_dim') == 'FEATURE_VECTOR_LENGTH':
                layer_setup['config']['input_dim'] = input_shape
            
            # Set layer output
            if layer_setup.get_path('config.units') == 'CLASS_COUNT':
                layer_setup['config']['units'] = len(self.class_labels)

            if layer_setup.get('config'):                
                self.model.add(LayerClass(**dict(layer_setup.get('config'))))
            else:
                self.model.add(LayerClass())

        try:
            OptimizerClass = getattr(importlib.import_module("keras.optimizers"),
                                     self.learner_params.get_path('model.optimizer.type')
                                     )

        except AttributeError:
            message = '{name}: Invalid Keras optimizer type [{type}].'.format(
                name=self.__class__.__name__,
                type=self.learner_params.get_path('model.optimizer.type')
            )
            self.logger.exception(message)
            raise AttributeError(message)

        self.model.compile(
            loss=self.learner_params.get_path('model.loss'),
            optimizer=OptimizerClass(**dict(self.learner_params.get_path('model.optimizer.parameters', {}))),
            metrics=self.learner_params.get_path('model.metrics')
        )

    def __getstate__(self):
        data = {}
        excluded_fields = ['model']

        for item in self:
            if item not in excluded_fields and self.get(item):
                data[item] = copy.deepcopy(self.get(item))
        data['model'] = os.path.splitext(self.filename)[0] + '.model.hdf5'
        return data

    def _after_load(self, to_return=None):
        with SuppressStdoutAndStderr():
            from keras.models import Sequential, load_model

        if isinstance(self.model, str):
            keras_model_filename = self.model
        else:
            keras_model_filename = os.path.splitext(self.filename)[0] + '.model.hdf5'

        if os.path.isfile(keras_model_filename):
            with SuppressStdoutAndStderr():
                self.model = load_model(keras_model_filename)
        else:
            message = '{name}: Keras model not found [{filename}]'.format(
                name=self.__class__.__name__,
                filename=keras_model_filename
            )

            self.logger.exception(message)
            raise IOError(message)

    def _after_save(self, to_return=None):
        # Save keras model and weight
        keras_model_filename = os.path.splitext(self.filename)[0] + '.model.hdf5'
        model_weights_filename = os.path.splitext(self.filename)[0] + '.weights.hdf5'
        self.model.save(keras_model_filename)
        self.model.save_weights(model_weights_filename)

    def _setup_keras(self):
        """Setup keras backend and parameters"""
        # Set backend and parameters before importing keras
        if self.show_extra_debug:
            self.logger.debug('  ')
            self.logger.debug('  Keras backend \t[{backend}]'.format(
                backend=self.learner_params.get_path('keras.backend', 'theano'))
            )

        if self.learner_params.get_path('keras.backend', 'theano') == 'theano':
            # Default flags
            flags = [
                'warn.round=False'
            ]

            # Set device
            if self.learner_params.get_path('keras.backend_parameters.device'):
                flags.append('device=' + self.learner_params.get_path('keras.backend_parameters.device', 'cpu'))

                if self.show_extra_debug:
                    self.logger.debug('  Theano device \t[{device}]'.format(
                        device=self.learner_params.get_path('keras.backend_parameters.device', 'cpu'))
                    )

            # Set floatX
            if self.learner_params.get_path('keras.backend_parameters.floatX'):
                flags.append('floatX=' + self.learner_params.get_path('keras.backend_parameters.floatX', 'float32'))

                if self.show_extra_debug:
                    self.logger.debug('  Theano floatX \t[{float}]'.format(
                        float=self.learner_params.get_path('keras.backend_parameters.floatX', 'float32'))
                    )

            # Set fastmath
            if self.learner_params.get_path('keras.backend_parameters.fastmath') is not None:
                if self.learner_params.get_path('keras.backend_parameters.fastmath', False):
                    flags.append('nvcc.fastmath=True')
                else:
                    flags.append('nvcc.fastmath=False')

                if self.show_extra_debug:
                    self.logger.debug('  NVCC fastmath \t[{flag}]'.format(
                        flag=str(self.learner_params.get_path('keras.backend_parameters.fastmath', False)))
                    )

            # Set environmental variable for Theano
            os.environ["THEANO_FLAGS"] = ','.join(flags)

        elif self.learner_params.get_path('keras.backend', 'tensorflow') == 'tensorflow':
            # Set device
            if self.learner_params.get_path('keras.backend_parameters.device', 'cpu'):

                # In case of CPU disable visible GPU.
                if self.learner_params.get_path('keras.backend_parameters.device', 'cpu') == 'cpu':
                    os.environ["CUDA_VISIBLE_DEVICES"] = ''

                if self.show_extra_debug:
                    self.logger.debug('  Tensorflow device \t[{device}]'.format(
                        device=self.learner_params.get_path('keras.backend_parameters.device', 'cpu')))

        else:
            message = '{name}: Keras backend not supported [backend].'.format(
                name=self.__class__.__name__,
                backend=self.learner_params.get_path('keras.backend')
            )
            self.logger.exception(message)
            raise AssertionError(message)

        # Select Keras backend
        os.environ["KERAS_BACKEND"] = self.learner_params.get_path('keras.backend', 'theano')


class SceneClassifier(LearnerContainer):
    """Scene classifier (Frame classifier / Multiclass - Singlelabel)"""
    def predict(self, feature_data, recognizer_params=None):
        """Predict scene label for given feature matrix

        Parameters
        ----------
        feature_data : numpy.ndarray
        recognizer_params : DottedDict

        Returns
        -------
        str
            class label

        """

        if recognizer_params is None:
            recognizer_params = {}

        if not isinstance(recognizer_params, DottedDict):
            # Convert parameters to DottedDict
            recognizer_params = DottedDict(recognizer_params)

        if isinstance(feature_data, FeatureContainer):
            # If we have featureContainer as input, get feature_data
            feature_data = feature_data.feat[0]

        # Get frame wise probabilities
        frame_probabilities = self._frame_probabilities(feature_data)

        # Accumulate probabilities
        if recognizer_params.get_path('frame_accumulation.enable', True):
            probabilities = self._accumulate_probabilities(probabilities=frame_probabilities,
                                                           accumulation_type=recognizer_params.get_path('frame_accumulation.type'))
        else:
            # Pass probabilities
            probabilities = frame_probabilities

        # Probability binarization
        if recognizer_params.get_path('frame_binarization.enable', True):
            if recognizer_params.get_path('frame_binarization.type') == 'global_threshold':
                frame_decisions = numpy.argmax(
                    probabilities > recognizer_params.get_path('frame_binarization.threshold', 0.5),
                    axis=0
                )

            elif recognizer_params.get_path('frame_binarization.type') == 'frame_max':
                frame_decisions = numpy.argmax(probabilities, axis=0)

            else:
                message = '{name}: Unknown frame_binarization type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=recognizer_params.get_path('frame_binarization.type')
                )

                self.logger.exception(message)
                raise AssertionError(message)

        # Decision making
        if recognizer_params.get_path('decision_making.enable', True):
            if recognizer_params.get_path('decision_making.type') == 'maximum':
                classification_result_id = numpy.argmax(probabilities)

            elif recognizer_params.get_path('decision_making.type') == 'majority_vote':
                counts = numpy.bincount(frame_decisions)
                classification_result_id = numpy.argmax(counts)

            else:
                message = '{name}: Unknown decision_making type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=recognizer_params.get_path('decision_making.type')
                )

                self.logger.exception(message)
                raise AssertionError(message)

        return self['class_labels'][classification_result_id]

    def _generate_validation(self, annotations, validation_type='generated_scene_balanced',
                             valid_percentage=0.20, seed=None):
        self.set_seed(seed=seed)
        validation_files = []

        if validation_type == 'generated_scene_balanced':
            # Get training data per scene label
            annotation_data = {}
            for audio_filename in sorted(annotations.keys()):
                scene_label = annotations[audio_filename]['scene_label']
                location_id = annotations[audio_filename]['location_identifier']
                if scene_label not in annotation_data:
                    annotation_data[scene_label] = {}
                if location_id not in annotation_data[scene_label]:
                    annotation_data[scene_label][location_id] = []
                annotation_data[scene_label][location_id].append(audio_filename)

            training_files = []
            validation_amounts = {}

            for scene_label in sorted(annotation_data.keys()):
                validation_amount = []
                sets_candidates = []
                for i in range(0, 1000):
                    current_locations = list(annotation_data[scene_label].keys())
                    random.shuffle(current_locations, random.random)
                    valid_percentage_index = int(numpy.ceil(valid_percentage * len(annotation_data[scene_label])))
                    current_validation_locations = current_locations[0:valid_percentage_index]
                    current_training_locations = current_locations[valid_percentage_index:]

                    # Collect validation files
                    current_validation_files = []
                    for location_id in current_validation_locations:
                        current_validation_files += annotation_data[scene_label][location_id]

                    # Collect training files
                        current_training_files = []
                    for location_id in current_training_locations:
                        current_training_files += annotation_data[scene_label][location_id]

                    validation_amount.append(
                        len(current_validation_files) / float(len(current_validation_files) + len(current_training_files))
                    )

                    sets_candidates.append({
                        'validation': current_validation_files,
                        'training': current_training_files,
                    })

                best_set_id = numpy.argmin(numpy.abs(numpy.array(validation_amount) - valid_percentage))
                validation_files += sets_candidates[best_set_id]['validation']
                training_files += sets_candidates[best_set_id]['training']
                validation_amounts[scene_label] = validation_amount[best_set_id]

            if self.show_extra_debug:
                self.logger.debug('  Validation set statistics')
                self.logger.debug('  {0:<20s} | {1:10s} '.format('Scene label', 'Validation amount (%)'))
                self.logger.debug('  {0:<20s} + {1:10s} '.format('-'*20, '-'*20))

                for scene_label in sorted(validation_amounts.keys()):
                    self.logger.debug('  {0:<20s} | {1:4.2f} '.format(scene_label, validation_amounts[scene_label]*100))
        else:
            message = '{name}: Unknown validation_type [{type}].'.format(
                name=self.__class__.__name__,
                type=validation_type
            )

            self.logger.exception(message)
            raise AssertionError(message)

        return validation_files

    def _accumulate_probabilities(self, probabilities, accumulation_type='sum'):
        accumulated = numpy.ones(len(self.class_labels)) * -numpy.inf
        for row_id in range(0, probabilities.shape[0]):
            if accumulation_type == 'sum':
                accumulated[row_id] = numpy.sum(probabilities[row_id, :])
            elif accumulation_type == 'prod':
                accumulated[row_id] = numpy.prod(probabilities[row_id, :])
            elif accumulation_type == 'mean':
                accumulated[row_id] = numpy.mean(probabilities[row_id, :])
            else:
                message = '{name}: Unknown accumulation type [{type}].'.format(
                    name=self.__class__.__name__,
                    type=accumulation_type
                )

                self.logger.exception(message)
                raise AssertionError(message)

        return accumulated

    def _get_target_matrix_dict(self, data, annotations):
        activity_matrix_dict = {}
        for audio_filename in annotations:
            frame_count = data[audio_filename].feat[0].shape[0]
            pos = self.class_labels.index(annotations[audio_filename]['scene_label'])
            roll = numpy.zeros((frame_count, len(self.class_labels)))
            roll[:, pos] = 1
            activity_matrix_dict[audio_filename] = roll
        return activity_matrix_dict

    def learn(self, data, annotations):
        message = '{name}: Implement learn function.'.format(
            name=self.__class__.__name__
        )

        self.logger.exception(message)
        raise AssertionError(message)


class SceneClassifierGMMdeprecated(SceneClassifier):
    """Scene classifier with GMM"""
    def __init__(self, *args, **kwargs):
        super(SceneClassifierGMMdeprecated, self).__init__(*args, **kwargs)
        self.method = 'gmm_deprecated'

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore", DeprecationWarning)
        from sklearn import mixture

        training_files = annotations.keys()  # Collect training files
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)
        X_training = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training = numpy.vstack([activity_matrix_dict[x] for x in training_files])

        class_progress = tqdm(self['class_labels'],
                              file=sys.stdout,
                              leave=False,
                              desc='           {0:>15s}'.format('Learn '),
                              miniters=1,
                              disable=self.disable_progress_bar
                              )

        for class_id, class_label in enumerate(class_progress):
            if self.log_progress:
                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {class_label:<15s}'.format(
                    title='Learn',
                    item_id=class_id,
                    total=len(self.class_labels),
                    class_label=class_label)
                )

            current_class_data = X_training[Y_training[:, class_id] > 0, :]

            self['model'][class_label] = mixture.GMM(**self.learner_params).fit(current_class_data)

        return self

    def _frame_probabilities(self, feature_data):
        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore", DeprecationWarning)
        from sklearn import mixture

        logls = numpy.ones((len(self['model']), feature_data.shape[0])) * -numpy.inf

        for label_id, label in enumerate(self['class_labels']):
            logls[label_id] = self['model'][label].score(feature_data)

        return logls


class SceneClassifierGMM(SceneClassifier):
    """Scene classifier with GMM"""
    def __init__(self, *args, **kwargs):
        super(SceneClassifierGMM, self).__init__(*args, **kwargs)
        self.method = 'gmm'

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        from sklearn.mixture import GaussianMixture

        training_files = annotations.keys()  # Collect training files
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)
        X_training = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training = numpy.vstack([activity_matrix_dict[x] for x in training_files])

        class_progress = tqdm(self['class_labels'],
                              file=sys.stdout,
                              leave=False,
                              desc='           {0:>15s}'.format('Learn '),
                              miniters=1,
                              disable=self.disable_progress_bar
                              )

        for class_id, class_label in enumerate(class_progress):
            if self.log_progress:
                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {class_label:<15s}'.format(
                    title='Learn',
                    item_id=class_id,
                    total=len(self.class_labels),
                    class_label=class_label)
                )
            current_class_data = X_training[Y_training[:, class_id] > 0, :]

            self['model'][class_label] = GaussianMixture(**self.learner_params).fit(current_class_data)

        return self

    def _frame_probabilities(self, feature_data):
        logls = numpy.ones((len(self['model']), feature_data.shape[0])) * -numpy.inf

        for label_id, label in enumerate(self['class_labels']):
            logls[label_id] = self['model'][label].score(feature_data)

        return logls


class SceneClassifierMLP(SceneClassifier, KerasMixin):
    """Scene classifier with MLP"""
    def __init__(self, *args, **kwargs):
        super(SceneClassifierMLP, self).__init__(*args, **kwargs)
        self.method = 'mlp'

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        training_files = annotations.keys()  # Collect training files
        if self.learner_params.get_path('validation.enable', False):
            validation_files = self._generate_validation(
                annotations=annotations,
                validation_type=self.learner_params.get_path('validation.setup_source'),
                valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                seed=self.learner_params.get_path('validation.seed')
            )
            training_files = list(set(training_files) - set(validation_files))
        else:
            validation_files = []

        # Double check that training and validation files are not overlapping.
        if set(training_files).intersection(validation_files):
            message = '{name}: Training and validation file lists are overlapping!'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)


        # Convert annotations into activity matrix format
        activity_matrix_dict = self._get_target_matrix_dict(data=data, annotations=annotations)

        self.set_seed()

        self._setup_keras()

        with SuppressStdoutAndStderr():
            # Import keras and suppress backend announcement printed to stderr
            import keras

        self.create_model(input_shape=self._get_input_size(data=data))

        if self.show_extra_debug:
            self.log_model_summary()

        X_training = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training = numpy.vstack([activity_matrix_dict[x] for x in training_files])

        class FancyProgbarLogger(keras.callbacks.Callback):
            """Callback that prints metrics to stdout.
            """

            def __init__(self, callbacks=None, queue_length=10, metric=None, disable_progress_bar=False, log_progress=False):
                self.metric = metric
                self.disable_progress_bar = disable_progress_bar
                self.log_progress = log_progress
                self.timer = Timer()

            def on_train_begin(self, logs=None):
                self.logger = logging.getLogger(__name__)
                self.verbose = self.params['verbose']
                self.epochs = self.params['epochs']
                if self.log_progress:
                    self.logger.info('Starting training process')
                self.pbar = tqdm(total=self.epochs,
                                 file=sys.stdout,
                                 desc='           {0:>15s}'.format('Learn (epoch)'),
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar
                                 )

            def on_train_end(self, logs=None):
                self.pbar.close()

            def on_epoch_begin(self, epoch, logs=None):
                if self.log_progress:
                    self.logger.info('  Epoch %d/%d' % (epoch + 1, self.epochs))
                self.seen = 0
                self.timer.start()

            def on_batch_begin(self, batch, logs=None):
                if self.seen < self.params['samples']:
                    self.log_values = []

            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                batch_size = logs.get('size', 0)
                self.seen += batch_size

                for k in self.params['metrics']:
                    if k in logs:
                        self.log_values.append((k, logs[k]))

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                postfix = {
                    'train': None,
                    'validation': None,
                }
                for k in self.params['metrics']:
                    if k in logs:
                        self.log_values.append((k, logs[k]))
                        if self.metric and k.endswith(self.metric):
                            if k.startswith('val_'):
                                postfix['validation'] = '{:4.2f}'.format(logs[k] * 100.0)
                            else:
                                postfix['train'] = '{:4.2f}'.format(logs[k] * 100.0)
                self.timer.stop()
                if self.log_progress:
                    self.logger.info('                train={train}, validation={validation}, time={time}'.format(
                        train=postfix['train'],
                        validation=postfix['validation'],
                        time=self.timer.get_string())
                    )

                self.pbar.set_postfix(postfix)
                self.pbar.update(1)

        # Add model callbacks
        fancy_logger = FancyProgbarLogger(metric=self.learner_params.get_path('model.metrics')[0],
                                          disable_progress_bar=self.disable_progress_bar,
                                          log_progress=self.log_progress)

        # Callback list, always have FancyProgbarLogger
        callbacks = [fancy_logger]

        callback_params = self.learner_params.get_path('training.callbacks', [])
        if callback_params:
            for cp in callback_params:
                if cp['type'] == 'ModelCheckpoint' and not cp['parameters'].get('filepath'):
                    cp['parameters']['filepath'] = os.path.splitext(self.filename)[0] + '.weights.{epoch:02d}-{val_loss:.2f}.hdf5'

                if cp['type'] == 'EarlyStopping' and cp.get('parameters').get('monitor').startswith('val_') and not self.learner_params.get_path('validation.enable', False):
                    message = '{name}: Cannot use callback type [{type}] with monitor parameter [{monitor}] as there is no validation set.'.format(
                        name=self.__class__.__name__,
                        type=cp['type'],
                        monitor=cp.get('parameters').get('monitor')
                    )

                    self.logger.exception(message)
                    raise AttributeError(message)

                try:
                    # Get Callback class
                    CallbackClass = getattr(importlib.import_module("keras.callbacks"), cp['type'])

                    # Add callback to list
                    callbacks.append(CallbackClass(**cp.get('parameters', {})))

                except AttributeError:
                    message = '{name}: Invalid Keras callback type [{type}]'.format(
                        name=self.__class__.__name__,
                        type=cp['type']
                    )

                    self.logger.exception(message)
                    raise AttributeError(message)

        self.set_seed()
        if self.show_extra_debug:
            self.logger.debug('  Training items \t[{examples:d}]'.format(examples=len(X_training)))
        if validation_files:
            X_validation = numpy.vstack([data[x].feat[0] for x in validation_files])
            Y_validation = numpy.vstack([activity_matrix_dict[x] for x in validation_files])
            validation = (X_validation, Y_validation)
            if self.show_extra_debug:
                self.logger.debug('  Validation items \t[{validation:d}]'.format(validation=len(X_validation)))
        else:
            validation = None
        if self.show_extra_debug:
            self.logger.debug('  Batch size \t[{batch:d}]'.format(
                batch=self.learner_params.get_path('training.batch_size', 1))
            )

            self.logger.debug('  Epochs \t\t[{epoch:d}]'.format(
                epoch=self.learner_params.get_path('training.epochs', 1))
            )

        hist = self.model.fit(x=X_training,
                              y=Y_training,
                              batch_size=self.learner_params.get_path('training.batch_size', 1),
                              epochs=self.learner_params.get_path('training.epochs', 1),
                              validation_data=validation,
                              verbose=0,
                              shuffle=self.learner_params.get_path('training.shuffle', True),
                              callbacks=callbacks
                              )
        self['learning_history'] = hist.history

    def _frame_probabilities(self, feature_data):
        return self.model.predict(x=feature_data).T


class EventDetector(LearnerContainer):
    """Event detector (Frame classifier / Multiclass - Multilabel)"""
    @staticmethod
    def _contiguous_regions(activity_array):
        """Find contiguous regions from bool valued numpy.array.
        Transforms boolean values for each frame into pairs of onsets and offsets.
        Parameters
        ----------
        activity_array : numpy.array [shape=(t)]
            Event activity array, bool values
        Returns
        -------
        change_indices : numpy.ndarray [shape=(2, number of found changes)]
            Onset and offset indices pairs in matrix
        """

        # Find the changes in the activity_array
        change_indices = numpy.diff(activity_array).nonzero()[0]

        # Shift change_index with one, focus on frame after the change.
        change_indices += 1

        if activity_array[0]:
            # If the first element of activity_array is True add 0 at the beginning
            change_indices = numpy.r_[0, change_indices]

        if activity_array[-1]:
            # If the last element of activity_array is True, add the length of the array
            change_indices = numpy.r_[change_indices, activity_array.size]

        # Reshape the result into two columns
        return change_indices.reshape((-1, 2))

    def _slide_and_accumulate(self, input_probabilities, window_length, accumulation_type='sliding_sum'):
        # Lets keep the system causal and use look-back while smoothing (accumulating) likelihoods
        output_probabilities = copy.deepcopy(input_probabilities)
        for stop_id in range(0, input_probabilities.shape[0]):
            start_id = stop_id - window_length
            if start_id < 0:
                start_id = 0
            if start_id != stop_id:
                if accumulation_type == 'sliding_sum':
                    output_probabilities[start_id] = numpy.sum(input_probabilities[start_id:stop_id])
                elif accumulation_type == 'sliding_mean':
                    output_probabilities[start_id] = numpy.mean(input_probabilities[start_id:stop_id])
                elif accumulation_type == 'sliding_median':
                    output_probabilities[start_id] = numpy.median(input_probabilities[start_id:stop_id])
                else:
                    message = '{name}: Unknown slide and accumulate type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=accumulation_type
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                output_probabilities[start_id] = input_probabilities[start_id]

        return output_probabilities

    def _activity_processing(self, activity_vector, window_size, processing_type="median_filtering"):
        if processing_type == 'median_filtering':
            return scipy.signal.medfilt(volume=activity_vector, kernel_size=window_size)
        else:
            message = '{name}: Unknown activity processing type [{type}].'.format(
                name=self.__class__.__name__,
                type=processing_type
            )

            self.logger.exception(message)
            raise AssertionError(message)

    def _get_target_matrix_dict(self, data, annotations):

        activity_matrix_dict = {}
        for audio_filename in annotations:
            # Create event roll
            event_roll = EventRoll(metadata_container=annotations[audio_filename],
                                   label_list=self.class_labels,
                                   time_resolution=self.params.get_path('hop_length_seconds')
                                   )
            # Pad event roll to full length of the signal
            activity_matrix_dict[audio_filename] = event_roll.pad(length=data[audio_filename].feat[0].shape[0])

        return activity_matrix_dict

    def _generate_validation(self, annotations, validation_type='generated_scene_location_event_balanced',
                             valid_percentage=0.20, seed=None):

        self.set_seed(seed=seed)
        validation_files = []

        if validation_type == 'generated_scene_location_event_balanced':
            # Get training data per scene label
            annotation_data = {}
            for audio_filename in sorted(annotations.keys()):
                scene_label = annotations[audio_filename][0].scene_label
                location_id = annotations[audio_filename][0].location_identifier
                if scene_label not in annotation_data:
                    annotation_data[scene_label] = {}
                if location_id not in annotation_data[scene_label]:
                    annotation_data[scene_label][location_id] = []
                annotation_data[scene_label][location_id].append(audio_filename)

            # Get event amounts
            event_amounts = {}
            for scene_label in list(annotation_data.keys()):
                if scene_label not in event_amounts:
                    event_amounts[scene_label] = {}
                for location_id in list(annotation_data[scene_label].keys()):
                    for audio_filename in annotation_data[scene_label][location_id]:
                        current_event_amounts = annotations[audio_filename].event_stat_counts()
                        for event_label, count in iteritems(current_event_amounts):
                            if event_label not in event_amounts[scene_label]:
                                event_amounts[scene_label][event_label] = 0
                            event_amounts[scene_label][event_label] += count

            for scene_label in list(annotation_data.keys()):
                # Optimize scene sets separately
                validation_set_candidates = []
                validation_set_MAE = []
                validation_set_event_amounts = []
                training_set_event_amounts = []
                for i in range(0, 1000):
                    location_ids = list(annotation_data[scene_label].keys())
                    random.shuffle(location_ids, random.random)

                    valid_percentage_index = int(numpy.ceil(valid_percentage * len(location_ids)))

                    current_validation_files = []
                    for loc_id in location_ids[0:valid_percentage_index]:
                        current_validation_files += annotation_data[scene_label][loc_id]

                    current_training_files = []
                    for loc_id in location_ids[valid_percentage_index:]:
                        current_training_files += annotation_data[scene_label][loc_id]

                    # event count in training set candidate
                    training_set_event_counts = numpy.zeros(len(event_amounts[scene_label]))
                    for audio_filename in current_training_files:
                        current_event_amounts = annotations[audio_filename].event_stat_counts()
                        for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                            if event_label in current_event_amounts:
                                training_set_event_counts[event_label_id] += current_event_amounts[event_label]

                    # Accept only sets which leave at least one example for training
                    if numpy.all(training_set_event_counts > 0):
                        # event counts in validation set candidate
                        validation_set_event_counts = numpy.zeros(len(event_amounts[scene_label]))
                        for audio_filename in current_validation_files:
                            current_event_amounts = annotations[audio_filename].event_stat_counts()

                            for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                                if event_label in current_event_amounts:
                                    validation_set_event_counts[event_label_id] += current_event_amounts[event_label]

                        # Accept only sets which have examples from each sound event class
                        if numpy.all(validation_set_event_counts > 0):
                            validation_amount = validation_set_event_counts / (validation_set_event_counts + training_set_event_counts)
                            validation_set_candidates.append(current_validation_files)
                            validation_set_MAE.append(mean_absolute_error(numpy.ones(len(validation_amount)) * valid_percentage, validation_amount))
                            validation_set_event_amounts.append(validation_set_event_counts)
                            training_set_event_amounts.append(training_set_event_counts)

                # Generate balance validation set
                # Selection done based on event counts (per scene class)
                # Target count specified percentage of training event count
                if validation_set_MAE:
                    best_set_id = numpy.argmin(validation_set_MAE)
                    validation_files += validation_set_candidates[best_set_id]

                    if self.show_extra_debug:
                        self.logger.debug('  Valid sets found [{sets}]'.format(
                            sets=len(validation_set_MAE))
                        )

                        self.logger.debug('  Best fitting set ID={id}, Error={error:4.2}%'.format(
                            id=best_set_id,
                            error=validation_set_MAE[best_set_id]*100)
                        )

                        self.logger.debug('  Validation event counts in respect of all data:')
                        event_amount_percentages = validation_set_event_amounts[best_set_id] / (validation_set_event_amounts[best_set_id] + training_set_event_amounts[best_set_id])

                        self.logger.debug('  {event:<20s} | {amount:10s} '.format(
                            event='Event label',
                            amount='Validation amount (%)')
                        )

                        self.logger.debug('  {event:<20s} + {amount:10s} '.format(
                            event='-' * 20,
                            amount='-' * 20)
                        )

                        for event_label_id, event_label in enumerate(event_amounts[scene_label]):
                            self.logger.debug('  {event:<20s} | {amount:4.2f} '.format(
                                event=event_label,
                                amount=numpy.round(event_amount_percentages[event_label_id] * 100))
                            )

                else:
                    message = '{name}: Validation setup creation was not successful! Could not find a set with examples for each event class in both training and validation.'.format(
                        name=self.__class__.__name__
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

        elif validation_type == 'generated_event_file_balanced':
            # Get event amounts
            event_amounts = {}
            for audio_filename in sorted(annotations.keys()):
                event_label = annotations[audio_filename][0].event_label
                if event_label not in event_amounts:
                    event_amounts[event_label] = []
                event_amounts[event_label].append(audio_filename)

            if self.show_extra_debug:
                self.logger.debug('  {event_label:<20s} | {amount:20s} '.format(
                    event_label='Event label',
                    amount='Validation amount, files (%)')
                )

                self.logger.debug('  {event_label:<20s} + {amount:20s} '.format(
                    event_label='-' * 20,
                    amount='-' * 20)
                )

            for event_label in sorted(list(event_amounts.keys())):
                files = numpy.array(list(event_amounts[event_label]))
                random.shuffle(files, random.random)
                valid_percentage_index = int(numpy.ceil(valid_percentage * len(files)))
                validation_files += files[0:valid_percentage_index].tolist()

                if self.show_extra_debug:
                    self.logger.debug('  {event_label:<20s} | {amount:4.2f} '.format(
                        event_label=event_label if event_label else '-',
                        amount=valid_percentage_index / float(len(files)) * 100.0)
                    )

            random.shuffle(validation_files, random.random)

        else:
            message = '{name}: Unknown validation_type [{type}].'.format(
                name=self.__class__.__name__,
                type=validation_type
            )

            self.logger.exception(message)
            raise AssertionError(message)

        return validation_files


class EventDetectorGMMdeprecated(EventDetector):
    def __init__(self, *args, **kwargs):
        super(EventDetectorGMMdeprecated, self).__init__(*args, **kwargs)
        self.method = 'gmm_deprecated'

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore", DeprecationWarning)
        from sklearn import mixture

        if not self.params.get_path('hop_length_seconds'):
            message = '{name}: No hop length set.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        class_progress = tqdm(self.class_labels,
                              file=sys.stdout,
                              leave=False,
                              desc='           {0:>15s}'.format('Learn '),
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',  # [{elapsed}<{remaining}, {rate_fmt}]',
                              disable=self.disable_progress_bar
                              )

        # Collect training examples

        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)
        for event_id, event_label in enumerate(class_progress):
            if self.log_progress:
                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {event_label:<15s}'.format(
                    title='Learn',
                    item_id=event_id,
                    total=len(self.class_labels),
                    event_label=event_label)
                )
            data_positive = []
            data_negative = []

            for audio_filename, activity_matrix in iteritems(activity_matrix_dict):

                positive_mask = activity_matrix[:, event_id].astype(bool)
                # Store positive examples
                if any(positive_mask):
                    data_positive.append(data[audio_filename].feat[0][positive_mask, :])

                # Store negative examples
                if any(~positive_mask):
                    data_negative.append(data[audio_filename].feat[0][~positive_mask, :])

            if event_label not in self['model']:
                self['model'][event_label] = {'positive': None, 'negative': None}

            self['model'][event_label]['positive'] = mixture.GMM(**self.learner_params).fit(numpy.concatenate(data_positive))
            self['model'][event_label]['negative'] = mixture.GMM(**self.learner_params).fit(numpy.concatenate(data_negative))

    def _frame_probabilities(self, feature_data, accumulation_window_length_frames=None):
        probabilities = numpy.ones((len(self.class_labels), feature_data.shape[0])) * -numpy.inf
        for event_id, event_label in enumerate(self.class_labels):
            positive = self['model'][event_label]['positive'].score_samples(feature_data.feat[0])[0]
            negative = self['model'][event_label]['negative'].score_samples(feature_data.feat[0])[0]
            if accumulation_window_length_frames:
                positive = self._slide_and_accumulate(input_probabilities=positive, window_length=accumulation_window_length_frames)
                negative = self._slide_and_accumulate(input_probabilities=negative, window_length=accumulation_window_length_frames)
            probabilities[event_id, :] = positive - negative

        return probabilities

    def predict(self, feature_data, recognizer_params=None):
        if recognizer_params is None:
            recognizer_params = {}

        recognizer_params = DottedDict(recognizer_params)

        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore", DeprecationWarning)

        frame_probabilities = self._frame_probabilities(
            feature_data=feature_data,
            accumulation_window_length_frames=recognizer_params.get_path('frame_accumulation.window_length_frames')
        )

        results = []
        for event_id, event_label in enumerate(self.class_labels):
            if recognizer_params.get_path('frame_binarization.enable'):
                if recognizer_params.get_path('frame_binarization.type') == 'global_threshold':
                    event_activity = frame_probabilities[event_id, :] > recognizer_params.get_path('frame_binarization.threshold', 0.0)
                else:
                    message = '{name}: Unknown frame_binarization type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=recognizer_params.get_path('frame_binarization.type')
                    )
                    self.logger.exception(message)
                    raise AssertionError(message)
            else:
                message = '{name}: No frame_binarization enabled.'.format(name=self.__class__.__name__)
                self.logger.exception(message)
                raise AssertionError(message)

            event_segments = self._contiguous_regions(event_activity) * self.params.get_path('hop_length_seconds')
            for event in event_segments:
                results.append(MetaDataItem({'event_onset': event[0], 'event_offset': event[1], 'event_label': event_label}))

        return MetaDataContainer(results)


class EventDetectorGMM(EventDetector):
    def __init__(self, *args, **kwargs):
        super(EventDetectorGMM, self).__init__(*args, **kwargs)
        self.method = 'gmm'

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        from sklearn.mixture import GaussianMixture

        if not self.params.get_path('hop_length_seconds'):
            message = '{name}: No hop length set.'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        class_progress = tqdm(self.class_labels,
                              file=sys.stdout,
                              leave=False,
                              desc='           {0:>15s}'.format('Learn '),
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',  # [{elapsed}<{remaining}, {rate_fmt}]',
                              disable=self.disable_progress_bar
                              )

        # Collect training examples
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

        for event_id, event_label in enumerate(class_progress):
            if self.log_progress:
                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {event_label:<15s}'.format(
                    title='Learn',
                    item_id=event_id,
                    total=len(self.class_labels),
                    event_label=event_label)
                )
            data_positive = []
            data_negative = []

            for audio_filename, activity_matrix in iteritems(activity_matrix_dict):

                positive_mask = activity_matrix[:, event_id].astype(bool)
                # Store positive examples
                if any(positive_mask):
                    data_positive.append(data[audio_filename].feat[0][positive_mask, :])

                # Store negative examples
                if any(~positive_mask):
                    data_negative.append(data[audio_filename].feat[0][~positive_mask, :])

            self['model'][event_label] = {
                'positive': GaussianMixture(**self.learner_params).fit(numpy.concatenate(data_positive)),
                'negative': GaussianMixture(**self.learner_params).fit(numpy.concatenate(data_negative))
            }

    def predict(self, feature_data, recognizer_params=None):

        if recognizer_params is None:
            recognizer_params = {}

        recognizer_params = DottedDict(recognizer_params)

        results = []
        for event_id, event_label in enumerate(self.class_labels):
            # Evaluate positive and negative models
            positive = self['model'][event_label]['positive'].score_samples(feature_data.feat[0])
            negative = self['model'][event_label]['negative'].score_samples(feature_data.feat[0])

            # Accumulate
            if recognizer_params.get_path('frame_accumulation.enable'):
                positive = self._slide_and_accumulate(
                    input_probabilities=positive,
                    window_length=recognizer_params.get_path('frame_accumulation.window_length_frames'),
                    accumulation_type=recognizer_params.get_path('frame_accumulation.type')
                )

                negative = self._slide_and_accumulate(
                    input_probabilities=negative,
                    window_length=recognizer_params.get_path('frame_accumulation.window_length_frames'),
                    accumulation_type=recognizer_params.get_path('frame_accumulation.type')
                )

            # Likelihood ratio
            frame_probabilities = positive - negative

            # Binarization
            if recognizer_params.get_path('frame_binarization.enable'):
                if recognizer_params.get_path('frame_binarization.type') == 'global_threshold':
                    event_activity = frame_probabilities > recognizer_params.get_path('frame_binarization.threshold', 0.0)
                else:
                    message = '{name}: Unknown frame_binarization type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=recognizer_params.get_path('frame_binarization.type')
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                message = '{name}: No frame_binarization enabled.'.format(name=self.__class__.__name__)
                self.logger.exception(message)
                raise AssertionError(message)

            # Get events
            event_segments = self._contiguous_regions(event_activity) * self.params.get_path('hop_length_seconds')

            # Add events
            for event in event_segments:
                results.append(MetaDataItem({'event_onset': event[0],
                                             'event_offset': event[1],
                                             'event_label': event_label}))

        return MetaDataContainer(results)


class EventDetectorMLP(EventDetector, KerasMixin):
    def __init__(self, *args, **kwargs):
        super(EventDetectorMLP, self).__init__(*args, **kwargs)
        self.method = 'mlp'

    def learn(self, data, annotations):
        """Learn based on data ana annotations

        Parameters
        ----------
        data : dict of FeatureContainers
            Feature data
        annotations : dict of MetadataContainers
            Meta data

        Returns
        -------
        self

        """

        # Collect training files
        training_files = annotations.keys()

        # Validation files
        if self.learner_params.get_path('validation.enable', False):
            if self.learner_params.get_path('validation.setup_source').startswith('generated'):
                validation_files = self._generate_validation(annotations=annotations,
                                                             validation_type=self.learner_params.get_path('validation.setup_source', 'generated_scene_event_balanced'),
                                                             valid_percentage=self.learner_params.get_path('validation.validation_amount', 0.20),
                                                             seed=self.learner_params.get_path('validation.seed'),
                                                             )

            training_files = list(set(training_files) - set(validation_files))
        else:
            validation_files = []

        # Double check that training and validation files are not overlapping.
        if set(training_files).intersection(validation_files):
            message = '{name}: Training and validation file lists are overlapping!'.format(
                name=self.__class__.__name__
            )

            self.logger.exception(message)
            raise ValueError(message)

        # Convert annotations into activity matrix format
        activity_matrix_dict = self._get_target_matrix_dict(data, annotations)

        self.set_seed()

        self._setup_keras()

        with SuppressStdoutAndStderr():
            # Import keras and suppress backend announcement printed to stderr
            import keras

        self.create_model(input_shape=self._get_input_size(data=data))

        if self.show_extra_debug:
            self.log_model_summary()

        X_training = numpy.vstack([data[x].feat[0] for x in training_files])
        Y_training = numpy.vstack([activity_matrix_dict[x] for x in training_files])
        #np.savetxt('Real_TrainData_fold1.txt',X_training,delimiter = ',')
        #np.savetxt('Real_TrainLabel_fold1.txt',Y_training,delimiter = ',')
        #exit()

        class_weight = None
        if len(self.class_labels) == 1:
            # Special case with binary classifier
            if self.learner_params.get_path('training.class_weight'):
                class_weight = {}
                for class_id, weight in enumerate(self.learner_params.get_path('training.class_weight')):
                    class_weight[class_id] = float(weight)

            if self.show_extra_debug:
                negative_examples_id = numpy.where(Y_training[:, 0] == 0)[0]
                positive_examples_id = numpy.where(Y_training[:, 0] == 1)[0]

                self.logger.debug('  Positives items \t[{positives:d}]\t({perc:.2f} %)'.format(
                    positives=len(positive_examples_id),
                    perc=len(positive_examples_id)/float(len(positive_examples_id)+len(negative_examples_id))*100
                ))
                self.logger.debug('  Negatives items \t[{negatives:d}]\t({perc:.2f} %)'.format(
                    negatives=len(negative_examples_id),
                    perc=len(negative_examples_id) / float(len(positive_examples_id) + len(negative_examples_id)) * 100
                ))

                self.logger.debug('  Class weights \t[{weights}]\t'.format(weights=class_weight))

        class FancyProgbarLogger(keras.callbacks.Callback):
            """Callback that prints metrics to stdout.
            """

            def __init__(self, callbacks=None, queue_length=10, metric=None, disable_progress_bar=False, log_progress=False):
                if isinstance(metric, str):
                    self.metric = metric
                elif callable(metric):
                    self.metric = metric.__name__
                self.disable_progress_bar = disable_progress_bar
                self.log_progress = log_progress
                self.timer = Timer()

            def on_train_begin(self, logs=None):
                self.logger = logging.getLogger(__name__)
                self.verbose = self.params['verbose']
                self.epochs = self.params['epochs']
                if self.log_progress:
                    self.logger.info('Starting training process')
                self.pbar = tqdm(total=self.epochs,
                                 file=sys.stdout,
                                 desc='           {0:>15s}'.format('Learn (epoch)'),
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar
                                 )

            def on_train_end(self, logs=None):
                self.pbar.close()

            def on_epoch_begin(self, epoch, logs=None):
                if self.log_progress:
                    self.logger.info('  Epoch %d/%d' % (epoch + 1, self.epochs))
                self.seen = 0
                self.timer.start()

            def on_batch_begin(self, batch, logs=None):
                if self.seen < self.params['samples']:
                    self.log_values = []

            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                batch_size = logs.get('size', 0)
                self.seen += batch_size

                for k in self.params['metrics']:
                    if k in logs:
                        self.log_values.append((k, logs[k]))

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                postfix = {
                    'train': None,
                    'validation': None,
                }
                for k in self.params['metrics']:
                    if k in logs:
                        self.log_values.append((k, logs[k]))
                        if self.metric and k.endswith(self.metric):
                            if k.startswith('val_'):
                                postfix['validation'] = '{:4.4f}'.format(logs[k])
                            else:
                                postfix['train'] = '{:4.4f}'.format(logs[k])
                self.timer.stop()
                if self.log_progress:
                    self.logger.info('                train={train}, validation={validation}, time={time}'.format(
                        train=postfix['train'],
                        validation=postfix['validation'],
                        time=self.timer.get_string())
                    )

                self.pbar.set_postfix(postfix)
                self.pbar.update(1)

        # Add model callbacks
        fancy_logger = FancyProgbarLogger(
            metric=self.learner_params.get_path('model.metrics')[0],
            disable_progress_bar=self.disable_progress_bar,
            log_progress=self.log_progress
        )

        callbacks = [fancy_logger]
        callback_params = self.learner_params.get_path('training,callbacks', [])
        if callback_params:
            for cp in callback_params:
                if cp['type'] == 'ModelCheckpoint' and not cp['parameters'].get('filepath'):
                    cp['parameters']['filepath'] = os.path.splitext(self.filename)[0] + '.weights.{epoch:02d}-{val_loss:.2f}.hdf5'

                if cp['type'] == 'EarlyStopping' and cp.get('parameters').get('monitor').startswith('val_') and not self.learner_params.get_path('validation.enable', False):
                    message = '{name}: Cannot use callback type [{type}] with monitor parameter [{monitor}] as there is no validation set.'.format(
                        name=self.__class__.__name__,
                        type=cp['type'],
                        monitor=cp.get('parameters').get('monitor')
                    )

                    self.logger.exception(message)
                    raise AttributeError(message)

                try:
                    CallbackClass = getattr(importlib.import_module("keras.callbacks"), cp['type'])
                    callbacks.append(CallbackClass(**cp.get('parameters', {})))

                except AttributeError:
                    message = '{name}: Invalid Keras callback type [{type}]'.format(
                        name=self.__class__.__name__,
                        type=cp['type']
                    )

                    self.logger.exception(message)
                    raise AttributeError(message)

        self.set_seed()
        if self.show_extra_debug:
            self.logger.debug('  Training items \t[{examples:d}]'.format(examples=len(X_training)))

        if validation_files:
            X_validation = numpy.vstack([data[x].feat[0] for x in validation_files])
            Y_validation = numpy.vstack([activity_matrix_dict[x] for x in validation_files])
            validation = (X_validation, Y_validation)
            #np.savetxt('Real_TestData_fold1.txt',X_validation,delimiter=',')
            #np.savetxt('Real_TestLabel_fold1.txt',Y_validation,delimiter=',')
            #exit()

            if self.show_extra_debug:
                self.logger.debug('  Validation items \t[{validation:d}]'.format(validation=len(X_validation)))
        else:
            validation = None

        #embed()#
        #
        #import matplotlib.pyplot as plt
        #plt.plot(Y_training)
        #plt.show()

        if self.show_extra_debug:
            self.logger.debug('  Feature vector \t[{vector:d}]'.format(vector=self._get_input_size(data=data)))
            self.logger.debug('  Batch size \t[{batch:d}]'.format(batch=self.learner_params.get_path('training.batch_size', 1)))
            self.logger.debug('  Epochs \t\t[{epoch:d}]'.format(epoch=self.learner_params.get_path('training.epochs', 1)))

        hist = self.model.fit(
            x=X_training,
            y=Y_training,
            batch_size=self.learner_params.get_path('training.batch_size', 1),
            epochs=self.learner_params.get_path('training.epochs', 1),
            validation_data=validation,
            verbose=0,
            shuffle=self.learner_params.get_path('training.shuffle', True),
            callbacks=callbacks,
            class_weight=class_weight
        )

        self['learning_history'] = hist.history
    def predict(self, feature_data,recognizer_params=None):

        if recognizer_params is None:
            recognizer_params = {}

        recognizer_params = DottedDict(recognizer_params)

        if isinstance(feature_data, FeatureContainer):
            feature_data = feature_data.feat[0]
        frame_probabilities = self.model.predict(x=feature_data).T
    
        if recognizer_params.get_path('frame_accumulation.enable'):
            for event_id, event_label in enumerate(self.class_labels):
                frame_probabilities[event_id, :] = self._slide_and_accumulate(
                    input_probabilities=frame_probabilities[event_id, :],
                    window_length=recognizer_params.get_path('frame_accumulation.window_length_frames'),
                    accumulation_type=recognizer_params.get_path('frame_accumulation.type'),
                )


        results = []
        for event_id, event_label in enumerate(self.class_labels):
            if recognizer_params.get_path('frame_binarization.enable'):
                if recognizer_params.get_path('frame_binarization.type') == 'global_threshold':
                    event_activity = frame_probabilities[event_id, :] > recognizer_params.get_path('frame_binarization.threshold', 0.5)
                else:
                    message = '{name}: Unknown frame_binarization type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=recognizer_params.get_path('frame_binarization.type')
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                message = '{name}: No frame_binarization enabled.'.format(
                    name=self.__class__.__name__
                )

                self.logger.exception(message)
                raise AssertionError(message)

            if recognizer_params.get_path('event_activity_processing.enable'):
                event_activity = self._activity_processing(activity_vector=event_activity,
                                                           window_size=recognizer_params.get_path('event_activity_processing.window_length_frames'))

            event_segments = self._contiguous_regions(event_activity) * self.params.get_path('hop_length_seconds')
            for event in event_segments:
                results.append(MetaDataItem({'event_onset': event[0], 'event_offset': event[1], 'event_label': event_label}))

        return MetaDataContainer(results)
    def predict_UseOutsideModel(self, feature_data,fold,item,recognizer_params=None):

        if recognizer_params is None:
            recognizer_params = {}

        recognizer_params = DottedDict(recognizer_params)

        if isinstance(feature_data, FeatureContainer):
            feature_data = feature_data.feat[0]     
        frame_probabilities = self.model.predict(x=feature_data).T         
        ########################### Use the outside model to predict the probabilities below 

        ##### Note: the number of switch should be equal to 1 or 0 (the original DCASE Challenge 2017 Task 3)
        ### DNN-S evaluation switch
        UseMLPJointSoft = 1
        ### RNN-S evaluation switch
        UseRNNJointSoft = 0

        UseDistSchem    = 0             
        UseCNNModel     = 0
        UseMLPModel     = 0
        UseMLPJointHard = 0 
        UseCNNJointHard = 0
        UseCNNJointSoft = 0

        if(UseCNNModel):            
            Modelname = 'H:/My Documents/Tools/RandomForestRegression/RFCForDCASE2017/CNN_Folder/Multi-Label-Image-Classification-master/Model_CNN_Classification/model_fold' + str(fold) +'.json'
            WeightName = 'H:/My Documents/Tools/RandomForestRegression/RFCForDCASE2017/CNN_Folder/Multi-Label-Image-Classification-master/Model_CNN_Classification/model_fold' + str(fold) +'.h5'                        
            json_file = open(Modelname, 'r')
            loaded_model_json = json_file.read()
            json_file.close()            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(WeightName)    
            x_test = feature_data[:,0:DatDim]            
            x_test = AggreData(x_test,AggreNum = AggreNum)   
            x_test = x_test.reshape(x_test.shape[0],1, AggreNum, DatDim)                                  
            out_c = loaded_model.predict_proba(x_test)                                
            Modelname = 'H:/My Documents/Tools/RandomForestRegression/RFCForDCASE2017/CNN_Folder/Multi-Label-Image-Classification-master/Model_CNN_Regression/model_fold' + str(fold) +'.json'
            WeightName = 'H:/My Documents/Tools/RandomForestRegression/RFCForDCASE2017/CNN_Folder/Multi-Label-Image-Classification-master/Model_CNN_Regression/model_fold' + str(fold) +'.h5'            
            json_file = open(Modelname, 'r')
            loaded_model_json = json_file.read()
            json_file.close()            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(WeightName)    
            x_test = feature_data[:,0:DatDim]            
            x_test = AggreData(x_test,AggreNum = AggreNum)   
            x_test = x_test.reshape(x_test.shape[0],1, AggreNum, DatDim)                                  
            out_r = loaded_model.predict_proba(x_test)                        
            frame_probabilities = 0.5*out_c[:,0:6].T + 0.5*out_r[:,0:6].T
        if(UseMLPModel):
            Modelname = 'H:/My Documents/Tools/RandomForestRegression/Attention_Training/MLP_regression/model_fold' + str(fold) +'.json'
            WeightName = 'H:/My Documents/Tools/RandomForestRegression/Attention_Training/MLP_regression/model_fold' + str(fold) +'.h5'                      
            json_file = open(Modelname, 'r')
            loaded_model_json = json_file.read()
            json_file.close()            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(WeightName)    
            x_test = feature_data[:,0:DatDim]
            x_test = AggreData(x_test,AggreNum = 10)   
            x_test = (x_test + 4.1408)/(7.7166+4.1408)                                           
            out_r = loaded_model.predict(x_test)  #### out_r is 12031*6 dimension                                 
            #frame_probabilities = 0*out_c[:,0:6].T + 1*out_r[:,0:6].T
            frame_probabilities = 1*out_r[:,0:6].T 
        '''
        if(UseAttentionMLP): 
            ##### load the attention model       
            Modelname = 'H:/My Documents/Tools/RandomForestRegression/Attention_Training/MLP_regression/model_fold0_attention_regression_05_080_useFeatureImportance.json'
            WeightName = 'H:/My Documents/Tools/RandomForestRegression/Attention_Training/MLP_regression/model_fold0_attention_regression_05_080_useFeatureImportance.h5'                      
            json_file = open(Modelname, 'r')
            loaded_model_json = json_file.read()
            json_file.close()            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(WeightName)                
            x_test = feature_data[:,0:DatDim]
            x_test = AggreData(x_test,AggreNum = 10) 
            
            x_test = (x_test + 4.1408)/(7.7166+4.1408)  
            x_test_backup = x_test
            attention_vector = get_activations(loaded_model,x_test,print_shape_only=True,layer_name='attention_vec')[0]
            x_test = np.hstack((x_test_backup,attention_vector)) 
        '''
        if(UseRNNJointSoft):                  
            DatDim = 40
            AggreNum = 10
            input_length = AggreNum
            input_dim = DatDim
            output_length = 1
            output_dim = 6  ## number of event classes
            hidden_dim = 150
            loaded_model = AttentionSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=1)
            loaded_model.load_weights('RNN-S.h5')  
            print('--- loading model successfullly -----------------')
            x_test = feature_data[:,0:DatDim]
            x_test = AggreData(x_test,AggreNum = AggreNum)  
            ### normalization operation           
            x_test = (x_test + 4.1408)/(7.7166+4.1408)  ###  min-max normalization
            print(x_test.shape) 
            x_test = x_test.reshape(x_test.shape[0],AggreNum,DatDim)
            out_r = loaded_model.predict(x_test)  #### out_r is 12031*6 dimension 
            print(out_r.shape)                
            #frame_probabilities = 0*out_c[:,0:6].T + 1*out_r[:,0:6].T
            frame_probabilities = 1*out_r[:,0,0:6].T
        if(UseMLPJointHard):
            Modelname = 'H:/My Documents/Tools/RandomForestRegression/Attention_Training/MLP_regression/model_fold' + str(fold) +'.json'
            WeightName = 'H:/My Documents/Tools/RandomForestRegression/Attention_Training/MLP_regression/model_fold' + str(fold) +'.h5'            
            json_file = open(Modelname, 'r')
            loaded_model_json = json_file.read()
            json_file.close()     
            DatDim = 40
            AggreNum = 30       
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(WeightName)    
            x_test = feature_data[:,0:DatDim]
            x_test = AggreData(x_test,AggreNum = AggreNum)                                
            out = loaded_model.predict(x_test)  #### out is 12031*12 dimension                        
            frame_probabilities = 0*out[:,0:6].T + 1*out[:,6:12].T 
        if(UseMLPJointSoft):
            Modelname = 'DNN-S.json'
            WeightName = 'DNN-S.h5'            
            json_file = open(Modelname, 'r')
            loaded_model_json = json_file.read()
            json_file.close()    
            DatDim = 40     
            AggreNum = 30   
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(WeightName)    
            x_test = feature_data[:,0:DatDim]
            x_test = AggreData(x_test,AggreNum = AggreNum)  
            out = loaded_model.predict([x_test,x_test])  #### out is 12031*12 dimension                        
            frame_probabilities = 0*out[:,0:6].T + 1*out[:,6:12].T             
        if(UseCNNJointHard):
            
            DatDim = 40
            AggreNum = 30   
            Modelname = 'H:/My Documents/Tools/RandomForestRegression/RFCForDCASE2017/CNN_Folder/Multi-Label-Image-Classification-master/Model_CNN_JointHard/model_fold' + str(fold) +'.json'
            WeightName = 'H:/My Documents/Tools/RandomForestRegression/RFCForDCASE2017/CNN_Folder/Multi-Label-Image-Classification-master/Model_CNN_JointHard/model_fold' + str(fold) +'.h5'            
            json_file = open(Modelname, 'r')
            loaded_model_json = json_file.read()
            json_file.close()            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(WeightName)    
            x_test = feature_data[:,0:DatDim]            
            x_test = AggreData(x_test,AggreNum = AggreNum)   
            x_test = x_test.reshape(x_test.shape[0],1, AggreNum, DatDim)                              
            out = loaded_model.predict([x_test,x_test])  #### out is 12031*12 dimension                        
            frame_probabilities = 0*out[:,0:6].T + 1*out[:,6:12].T             
        if(UseDistSchem):
            Meanfile = 'H:/My Documents/Tools/RandomForestRegression/RFCForDCASE2017/CNN_Folder/Multi-Label-Image-Classification-master/TrainData_Mel/ClassMean.txt'
            Varfile = 'H:/My Documents/Tools/RandomForestRegression/RFCForDCASE2017/CNN_Folder/Multi-Label-Image-Classification-master/TrainData_Mel/ClassVar.txt'
            mean = extract_data_raw_test(Meanfile)
            var = extract_data_raw_test(Varfile)
            for t in range(frame_probabilities.shape[1]):
                Dist = np.zeros((ClassNum))
                Prob = np.zeros((ClassNum))
                for c in range (ClassNum):
                    Dist[c] = CalculateDist(data = feature_data[t,0:DatDim],mean=mean[:,c],var=var[:,c])
                    Prob[c] = np.exp(-Dist[c])
                    frame_probabilities[c][t] = frame_probabilities[c][t]*0.9 + 0.1*Prob[c]            
        ######################  Use the outside model to predict the probabilities above            
############################################
        if recognizer_params.get_path('frame_accumulation.enable'):
            for event_id, event_label in enumerate(self.class_labels):
                frame_probabilities[event_id, :] = self._slide_and_accumulate(
                    input_probabilities=frame_probabilities[event_id, :],
                    window_length=recognizer_params.get_path('frame_accumulation.window_length_frames'),
                    accumulation_type=recognizer_params.get_path('frame_accumulation.type'),
                )


        results = []
        for event_id, event_label in enumerate(self.class_labels):
            if recognizer_params.get_path('frame_binarization.enable'):
                if recognizer_params.get_path('frame_binarization.type') == 'global_threshold':                    
                    if((UseCNNModel)|(UseCNNJointHard)|(UseCNNJointSoft)):
                        thre = 0.5
                    else:
                        thre = recognizer_params.get_path('frame_binarization.threshold', 0.5)                     
                    print(thre)
                    event_activity = frame_probabilities[event_id, :] > thre                    
                else:
                    message = '{name}: Unknown frame_binarization type [{type}].'.format(
                        name=self.__class__.__name__,
                        type=recognizer_params.get_path('frame_binarization.type')
                    )

                    self.logger.exception(message)
                    raise AssertionError(message)

            else:
                message = '{name}: No frame_binarization enabled.'.format(
                    name=self.__class__.__name__
                )

                self.logger.exception(message)
                raise AssertionError(message)

            if recognizer_params.get_path('event_activity_processing.enable'):
                event_activity = self._activity_processing(activity_vector=event_activity,
                                                           window_size=recognizer_params.get_path('event_activity_processing.window_length_frames'))

            event_segments = self._contiguous_regions(event_activity) * self.params.get_path('hop_length_seconds')
            for event in event_segments:
                results.append(MetaDataItem({'event_onset': event[0], 'event_offset': event[1], 'event_label': event_label}))

        return MetaDataContainer(results)

