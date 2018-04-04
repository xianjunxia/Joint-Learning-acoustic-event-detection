#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Application core
================

An application core is a utility class to handle all "business logic" for application.
It handles communication between :func:`dcase_framework.datasets.Dataset`,
:func:`dcase_framework.features.FeatureExtractor`,
:func:`dcase_framework.features.FeatureNormalizer`, and
:func:`dcase_framework.learners.LearnerContainer` classes.


Usage examples:

.. code-block:: python
    :linenos:

    params = ParameterContainer(filename='parameters.yaml').load().process()

    app = AcousticSceneClassificationAppCore(name='Test system',
                                             params=params,
                                             setup_label='Development setup'
                                             )
    # Show all datasets available
    app.show_dataset_list()

    # Show parameters
    app.show_parameters()

    # Initialize application
    app.initialize()

    # Extract features
    app.feature_extraction()

    # Normalize features
    app.feature_normalization()

    # Train system
    app.system_training()

    # Test system
    app.system_testing()

    # Evaluate system
    app.system_evaluation()

AcousticSceneClassificationAppCore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Application core for acoustic scene classification applications.

.. autosummary::
    :toctree: generated/

    AcousticSceneClassificationAppCore
    AcousticSceneClassificationAppCore.show_dataset_list
    AcousticSceneClassificationAppCore.show_parameters
    AcousticSceneClassificationAppCore.initialize
    AcousticSceneClassificationAppCore.feature_extraction
    AcousticSceneClassificationAppCore.feature_normalization
    AcousticSceneClassificationAppCore.system_training
    AcousticSceneClassificationAppCore.system_testing
    AcousticSceneClassificationAppCore.system_evaluation

SoundEventAppCore
^^^^^^^^^^^^^^^^^

Application core for sound event detection applications.

.. autosummary::
    :toctree: generated/

    SoundEventAppCore
    SoundEventAppCore.show_dataset_list
    SoundEventAppCore.show_parameters
    SoundEventAppCore.initialize
    SoundEventAppCore.feature_extraction
    SoundEventAppCore.feature_normalization
    SoundEventAppCore.system_training
    SoundEventAppCore.system_testing
    SoundEventAppCore.system_evaluation

BinarySoundEventAppCore
^^^^^^^^^^^^^^^^^^^^^^^

Application core for binary sound event detection applications.

.. autosummary::
    :toctree: generated/

    BinarySoundEventAppCore
    BinarySoundEventAppCore.show_dataset_list
    BinarySoundEventAppCore.show_parameters
    BinarySoundEventAppCore.initialize
    BinarySoundEventAppCore.feature_extraction
    BinarySoundEventAppCore.feature_normalization
    BinarySoundEventAppCore.system_training
    BinarySoundEventAppCore.system_testing
    BinarySoundEventAppCore.system_evaluation

AppCore -- base class
^^^^^^^^^^^^^^^^^^^^^

Base class for all application cores.

.. autosummary::
    :toctree: generated/

    AppCore
    AppCore.show_dataset_list
    AppCore.show_parameters
    AppCore.initialize
    AppCore.feature_extraction
    AppCore.feature_normalization
    AppCore.system_training
    AppCore.system_testing
    AppCore.system_evaluation

"""
from __future__ import print_function, absolute_import
from six import iteritems

import pickle
import sys
import os
import logging
import sed_eval
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from .containers import DottedDict
from .learners import EventDetectorMLP
from .files import ParameterFile
from .features import FeatureContainer, FeatureRepository, FeatureExtractor, FeatureNormalizer, \
    FeatureStacker, FeatureAggregator, FeatureMasker
from .datasets import *
from .utils import filelist_exists, Timer
from .decorators import before_and_after_function_wrapper
from .learners import *
from .metadata import MetaDataContainer, MetaDataItem
from .ui import FancyLogger
from .utils import get_class_inheritors
from .parameters import ParameterContainer
from .files import ParameterFile
from IPython import embed
import librosa
import sklearn


def extract_data(filename_feature,DatDim):

    # Arrays to hold the labels and feature vectors.    
    labels = []
    fvecs = []
    fvecl = []
    file = open(filename_feature)
    i = 0
    for line in file:
        row = line.split(",")
        i = i + 1
        fvecs.append([float(x) for x in row[0:DatDim*DatDim]])
        #fvecl.append([float(x) for x in row[DatDim:DatDim+ClassNum]])        
    fvecs_np = np.array(fvecs).astype(np.float32)     
    return fvecs_np
def get_CSVfile(cpicklefile,csvfile):     
    fvecs = []  
    feature_container = FeatureContainer(filename=cpicklefile)   
    length = feature_container.shape 
    fvecs = numpy.ones((feature_container.shape[0],feature_container.shape[1]))
    for i in range(feature_container.shape[0]):
        for j in range(feature_container.shape[1]):
            fvecs[i][j] = feature_container.feat[0][i][j]       
    print('Changing features successfully')  
    print(fvecs.shape) 
        
    np.savetxt(csvfile,fvecs,delimiter = ',')
def getCQCC_Combined(filename):        
    fvecs = []
    CqccShape = 60
    csvfile = filename + '.bn'
    picklename = filename + '.cpickle'
    print(picklename)
    feature_container = FeatureContainer(filename=picklename)
    '''
    file = open(csvfile)
    i = 0
    for line in file:
        i = i+1
    '''
    fvecs = numpy.zeros((feature_container.feat[0].shape[0],40+CqccShape))
    ################################ get the cqcc acoustic features
    fvecs_temp = numpy.zeros((feature_container.feat[0].shape[0],40+CqccShape))
    i = 0
    file = open(csvfile)
    for line in file:
        if(i>feature_container.feat[0].shape[0]):
            break
        row = line.split(",")
        for j in range(CqccShape):
            fvecs_temp[i][j] = row[j]            
            j = j+1
        i = i + 1    

    ################################ get the cqcc plus the mel cepstrum acoustic features
    file = open(csvfile)
    i = 0
    for line in file:        
        j = 0
        if(i>feature_container.feat[0].shape[0]):
            break        
        for j in range(CqccShape):
            fvecs[i][j] = fvecs_temp[i][j]  
        for j in range(40):
            fvecs[i][CqccShape+j] = feature_container.feat[0][i][j]  
            j = j+1
        i= i+1 
    print('Changing features successfully') 
    print(fvecs.shape)    
    return fvecs
def getCQCC_Separate(filename): 
    CqccShape = 60
    fvecs = []
    csvfile = filename + '.bn'   
    picklename = filename + '.cpickle'
    print(picklename)
    feature_container = FeatureContainer(filename=picklename)
    length = feature_container.shape[0]
    '''
    file = open(csvfile)
    i = 0
    for line in file:
        i = i+1
    '''
    fvecs = numpy.ones((length,CqccShape))
    i = 0
    file = open(csvfile)
    for line in file:
        #if(i>=length):
        #    break            
        row = line.split(",")
        for j in range(CqccShape):
            fvecs[i][j] = row[j]            
            j = j+1
        i= i+1 
    print('Changing into separate features successfully')    
    print(fvecs.shape)    
    return fvecs
class AppCore(object):

    valid_dataset_group = 'acoustic scene'

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        name : str
            Application name
            Default value "Application"
        system_desc : str
            System description
            Default value "None"
        system_parameter_set_id : str
            System parameter set id
            Default value "None"
        setup_label : str
            Application setup label
            Default value "System"
        params : ParameterContainer
            Parameter container containing all parameters needed by application.
        dataset : str or class
            Dataset, if none given dataset name is taken from parameters "dataset->parameters->name"
            Default value "none"
        dataset_evaluation_mode : str
            Dataset evaluation mode, "full" or "folds". If none given, taken from parameters "dataset->parameter->evaluation_mode"
            Default value "none"
        show_progress_in_console : bool
            Show progress in console.
            Default value "True"
        log_system_progress : bool
            Show progress in log.
            Default value "False"
        logger : logging
            Instance of logging
            Default value "none"
        Datasets : dict of Dataset classes
            Dict of datasets available for application. Dict key is name of the dataset and value link to class inherited from Dataset base class. Given dict is used to update internal dict.
            Default value "none"
        FeatureExtractor : class inherited from FeatureExtractor
            Feature extractor class. Use this to override default class.
            Default value "FeatureExtractor"
        FeatureNormalizer : class inherited from FeatureNormalizer
            Feature normalizer class. Use this to override default class.
            Default value "FeatureNormalizer"
        FeatureMasker : class inherited from FeatureMasker
            Feature masker class. Use this to override default class.
            Default value "FeatureMasker"
        FeatureContainer : class inherited from FeatureContainer
            Feature container class. Use this to override default class.
            Default value "FeatureContainer"
        FeatureStacker : class inherited from FeatureStacker
            Feature stacker class. Use this to override default class.
            Default value "FeatureStacker"
        FeatureAggregator : class inherited from FeatureAggregator
            Feature aggregate class. Use this to override default class.
            Default value "FeatureAggregator"
        Learners: dict of Learner classes
            Dict of learners available for application. Dict key is method the class implements and value link to class inherited from LearnerContainer base class. Given dict is used to update internal dict.
        ui : class inherited from FancyLogger
            Output formatter class. Use this to override default class.
            Default value "FancyLogger"
        Raises
        ------
        ValueError:
            No valid ParameterContainer given.

        """

        self.logger = kwargs.get('logger', logging.getLogger(__name__))


        # Fetch all datasets
        self.Datasets = {}

        for dataset_item in get_class_inheritors(Dataset):
            self.Datasets[dataset_item.__name__] = dataset_item

        if kwargs.get('Datasets'):
            self.Datasets.update(kwargs.get('Datasets'))

        # Store classes to allow override in the inherited classes
        self.FeatureExtractor = kwargs.get('FeatureExtractor', FeatureExtractor)
        self.FeatureNormalizer = kwargs.get('FeatureNormalizer', FeatureNormalizer)       
        self.FeatureMasker = kwargs.get('FeatureMasker', FeatureMasker)
        self.FeatureContainer = kwargs.get('FeatureContainer', FeatureContainer)
        self.FeatureStacker = kwargs.get('FeatureStacker', FeatureStacker)
        self.FeatureAggregator = kwargs.get('FeatureAggregator', FeatureAggregator)


        # Fetch all learners
        self.Learners = {}
        learner_list = get_class_inheritors(LearnerContainer)
        for learner_item in learner_list:
            #learner = learner_item()
            learner = learner_item
            
            if learner.method:
                self.Learners[learner.method] = learner_item

        
        if kwargs.get('Learners'):
            self.Learners.update(kwargs.get('Learners'))

        # Parameters
        self.params = kwargs.get('params')        
        if not self.params or not isinstance(self.params, ParameterContainer):
            # Parameters are empty or not ParameterContainer
            message = '{name}: No valid ParameterContainer given.'.format(
                name=self.__class__.__name__
            )
            self.logger.exception(message)
            raise ValueError(message)
        if not self.params.processed:
            # Process parameters
            self.params.process()

        # Set current dataset
        self.dataset = self._get_dataset(dataset=kwargs.get('dataset'))

        self.name = kwargs.get('name', 'Application')
        self.system_desc = kwargs.get('system_desc')        
        self.system_parameter_set_id = kwargs.get('system_parameter_set_id')
        self.setup_label = kwargs.get('setup_label', 'System')
        if kwargs.get('dataset_evaluation_mode'):
            self.dataset_evaluation_mode = kwargs.get('dataset_evaluation_mode')
        else:
            self.dataset_evaluation_mode = self.params.get_path('dataset.parameters.evaluation_mode', 'folds')

        self.disable_progress_bar = not kwargs.get('show_progress_in_console', True)
        self.log_system_progress = kwargs.get('log_system_progress', False)

        # Timer class
        self.timer = Timer()

        # UI
        self.ui = kwargs.get('ui', FancyLogger())

        # Log application title
        self.ui.title(self.name)
        self.ui.line()


    def _get_dataset(self, dataset=None):
        """Get dataset class

        Parameters
        ----------
        dataset : class or str
            Depending variable type:

            - If str, variable is used to index self.Datasets and initialize class
            - If instance of Dataset, used as such
            - If none given, dataset name taken from parameters ( 'dataset.parameters.name' )

            Default value "Non"

        Returns
        -------
        class
            Dataset class instance

        """

        # Get dataset container class     
        if not dataset:
            # If not dataset name given, dig name from parameters and use it.
            dataset_class_name = self.params.get_path('dataset.parameters.name')
            if dataset_class_name and dataset_class_name in self.Datasets:
                return self.Datasets[dataset_class_name](data_path=self.params.get_path('path.data'),
                                                         **self.params.get_path('dataset.parameters'))
            else:
                message = '{name}: No valid dataset given [{dataset}]'.format(
                    name=self.__class__.__name__,
                    dataset=dataset_class_name
                )

                self.logger.exception(message)
                raise ValueError(message)

        elif isinstance(dataset, str):
            return self.Datasets[dataset](data_path=self.params.get_path('path.data'),
                                          **self.params.get_path('dataset.parameters'))
        elif isinstance(dataset, Dataset):
            return dataset
        else:           
            message = '{name}: No valid dataset given [{dataset}]'.format(
                name=self.__class__.__name__,
                dataset=dataset
            )
            self.logger.exception(message)
            raise ValueError(message)

    def _get_learner(self, *args, **kwargs):
        method = kwargs.get('method',None)  
        #print(method)      
        for method in self.Learners:
            print(method)
        #print(self.Learners[method])
        if method in self.Learners:
            print('******************************')
            #print(self.Learners[method](*args, **kwargs))
            return self.Learners[method](*args, **kwargs)
        else:
            message = '{name}: No valid learner method given [{method}]'.format(
                name=self.__class__.__name__,
                method=method
            )

            self.logger.exception(message)
            raise ValueError(message)
            

    def _get_active_folds(self):
        """Get active folds

        Takes intersection of the folds provided by the dataset class and folds marks in
        parameters ('dataset.parameters.fold_list')

        Returns
        -------
        list
            List of fold ids

        """

        folds = self.dataset.folds(mode=self.dataset_evaluation_mode)
        active_fold_list = self.params.get_path('dataset.parameters.fold_list')
        if active_fold_list:
            folds = list(set(folds).intersection(active_fold_list))
        return folds
        

    def show_parameter_set_list(self, set_list):
        """List of datasets available

        Parameters
        ----------
        set_list : list of parameter dicts

        Returns
        -------
        None


        """
        output = ''
        output += '  Parameter set list\n'
        output += '  {set_name:<25s} | {description:80s} |\n'.format(set_name='Set Name',
                                                                     description='Description')
        output += '  {set_name:<25s} + {description:80s} +\n'.format(set_name='-' * 25,
                                                                     description='-'*80)

        for set_item in set_list:
            set_name = set_item.get('set_id')
            description = str(set_item.get('description')) if set_item.get('description') else ''
            output += '  {set_name:<25s} | {description:80s} |\n'.format(set_name=set_name,
                                                                         description=description)

        self.ui.line(output)
        

    def show_dataset_list(self):
        """List of datasets available

        Parameters
        ----------

        Returns
        -------
        None

        """

        output = ''
        output += '  Dataset list\n'
        output += '  {class_name:<45s} | {group:20s} | {valid:5s} | {files:10s} |\n'.format(class_name='Class Name',
                                                                                            group='Group',
                                                                                            valid='Valid',
                                                                                            files='Files')
        output += '  {class_name:<45s} + {group:20s} + {valid:5s} + {files:10s} +\n'.format(class_name='-' * 45,
                                                                                            group='-' * 20,
                                                                                            valid='-' * 5,
                                                                                            files='-' * 10)

        def get_row(data):
            file_count = 0
            if data.meta_container.exists():
                file_count = len(data.meta)

            return '  {class_name:<45s} | {group:20s} | {valid:5s} | {files:10s} |\n'.format(
                class_name=data.__class__.__name__,
                group=data.dataset_group,
                valid='Yes' if data.check_filelist() else 'No',
                files=str(file_count) if file_count else ''
                )

        for dataset_class_name, dataset_class in iteritems(self.Datasets):
            d = dataset_class(data_path=self.params.get_path('path.data'))
            output += get_row(d)

        self.ui.line(output)

    def show_parameters(self):
        """Show parameters"""
        self.ui.debug(str(self.params))

    def show_eval(self):
        pass

    def _before_initialize(self, *args, **kwargs):
        self.ui.section_header('Initialize [{setup_label}][{dataset_evaluation_mode}]'.format(
            setup_label=self.setup_label,
            dataset_evaluation_mode=self.dataset_evaluation_mode)
        )
        self.timer.start()

    def _after_initialize(self, to_return=None):
        self.ui.foot(time=self.timer.stop().get_string())

    @before_and_after_function_wrapper
    def initialize(self):    
        """Initialize application"""
        self.ui.data(field='System')
        if self.name:
            self.ui.data(field='Name',
                         value=self.name,
                         indent=4)

        if self.system_desc:
            self.ui.data(field='Description',
                         value=self.system_desc,
                         indent=4)

        if self.system_parameter_set_id:
            self.ui.data(field='Parameter set',
                         value=self.system_parameter_set_id,
                         indent=4)

        self.ui.data(field='Dataset')
        if self.dataset.storage_name:
            self.ui.data(field='Name',
                         value=self.dataset.storage_name,
                         indent=4)

        if self._get_active_folds():
            self.ui.data(field='Active folds',
                         value=str(self._get_active_folds()),
                         indent=4)

        if self.params.get_path('recognizer.enable') and self.params.get_path('general.challenge_submission_mode'):
            self.ui.data(field='Recognizer')
            self.ui.data(field='Save path',
                         value=os.path.relpath(self.params.get_path('path.recognizer_challenge_output')),
                         indent=4)

        if self.params.get_path('evaluator.enable') and self.params.get_path('evaluator.saving.enable'):
            self.ui.data(field='Evaluator')
            self.ui.data(field='Save path',
                         value=os.path.relpath(self.params.get_path('path.evaluator')),
                         indent=4)

        self.dataset.initialize()

    def _before_feature_extraction(self, *args, **kwargs):
        self.ui.section_header('Feature extractor')
        self.timer.start()

    def _after_feature_extraction(self, to_return=None):
        self.ui.foot(time=self.timer.stop().get_string(), item_count=len(to_return))

    @before_and_after_function_wrapper
    def feature_extraction(self, files=None, overwrite=None):          
        """Feature extraction stage

        Parameters
        ----------
        files : list
            file list

        overwrite : bool
            overwrite existing feature files
            (Default value=False)

        Returns
        -------
        None
        """
    

        
        file = open('C:\DCASE2017-debug\dcase_framework\lst_cqcc.lst')
        #feature_container = FeatureContainer(filename = 'C:/DCASE2017-debug/applications/system/task3/feature_normalizer/feature_extractor_f25a1b3722bd922a/fold0/scale_fold0_street.cpickle')
        #print(feature_container['mean'])
        #exit(0)
        
        for line in file:
            print('\n************************\n')
            print(line)
            line.strip('\n')
            name = 'sequence_' + line[49:53]
            name1 = line[0:53]
            print(name)
            mfccfilename = name1 + '.cpickle'                     
            PureData = 'C:/DCASE2017-debug/dcase_framework/CQCC/' + name + '.csv'            
            #PurePickle = 'C:/Users/21799506/Annaconda2/DCASE2017_Task3-master/Fe_dev/Mel/street/' + name + '.f'            
            get_CSVfile(mfccfilename,PureData)
            
            
            

            '''
            name = line[0:53]
            mfccfilename = name + '.cpickle'            
            cqccfilename = name + '_cqcc' + '.cpickle'
            feature_container = FeatureContainer().load(filename=mfccfilename)            
            feature_container.show()
            feature_container = FeatureContainer(filename=mfccfilename)
            print('\n-----------------------\n')
            fea = getCQCC_Separate(name)        
            feature_container1 = FeatureContainer(features=[fea])
            feature_container1.meta = feature_container.meta
            feature_container1.save(filename=cqccfilename)
            '''
            #feature_container1.show() 
            
                                  
            #name = line[49:53]
            #print(name)
            #PureData = 'C:/Users/21799506/Annaconda2/DCASE2017_Task3-master/Fe_dev/Mel/street/' + name + '.f'
            #name1 = line[0:53]
            #mfccfilename = name1 + '.csv'            
            #feature_container = FeatureContainer().load(filename=mfccfilename)            
            #pickle.dump(feature_container.feat, open(PureData,'wb'))
            #numpy.savetxt(PureData, feature_container.feat, delimiter=',')
            
        exit(0)
        
        #print('-----------------------------------------------------')
        #feature_container_ = FeatureContainer().load(filename='C:\DCASE2017-debug\dcase_framework\saved.mfcc.cpickle')
        #feature_container.log()
        #feature_container_.show()
        #print('Feature shape={shape}'.format(shape=feature_container.shape))
        #print('Feature channels={channels}'.format(channels=feature_container.channels))
        #print('Feature frames={frames}'.format(frames=feature_container.frames))
        #print('Feature vector length={vector_length}'.format(vector_length=feature_container.vector_length))
        #print(feature_container.feat)
        #print(feature_container.stat)
        #print(feature_container.meta)        

        
        if not files:
            files = []
            for fold in self._get_active_folds():
                for item_id, item in enumerate(self.dataset.train(fold)):
                    if item['file'] not in files:
                        files.append(item['file'])
                for item_id, item in enumerate(self.dataset.test(fold)):
                    if item['file'] not in files:
                        files.append(item['file'])
            files = sorted(files)

        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        feature_files = []
        feature_extractor = self.FeatureExtractor(overwrite=overwrite, store=True)
        for file_id, audio_filename in enumerate(tqdm(files, desc='           {0:<15s}'.format('Extracting features '),
                                                      file=sys.stdout,
                                                      leave=False,
                                                      disable=self.disable_progress_bar)):
            
            if self.log_system_progress:
                self.logger.info('  {title:<15s} [{file_id:d}/{total:d}] {file:<30s}'.format(
                    title='Extracting features ',
                    file_id=file_id,
                    total=len(files),
                    file=os.path.split(audio_filename)[-1])
                )

            # Get feature filename
            current_feature_files = self._get_feature_filename(audio_file=os.path.split(audio_filename)[1],
                                                               path=self.params.get_path('path.feature_extractor'))

            #print(current_feature_files)
            if not filelist_exists(current_feature_files) or overwrite:                
                feature_extractor.extract(
                    audio_file=self.dataset.relative_to_absolute_path(audio_filename),
                    extractor_params=DottedDict(self.params.get_path('feature_extractor.parameters')),
                    storage_paths=current_feature_files
                )

            feature_files.append(current_feature_files)

        return feature_files    
    def _before_feature_normalization(self, *args, **kwargs):
        self.ui.section_header('Feature normalizer')
        self.timer.start()

    def _after_feature_normalization(self, to_return=None):
        self.ui.foot(time=self.timer.stop().get_string())

    @before_and_after_function_wrapper
    def feature_normalization(self, overwrite=None):
        """Feature normalization stage

        Calculated normalization factors for each evaluation fold based on the training material available.

        Parameters
        ----------
        overwrite : bool
            overwrite existing normalizers
            (Default value=False)

        Returns
        -------
        None

        Raises
        -------
        IOError
            Feature file not found.

        """
        pass

    def _before_system_training(self, *args, **kwargs):
        self.ui.section_header('System training')
        self.timer.start()

    def _after_system_training(self, to_return=None):
        self.ui.foot(time=self.timer.stop().get_string())

    @before_and_after_function_wrapper
    
    def system_training(self, overwrite=None):
        """System training stage

        Parameters
        ----------

        overwrite : bool
            overwrite existing models
            (Default value=False)

        Returns
        -------
        None

        Raises
        -------
        IOError
            Feature normalizer not found.
            Feature file not found.

        """
        pass

    def _before_system_testing(self, *args, **kwargs):
        self.ui.section_header('System testing')
        self.timer.start()

    def _after_system_testing(self, to_return=None):
        self.ui.foot(time=self.timer.stop().get_string())

    def system_testing(self, overwrite=None):
        """System testing stage

        If extracted features are not found from disk, they are extracted but not saved.

        Parameters
        ----------
        overwrite : bool
            overwrite existing models
            (Default value=False)

        Returns
        -------
        None

        Raises
        -------
        IOError
            Model file not found.
            Audio file not found.

        """
        pass

    def _before_system_evaluation(self, *args, **kwargs):
        self.ui.section_header('System evaluation [{setup_label}][{dataset_evaluation_mode}]'.format(
            setup_label=self.setup_label,
            dataset_evaluation_mode=self.dataset_evaluation_mode)
        )
        self.timer.start()

    def _after_system_evaluation(self, to_return=None):
        self.ui.line(to_return)
        self.ui.foot(time=self.timer.stop().get_string())

    @before_and_after_function_wrapper
    def system_evaluation(self):
        """System evaluation stage.

        Testing outputs are collected and evaluated.

        Parameters
        ----------
        Returns
        -------
        None

        Raises
        -------
        IOError
            Result file not found

        """

        pass

    @before_and_after_function_wrapper
    def model_information(self):
        pass

    @staticmethod
    def _get_feature_filename(audio_file, path, extension='cpickle'):
        """Get feature filename

        Parameters
        ----------
        audio_file : str
            audio file name from which the features are extracted
        path :  str
            feature path
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        feature_filename : str, dict
            full feature filename or dict of filenames

        """

        audio_filename = os.path.split(audio_file)[1]
        if isinstance(path, dict):
            paths = {}
            for key, value in iteritems(path):
                paths[key] = os.path.join(value, os.path.splitext(audio_filename)[0] + '.' + extension)
            return paths

        else:
            return os.path.join(path, os.path.splitext(audio_filename)[0] + '.' + extension)

    @staticmethod
    def _get_feature_normalizer_filename(fold, path, extension='cpickle'):
        """Get normalizer filename

        Parameters
        ----------
        fold : int >= 0
            evaluation fold number
        path :  str
            normalizer path
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        normalizer_filename : str
            full normalizer filename

        """

        if isinstance(path, dict):
            paths = {}
            for key, value in iteritems(path):
                paths[key] = os.path.join(value, 'scale_fold' + str(fold) + '.' + extension)
            return paths
        elif isinstance(path, list):
            paths = []
            for value in path:
                paths.append(os.path.join(value, 'scale_fold' + str(fold) + '.' + extension))
            return paths
        else:
            return os.path.join(path, 'scale_fold' + str(fold) + '.' + extension)

    @staticmethod
    def _get_model_filename(fold, path, extension='cpickle'):
        """Get model filename

        Parameters
        ----------
        fold : int >= 0
            evaluation fold number
        path :  str
            model path
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        model_filename : str
            full model filename

        """

        return os.path.join(path, 'model_fold' + str(fold) + '.' + extension)

    @staticmethod
    def _get_result_filename(fold, path, extension='txt'):
        """Get result filename

        Parameters
        ----------
        fold : int >= 0
            evaluation fold number
        path :  str
            result path
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        result_filename : str
            full result filename

        """

        if fold == 0:
            return os.path.join(path, 'results.' + extension)
        else:
            return os.path.join(path, 'results_fold' + str(fold) + '.' + extension)


class AcousticSceneClassificationAppCore(AppCore):
    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        name : str
            Application name
            Default value "Application"
        setup_label : str
            Application setup label
            Default value "System"
        params : ParameterContainer
            Parameter container containing all parameters needed by application.
        dataset : str or class
            Dataset, if none given dataset name is taken from parameters "dataset->parameters->name"
            Default value "none"
        dataset_evaluation_mode : str
            Dataset evaluation mode, "full" or "folds". If none given, taken from parameters "dataset->parameter->evaluation_mode"
            Default value "none"
        show_progress_in_console : bool
            Show progress in console.
            Default value "True"
        log_system_progress : bool
            Log progress in console.
            Default value "False"
        logger : logging
            Instance of logging
            Default value "none"
        Datasets : dict of Dataset classes
            Dict of datasets available for application. Dict key is name of the dataset and value link to class inherited from Dataset base class. Given dict is used to update internal dict.
            Default value "none"
        FeatureExtractor : class inherited from FeatureExtractor
            Feature extractor class. Use this to override default class.
            Default value "FeatureExtractor"
        FeatureNormalizer : class inherited from FeatureNormalizer
            Feature normalizer class. Use this to override default class.
            Default value "FeatureNormalizer"
        FeatureMasker : class inherited from FeatureMasker
            Feature masker class. Use this to override default class.
            Default value "FeatureMasker"
        FeatureContainer : class inherited from FeatureContainer
            Feature container class. Use this to override default class.
            Default value "FeatureContainer"
        FeatureStacker : class inherited from FeatureStacker
            Feature stacker class. Use this to override default class.
            Default value "FeatureStacker"
        FeatureAggregator : class inherited from FeatureAggregator
            Feature aggregate class. Use this to override default class.
            Default value "FeatureAggregator"
        Learners: dict of Learner classes
            Dict of learners available for application. Dict key is method the class implements and value link to class inherited from LearnerContainer base class. Given dict is used to update internal dict.
        ui : class inherited from FancyLogger
            Output formatter class. Use this to override default class.
            Default value "FancyLogger"

        Raises
        ------
        ValueError:
            No valid ParameterContainer given.

        """

        super(AcousticSceneClassificationAppCore, self).__init__(*args, **kwargs)

        # Fetch datasets
        self.Datasets = {}

        for dataset_item in get_class_inheritors(AcousticSceneDataset):
            self.Datasets[dataset_item.__name__] = dataset_item

        if kwargs.get('Datasets'):
            self.Datasets.update(kwargs.get('Datasets'))

        # Set current dataset
        self.dataset = self._get_dataset(dataset=kwargs.get('dataset'))

        # Fetch all learners
        self.Learners = {}
        learner_list = get_class_inheritors(SceneClassifier)
        for learner_item in learner_list:
            #learner = learner_item()
            learner = learner_item
            if learner.method:
                self.Learners[learner.method] = learner_item

        if kwargs.get('Learners'):
            self.Learners.update(kwargs.get('Learners'))

    def show_eval(self):

        eval_path = self.params.get_path('path.evaluator')

        eval_files = []
        for filename in os.listdir(eval_path):
            if filename.endswith('.yaml'):
                eval_files.append(os.path.join(eval_path, filename))

        eval_data = {}
        for filename in eval_files:
            data = DottedDict(ParameterFile().load(filename=filename))
            set_id = data.get_path('parameters.set_id')
            if set_id not in eval_data:
                eval_data[set_id] = {}
            params_hash = data.get_path('parameters._hash')

            if params_hash not in eval_data[set_id]:
                eval_data[set_id][params_hash] = data

        output = ''
        output += '  Evaluated systems\n'
        output += '  {set_id:<15s} | {accuracy:8s} | {desc:46s} | {hash:32s} |\n'.format(
            set_id='Set id',
            desc='Description',
            hash='Hash',
            accuracy='Accuracy'
        )

        output += '  {set_id:<15s} + {accuracy:8s} + {desc:46s} + {hash:32s} +\n'.format(
            set_id='-' * 15,
            desc='-' * 46,
            hash='-' * 32,
            accuracy='-'*8
        )

        for set_id in sorted(list(eval_data.keys())):
            for params_hash in eval_data[set_id]:
                data = eval_data[set_id][params_hash]
                desc = data.get_path('parameters.description')
                output += '  {set_id:<15s} | {accuracy:<8s} | {desc:46s} | {hash:32s} |\n'.format(
                    set_id=set_id,
                    desc=(desc[:44] + '..') if len(desc) > 44 else desc,
                    hash=params_hash,
                    accuracy='{0:4.2f} %'.format(data.get_path('overall_results.overall.accuracy')*100)
                )
        self.ui.line(output)

    @before_and_after_function_wrapper
    def feature_normalization(self, overwrite=None):
        """Feature normalization stage

        Calculated normalization factors for each evaluation fold based on the training material available.

        Parameters
        ----------
        overwrite : bool
            overwrite existing normalizers
            (Default value=False)

        Returns
        -------
        None

        Raises
        -------
        IOError
            Feature file not found.

        """
        warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore", DeprecationWarning)

        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        if self.params.get_path('feature_normalizer.enable', False) and self.params.get_path('feature_normalizer.type', 'global') == 'global':
            fold_progress = tqdm(self._get_active_folds(),
                                 desc='           {0:<15s}'.format('Fold '),
                                 file=sys.stdout,
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar)

            for fold in fold_progress:
                if self.log_system_progress:
                    self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(title='Fold',
                                                                                  fold=fold,
                                                                                  total=len(fold_progress)))

                current_normalizer_files = self._get_feature_normalizer_filename(
                    fold=fold,
                    path=self.params.get_path('path.feature_normalizer')
                )

                method_progress = tqdm(current_normalizer_files,
                                       desc='           {0: >15s}'.format('Feature method '),
                                       file=sys.stdout,
                                       leave=False,
                                       miniters=1,
                                       disable=self.disable_progress_bar)

                for method in method_progress:
                    current_normalizer_file = current_normalizer_files[method]
                    if not os.path.isfile(current_normalizer_file) or overwrite:
                        normalizer = self.FeatureNormalizer()
                        item_progress = tqdm(self.dataset.train(fold),
                                             desc="           {0: >15s}".format('Collect data '),
                                             file=sys.stdout,
                                             leave=False,
                                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                             disable=self.disable_progress_bar
                                             )
                        for item_id, item in enumerate(item_progress):
                            feature_filename = self._get_feature_filename(
                                audio_file=item['file'],
                                path=self.params.get_path('path.feature_extractor', {})[method]
                            )
                            if self.log_system_progress:
                                self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {file:<30s}'.format(
                                    title='Item',
                                    item_id=item_id,
                                    total=len(item_progress),
                                    file=os.path.split(feature_filename)[-1])
                                )

                            if os.path.isfile(feature_filename):
                                feature_stats = self.FeatureContainer(filename=feature_filename)
                            else:
                                message = '{name}: Features not found [{file}]'.format(
                                    name=self.__class__.__name__,
                                    method=item['file']
                                )

                                self.logger.exception(message)
                                raise IOError(message)

                            # Accumulate statistics
                            normalizer.accumulate(feature_stats)

                        # Calculate normalization factors
                        normalizer.finalize().save(filename=current_normalizer_file)

    @before_and_after_function_wrapper
    def system_training(self, overwrite=None):
        """System training stage

        Parameters
        ----------

        overwrite : bool
            overwrite existing models
            (Default value=False)

        Returns
        -------
        None

        Raises
        -------
        IOError
            Feature normalizer not found.
            Feature file not found.

        """

        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        fold_progress = tqdm(self._get_active_folds(),
                             desc='           {0:<15s}'.format('Fold '),
                             file=sys.stdout,
                             leave=False,
                             miniters=1,
                             disable=self.disable_progress_bar)

        for fold in fold_progress:
            if self.log_system_progress:
                self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(title='Fold',
                                                                              fold=fold,
                                                                              total=len(fold_progress)))

            current_model_file = self._get_model_filename(fold=fold, path=self.params.get_path('path.learner'))
            if not os.path.isfile(current_model_file) or overwrite:
                # Feature stacker
                feature_stacker = self.FeatureStacker(recipe=self.params.get_path('feature_extractor.recipe'))

                feature_masker = None
                if self.params.get_path('learner.audio_error_handling'):
                    feature_masker = self.FeatureMasker(
                        hop_length_seconds=self.params.get_path('feature_extractor.hop_length_seconds')
                    )

                feature_aggregator = None
                if self.params.get_path('feature_aggregator.enable'):
                    feature_aggregator = self.FeatureAggregator(
                        recipe=self.params.get_path('feature_aggregator.aggregation_recipe'),
                        win_length_frames=self.params.get_path('feature_aggregator.win_length_frames'),
                        hop_length_frames=self.params.get_path('feature_aggregator.hop_length_frames')
                    )

                feature_normalizer = None
                if self.params.get_path('feature_normalizer.enable'):
                    # Load normalizers
                    feature_normalizer_filenames = self._get_feature_normalizer_filename(
                        fold=fold,
                        path=self.params.get_path('path.feature_normalizer')
                    )

                    normalizer_list = {}
                    for method, feature_normalizer_filename in iteritems(feature_normalizer_filenames):
                        if os.path.isfile(feature_normalizer_filename):
                            normalizer_list[method] = self.FeatureNormalizer().load(
                                filename=feature_normalizer_filename
                            )

                        else:
                            message = '{name}: Feature normalizer not found [{file}]'.format(
                                name=self.__class__.__name__,
                                method=feature_normalizer_filename
                            )

                            self.logger.exception(message)
                            raise IOError(message)

                    feature_normalizer = self.FeatureNormalizer(feature_stacker.normalizer(
                        normalizer_list=normalizer_list)
                    )

                # Collect training examples
                data = {}
                annotations = {}
                item_progress = tqdm(self.dataset.train(fold),
                                     desc="           {0: >15s}".format('Collect data '),
                                     file=sys.stdout,
                                     leave=False,
                                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                     disable=self.disable_progress_bar
                                     )
                for item_id, item in enumerate(item_progress):
                    if self.log_system_progress:
                        self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {item:<20s}'.format(
                            title='Collect data ',
                            item_id=item_id,
                            total=len(item_progress),
                            item=os.path.split(item['file'])[-1])
                        )

                    item_progress.set_postfix(file=os.path.splitext(os.path.split(item['file'])[-1])[0])
                    item_progress.update()

                    # Load features
                    feature_filenames = self._get_feature_filename(audio_file=item['file'],
                                                                   path=self.params.get_path('path.feature_extractor'))
                    feature_repository = FeatureRepository()
                    for method, feature_filename in iteritems(feature_filenames):
                        if os.path.isfile(feature_filename):
                            feature_repository[method] = self.FeatureContainer().load(filename=feature_filename)
                        else:
                            message = '{name}: Features not found [{file}]'.format(
                                name=self.__class__.__name__,
                                method=item['file']
                            )

                            self.logger.exception(message)
                            raise IOError(message)

                    # Mask annotated audio errors
                    if feature_masker:
                        feature_repository = feature_masker.process(
                            feature_repository=feature_repository,
                            mask_events=self.dataset.file_error_meta(item['file'])
                        )

                    # Stack features
                    feature_container = feature_stacker.process(feature_repository=feature_repository)

                    # Normalize features
                    if feature_normalizer:
                        feature_container = feature_normalizer.process(feature_container=feature_container)

                    # Aggregate features
                    if feature_aggregator:
                        feature_container = feature_aggregator.process(feature_container=feature_container)

                    data[item['file']] = feature_container
                    annotations[item['file']] = item

                learner = self._get_learner(method=self.params.get_path('learner.method'),
                                            class_labels=self.dataset.scene_labels,
                                            feature_masker=feature_masker,
                                            feature_normalizer=feature_normalizer,
                                            feature_stacker=feature_stacker,
                                            feature_aggregator=feature_aggregator,
                                            params=self.params.get_path('learner'),
                                            filename=current_model_file,
                                            disable_progress_bar=self.disable_progress_bar,
                                            log_progress=self.log_system_progress)
                learner.learn(data=data, annotations=annotations)
                learner.save()

            if self.params.get_path('learner.show_model_information'):
                # Load class model container
                model_filename = self._get_model_filename(fold=fold, path=self.params.get_path('path.learner'))

                if os.path.isfile(model_filename):
                    model_container = self._get_learner(method=self.params.get_path('learner.method')).load(filename=model_filename)
                else:
                    message = '{name}: Model file not found [{file}]'.format(
                        name=self.__class__.__name__,
                        method=model_filename
                    )

                    self.logger.exception(message)
                    raise IOError(message)


                if 'learning_history' in model_container:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)

                    ax.spines['top'].set_color('none')
                    ax.spines['bottom'].set_color('none')
                    ax.spines['left'].set_color('none')
                    ax.spines['right'].set_color('none')
                    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

                    # Set common labels
                    ax.set_xlabel('Epochs')

                    ax1.set_title('model accuracy')
                    if 'categorical_accuracy' in model_container['learning_history']:
                        ax1.set_title('model accuracy / categorical accuracy')
                        ax1.plot(model_container['learning_history']['categorical_accuracy'])
                        ax1.set_ylabel('accuracy')
                        if 'val_categorical_accuracy' in model_container['learning_history']:
                            ax1.plot(model_container['learning_history']['val_categorical_accuracy'])
                            ax1.legend(['train', 'validation'], loc='upper left')
                        else:
                            ax1.legend(['train'], loc='upper left')

                    if 'loss' in model_container['learning_history']:
                        ax2.set_title('model loss')
                        ax2.plot(model_container['learning_history']['loss'])
                        ax2.set_ylabel('loss')
                        if 'val_loss' in model_container['learning_history']:
                            ax2.plot(model_container['learning_history']['val_loss'])
                            ax2.legend(['train', 'validation'], loc='upper left')
                        else:
                            ax2.legend(['train'], loc='upper left')
                    plt.show()

    @before_and_after_function_wrapper
    def system_testing(self, overwrite=None):
        """System testing stage

        If extracted features are not found from disk, they are extracted but not saved.

        Parameters
        ----------
        overwrite : bool
            overwrite existing models
            (Default value=False)

        Returns
        -------
        None

        Raises
        -------
        IOError
            Model file not found.
            Audio file not found.

        """

        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)
        fold_progress = tqdm(self._get_active_folds(),
                             desc="           {0: >15s}".format('Fold '),
                             file=sys.stdout,
                             leave=False,
                             miniters=1,
                             disable=self.disable_progress_bar)

        for fold in fold_progress:
            if self.log_system_progress:
                self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(title='Fold',
                                                                              fold=fold,
                                                                              total=len(fold_progress)))

            current_result_file = self._get_result_filename(fold=fold, path=self.params.get_path('path.recognizer'))
            if not os.path.isfile(current_result_file) or overwrite:
                results = MetaDataContainer(filename=current_result_file)

                # Load class model container
                model_filename = self._get_model_filename(fold=fold, path=self.params.get_path('path.learner'))

                if os.path.isfile(model_filename):
                    model_container = self._get_learner(method=self.params.get_path('learner.method')).load(
                        filename=model_filename)
                else:
                    message = '{name}: Model file not found [{file}]'.format(
                        name=self.__class__.__name__,
                        method=model_filename
                    )

                    self.logger.exception(message)
                    raise IOError(message)

                item_progress = tqdm(self.dataset.test(fold),
                                     desc="           {0: >15s}".format('Testing '),
                                     file=sys.stdout,
                                     leave=False,
                                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                     disable=self.disable_progress_bar
                                     )
                for item_id, item in enumerate(item_progress):
                    if self.log_system_progress:
                        self.logger.info('  {title:<15s} [{item_id:d}/{total:d}] {item:<20s}'.format(title='Testing',
                                                                                                     item_id=item_id,
                                                                                                     total=len(
                                                                                                         item_progress),
                                                                                                     item=os.path.split(
                                                                                                         item['file'])[
                                                                                                         -1])
                                         )
                    # Load features
                    feature_filenames = self._get_feature_filename(audio_file=item['file'],
                                                                   path=self.params.get_path('path.feature_extractor'))
                    feature_repository = FeatureRepository()
                    for method, feature_filename in iteritems(feature_filenames):
                        if os.path.isfile(feature_filename):
                            feature_repository[method] = self.FeatureContainer().load(filename=feature_filename)
                        else:
                            message = '{name}: Features not found [{file}]'.format(
                                name=self.__class__.__name__,
                                method=item['file']
                            )

                            self.logger.exception(message)
                            raise IOError(message)

                    if self.params.get_path('recognizer.audio_error_handling'):
                        if model_container.feature_masker:
                            feature_repository = model_container.feature_masker.process(
                                feature_repository=feature_repository,
                                mask_events=self.dataset.file_error_meta(item['file'])
                            )
                        else:
                            feature_repository = self.FeatureMasker(hop_length_seconds=self.params.get_path(
                                'feature_extractor.hop_length_seconds')).process(
                                feature_repository=feature_repository,
                                mask_events=self.dataset.file_error_meta(item['file'])
                            )

                    # Stack features
                    feature_data = model_container.feature_stacker.process(feature_repository=feature_repository)

                    # Normalize features
                    if model_container.feature_normalizer:
                        feature_data = model_container.feature_normalizer.process(feature_container=feature_data)

                    # Aggregate features
                    if model_container.feature_aggregator:
                        feature_data = model_container.feature_aggregator.process(feature_container=feature_data)

                    current_result = model_container.predict(feature_data=feature_data,
                                                             recognizer_params=self.params.get_path('recognizer'))

                    # Store the result
                    results.append(MetaDataItem({
                        'file': self.dataset.absolute_to_relative(item['file']),
                        'scene_label': current_result}
                    ))

                # Save testing results
                results.save()

    @before_and_after_function_wrapper
    def system_evaluation(self):
        """System evaluation stage.

        Testing outputs are collected and evaluated.

        Parameters
        ----------
        Returns
        -------
        None

        Raises
        -------
        IOError
            Result file not found

        """
        if not self.dataset.reference_data_present:
            return '  No reference data available for dataset.'
        else:
            scene_metric = sed_eval.scene.SceneClassificationMetrics(scene_labels=self.dataset.scene_labels)
            results_fold = []
            fold_progress = tqdm(self._get_active_folds(),
                                 desc="           {0: >15s}".format('Fold '),
                                 file=sys.stdout,
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar)
            for fold in fold_progress:
                scene_metric_fold = sed_eval.scene.SceneClassificationMetrics(scene_labels=self.dataset.scene_labels)

                estimated_scene_list = MetaDataContainer(
                    filename=self._get_result_filename(fold=fold, path=self.params.get_path('path.recognizer'))).load()
                reference_scene_list = self.dataset.eval(fold=fold)
                for item in reference_scene_list:
                    item['file'] = self.dataset.absolute_to_relative(item['file'])

                scene_metric.evaluate(reference_scene_list=reference_scene_list,
                                      estimated_scene_list=estimated_scene_list)
                scene_metric_fold.evaluate(reference_scene_list=reference_scene_list,
                                           estimated_scene_list=estimated_scene_list)
                results_fold.append(scene_metric_fold.results())

            results = scene_metric.results()

            output = ''
            output += "  File-wise evaluation, over {fold_count:d} folds\n".format(
                fold_count=len(self._get_active_folds())
            )

            fold_labels = ''
            separator = '     =====================+======+======+==========+  +'
            if len(results_fold) > 1:
                for fold in self.dataset.folds(mode=self.dataset_evaluation_mode):
                    fold_labels += " {fold:8s} |".format(fold='Fold' + str(fold))
                    separator += "==========+"
            output += "     {scene:20s} | {ref:4s} : {sys:4s} | {acc:8s} |  |".format(
                scene='Scene label',
                ref='Nref',
                sys='Nsys',
                acc='Accuracy')
            output += fold_labels + '\n'

            output += separator + '\n'
            for label_id, label in enumerate(sorted(results['class_wise'])):
                fold_values = ''
                if len(results_fold) > 1:
                    for fold in self.dataset.folds(mode=self.dataset_evaluation_mode):
                        fold_values += " {value:5.1f} %  |".format(
                            value=results_fold[fold - 1]['class_wise'][label]['accuracy']['accuracy'] * 100
                        )

                output += "     {label:20s} | {ref:4d} : {sys:4d} | {acc:5.1f} %  |  |".format(
                    label=label,
                    ref=int(results['class_wise'][label]['count']['Nref']),
                    sys=int(results['class_wise'][label]['count']['Nsys']),
                    acc=results['class_wise'][label]['accuracy']['accuracy'] * 100)
                output += fold_values + '\n'

            output += separator + '\n'
            fold_values = ''
            if len(results_fold) > 1:
                for fold in self.dataset.folds(mode=self.dataset_evaluation_mode):
                    fold_values += " {:5.1f} %  |".format(results_fold[fold - 1]['overall']['accuracy'] * 100)

            output += "     {label:20s} | {ref:4d} : {sys:4d} | {acc:5.1f} %  |  |".format(
                label='Overall accuracy',
                ref=int(results['overall']['count']['Nref']),
                sys=int(results['overall']['count']['Nsys']),
                acc=results['overall']['accuracy'] * 100)
            output += fold_values + '\n'

            if self.params.get_path('evaluator.saving.enable'):
                filename = self.params.get_path('evaluator.saving.filename').format(
                    dataset_name=self.dataset.storage_name,
                    parameter_set=self.params['active_set'],
                    parameter_hash=self.params['_hash']
                )

                output_file = os.path.join(self.params.get_path('path.evaluator'), filename)

                output_data = {
                    'overall_results': results,
                    'fold_results': results_fold,
                    'parameters': dict(self.params)
                }
                ParameterFile(output_data, filename=output_file).save()
            return output


class SoundEventAppCore(AppCore):
    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        name : str
            Application name
            Default value "Application"
        setup_label : str
            Application setup label
            Default value "System"
        params : ParameterContainer
            Parameter container containing all parameters needed by application.
        dataset : str or class
            Dataset, if none given dataset name is taken from parameters "dataset->parameters->name"
            Default value "none"
        dataset_evaluation_mode : str
            Dataset evaluation mode, "full" or "folds". If none given, taken from parameters "dataset->parameter->evaluation_mode"
            Default value "none"
        show_progress_in_console : bool
            Show progress in console.
            Default value "True"
        log_system_progress : bool
            Log progress in console.
            Default value "False"
        logger : logging
            Instance of logging
            Default value "none"
        Datasets : dict of Dataset classes
            Dict of datasets available for application. Dict key is name of the dataset and value link to class inherited from Dataset base class. Given dict is used to update internal dict.
            Default value "none"
        FeatureExtractor : class inherited from FeatureExtractor
            Feature extractor class. Use this to override default class.
            Default value "FeatureExtractor"
        FeatureNormalizer : class inherited from FeatureNormalizer
            Feature normalizer class. Use this to override default class.
            Default value "FeatureNormalizer"
        FeatureMasker : class inherited from FeatureMasker
            Feature masker class. Use this to override default class.
            Default value "FeatureMasker"
        FeatureContainer : class inherited from FeatureContainer
            Feature container class. Use this to override default class.
            Default value "FeatureContainer"
        FeatureStacker : class inherited from FeatureStacker
            Feature stacker class. Use this to override default class.
            Default value "FeatureStacker"
        FeatureAggregator : class inherited from FeatureAggregator
            Feature aggregate class. Use this to override default class.
            Default value "FeatureAggregator"
        Learners: dict of Learner classes
            Dict of learners available for application. Dict key is method the class implements and value link to class inherited from LearnerContainer base class. Given dict is used to update internal dict.
        ui : class inherited from FancyLogger
            Output formatter class. Use this to override default class.
            Default value "FancyLogger"

        Raises
        ------
        ValueError:
            No valid ParameterContainer given.

        """

        super(SoundEventAppCore, self).__init__(*args, **kwargs)

        # Fetch datasets
        self.Datasets = {}

        for dataset_item in get_class_inheritors(SoundEventDataset):
            self.Datasets[dataset_item.__name__] = dataset_item

        if kwargs.get('Datasets'):
            self.Datasets.update(kwargs.get('Datasets'))

        # Set current dataset
        self.dataset = self._get_dataset(dataset=kwargs.get('dataset'))

        # Fetch all learners
        self.Learners = {}
        learner_list = get_class_inheritors(EventDetector)
        for learner_item in learner_list:
            #learner = learner_item()
            learner = learner_item
            if learner.method:
                self.Learners[learner.method] = learner_item

        if kwargs.get('Learners'):
            self.Learners.update(kwargs.get('Learners'))

    @before_and_after_function_wrapper
    def feature_normalization(self, overwrite=None):
        """Feature normalization stage

        Calculated normalization factors for each evaluation fold based on the training material available.

        Parameters
        ----------
        overwrite : bool
            overwrite existing normalizers
            (Default value=False)

        Raises
        -------
        ValueError:
            Unknown scene_handling mode
        IOError:
            Feature file not found.

        Returns
        -------
        None

        """

        print ('Normalization')
        
        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        if self.params.get_path('feature_normalizer.enable', False) and self.params.get_path('feature_normalizer.method', 'global') == 'global':
            if self.params.get_path('feature_normalizer.scene_handling') == 'scene-dependent':

                fold_progress = tqdm(self._get_active_folds(),
                                     desc='           {0:<15s}'.format('Fold '),
                                     file=sys.stdout,
                                     leave=False,
                                     miniters=1,
                                     disable=self.disable_progress_bar)

                for fold in fold_progress:
                    if self.log_system_progress:
                        self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(
                            title='Fold',
                            fold=fold,
                            total=len(fold_progress))
                        )

                    
                    for scene_label in self.dataset.scene_labels:
                        current_normalizer_files = self._get_feature_normalizer_filename(
                            fold=fold,
                            path=self.params.get_path('path.feature_normalizer'),
                            scene_label=scene_label
                        )

                        method_progress = tqdm(current_normalizer_files,
                                               desc='           {0: >15s}'.format('Feature method '),
                                               file=sys.stdout,
                                               leave=False,
                                               miniters=1,
                                               disable=self.disable_progress_bar)

                        for method in method_progress:
                            current_normalizer_file = current_normalizer_files[method]
                            if not os.path.isfile(current_normalizer_file) or overwrite:
                                normalizer = FeatureNormalizer()
                                item_progress = tqdm(self.dataset.train(fold, scene_label=scene_label),
                                                     desc="           {0: >15s}".format('Collect data '),
                                                     file=sys.stdout,
                                                     leave=False,
                                                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                                     disable=self.disable_progress_bar
                                                     )

                                for item_id, item in enumerate(item_progress):
                                    '''
                                    print('-----------------------')
                                    print(fold)
                                    print(item['file'])
                                    print('*****************************')
                                    '''
                                    feature_filename = self._get_feature_filename(
                                        audio_file=item['file'],
                                        path=self.params.get_path('path.feature_extractor', {})[method]
                                    )                                                                    
                                    if self.log_system_progress:
                                        self.logger.info(
                                            '  {title:<15s} [{item_id:d}/{total:d}] {file:<30s}'.format(
                                                title='Item',
                                                item_id=item_id,
                                                total=len(item_progress),
                                                file=os.path.split(feature_filename)[-1])
                                            )
                                    
                                    if os.path.isfile(feature_filename):

                                        feature_stats = FeatureContainer(filename=feature_filename)
                                        #print(feature_stats.shape)
                                    else:
                                        message = '{name}: Features not found [{file}]'.format(
                                            name=self.__class__.__name__,
                                            method=item['file']
                                        )

                                        self.logger.exception(message)
                                        raise IOError(message)

                                    # Accumulate statistics
                                    normalizer.accumulate(feature_stats)                                                                    
                                # Calculate normalization factors
                                normalizer.finalize().save(filename=current_normalizer_file)

            elif self.params.get_path('feature_normalizer.scene_handling') == 'scene-independent':
                message = '{name}: Scene handling mode not implemented yet [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('feature_normalizer.scene_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

            else:
                message = '{name}: Unknown scene handling mode [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('feature_normalizer.scene_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

    @before_and_after_function_wrapper
    def system_training(self, overwrite=None):
        """System training stage

        Parameters        ----------

        overwrite : bool
            overwrite existing models
            (Default value=False)

        Raises
        -------
        ValueError:
            Unknown scene_handling mode
        IOError:
            Feature normalizer not found.
            Feature file not found.

        Returns
        -------
        None

        """
        
        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        if self.params.get_path('learner.scene_handling') == 'scene-dependent':
            fold_progress = tqdm(self._get_active_folds(),
                                 desc='           {0:<15s}'.format('Fold '),
                                 file=sys.stdout,
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar)

            for fold in fold_progress:
                if self.log_system_progress:
                    self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(title='Fold',
                                                                                  fold=fold,
                                                                                  total=len(fold_progress)))
                scene_progress = tqdm(self.dataset.scene_labels,
                                      desc="           {0: >15s}".format('Scene '),
                                      file=sys.stdout,
                                      leave=False,
                                      miniters=1,
                                      disable=self.disable_progress_bar)
                for scene_label in scene_progress:
                    current_model_file = self._get_model_filename(fold=fold,
                                                                  path=self.params.get_path('path.learner'),
                                                                  scene_label=scene_label)
                    if not os.path.isfile(current_model_file) or overwrite:
                        # Feature stacker
                        feature_stacker = FeatureStacker(recipe=self.params.get_path('feature_extractor.recipe'))

                        feature_aggregator = None
                        if self.params.get_path('feature_aggregator.enable'):
                            feature_aggregator = FeatureAggregator(
                                recipe=self.params.get_path('feature_aggregator.aggregation_recipe'),
                                win_length_frames=self.params.get_path('feature_aggregator.win_length_frames'),
                                hop_length_frames=self.params.get_path('feature_aggregator.hop_length_frames')
                            )

                        feature_normalizer = None
                        if self.params.get_path('feature_normalizer.enable'):
                            # Load normalizers
                            feature_normalizer_filenames = self._get_feature_normalizer_filename(
                                fold=fold,
                                path=self.params.get_path('path.feature_normalizer'),
                                scene_label=scene_label
                            )

                            normalizer_list = {}
                            for method, feature_normalizer_filename in iteritems(feature_normalizer_filenames):
                                if os.path.isfile(feature_normalizer_filename):
                                    normalizer_list[method] = FeatureNormalizer().load(filename=feature_normalizer_filename)
                                else:
                                    message = '{name}: Feature normalizer not found [{file}]'.format(
                                        name=self.__class__.__name__,
                                        method=feature_normalizer_filename
                                    )

                                    self.logger.exception(message)
                                    raise IOError(message)

                            feature_normalizer = self.FeatureNormalizer(feature_stacker.normalizer(normalizer_list=normalizer_list))

                        # Collect training examples
                        train_meta = self.dataset.train(fold, scene_label=scene_label)
                        data = {}
                        annotations = {}

                        item_progress = tqdm(train_meta.file_list,
                                             desc="           {0: >15s}".format('Collect data '),
                                             file=sys.stdout,
                                             leave=False,
                                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                             disable=self.disable_progress_bar
                                             )
                        for item_id, audio_filename in enumerate(item_progress):
                            
                            if self.log_system_progress:
                                self.logger.info(
                                    '  {title:<15s} [{item_id:d}/{total:d}] {item:<20s}'.format(
                                        title='Collect data ',
                                        item_id=item_id,
                                        total=len(item_progress),
                                        item=os.path.split(audio_filename)[-1])
                                    )

                            item_progress.set_postfix(file=os.path.splitext(os.path.split(audio_filename)[-1])[0])
                            #item_progress.update()

                            # Load features
                            feature_filenames = self._get_feature_filename(
                                audio_file=audio_filename,
                                path=self.params.get_path('path.feature_extractor')
                            )

                            feature_list = {}
                            for method, feature_filename in iteritems(feature_filenames):
                                if os.path.isfile(feature_filename):
                                    feature_list[method] = FeatureContainer().load(filename=feature_filename)                                    
                                else:
                                    message = '{name}: Features not found [{file}]'.format(
                                        name=self.__class__.__name__,
                                        method=feature_filename
                                    )

                                    self.logger.exception(message)
                                    raise IOError(message)

                            # Stack features
                            feature_data = feature_stacker.feature_vector(feature_repository=feature_list)
                            

                            

                            # Normalize features
                               
                            #feature_normalizer = 0
                            if feature_normalizer:
                                feature_data = feature_normalizer.normalize(feature_data)

                            # Aggregate features
                            if feature_aggregator:
                                feature_data = feature_aggregator.process(feature_container=feature_data)

                            data[audio_filename] = feature_data
                            annotations[audio_filename] = train_meta.filter(filename=audio_filename)                            
                        method = self.params.get_path('learner.method') 
                        #print(self.params)               
                        fc = self._get_learner(method=self.params.get_path('learner.method'),                        
                                               class_labels=self.dataset.event_labels(scene_label=scene_label),
                                               feature_normalizer=feature_normalizer,
                                               feature_stacker=feature_stacker,
                                               feature_aggregator=feature_aggregator,
                                               params=self.params.get_path('learner'),
                                               filename=current_model_file,
                                               disable_progress_bar=self.disable_progress_bar,
                                               log_progress=self.log_system_progress)                 
                        fc.learn(data=data, annotations=annotations)
                        fc.save()

        elif self.params.get_path('learner.scene_handling') == 'scene-independent':
            message = '{name}: Scene handling mode not implemented yet [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('learner.scene_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

        else:
            message = '{name}: Unknown scene handling mode [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('learner.scene_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

    @before_and_after_function_wrapper
    def system_testing(self, overwrite=None):
        """System testing stage

        If extracted features are not found from disk, they are extracted but not saved.

        Parameters
        ----------
        overwrite : bool
            overwrite existing models
            (Default value=False)

        Raises
        -------
        ValueError:
            Unknown scene_handling mode
        IOError:
            Model file not found.
            Audio file not found.

        Returns
        -------
        None

        """

                
        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        if self.params.get_path('recognizer.scene_handling') == 'scene-dependent':          
            fold_progress = tqdm(self._get_active_folds(),
                                 desc="           {0: >15s}".format('Fold '),
                                 file=sys.stdout,
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar)
            for fold in fold_progress:
                print(fold)
                if self.log_system_progress:
                    self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(title='Fold',
                                                                                  fold=fold,
                                                                                  total=len(fold_progress)))
                print(self.dataset.scene_labels)
                scene_progress = tqdm(self.dataset.scene_labels,
                                      desc="           {0: >15s}".format('Scene '),
                                      file=sys.stdout,
                                      leave=False,
                                      miniters=1,
                                      disable=self.disable_progress_bar)
                print('scene_progress')
                for scene_label in scene_progress:
                    current_result_file = self._get_result_filename(fold=fold,
                                                                    path=self.params.get_path('path.recognizer'),
                                                                    scene_label=scene_label
                                                                    )
                    if not os.path.isfile(current_result_file) or overwrite:
                        results = MetaDataContainer(filename=current_result_file)

                        # Load class model container

                        model_filename = self._get_model_filename(fold=fold,
                                                                  path=self.params.get_path('path.learner'),
                                                                  scene_label=scene_label
                                                                  )

                        print(model_filename)
                        if os.path.isfile(model_filename):
                            print(self.params.get_path('learner.method'))
                            fb = self._get_learner(method='mlp',params=self.params.get_path('learner'))
                                                      
                            model_container=fb.load(filename=model_filename)
                            
                        else:
                            message = '{name}: Model file not found [{file}]'.format(
                                name=self.__class__.__name__,
                                method=model_filename
                            )

                            self.logger.exception(message)
                            raise IOError(message)
                        
                        item_progress = tqdm(self.dataset.test(fold, scene_label=scene_label),
                                             desc="           {0: >15s}".format('Testing '),
                                             file=sys.stdout,
                                             leave=False,
                                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                             disable=self.disable_progress_bar
                                             )
                        for item_id, item in enumerate(item_progress):
                            
                            if self.log_system_progress:
                                self.logger.info(
                                    '  {title:<15s} [{item_id:d}/{total:d}] {item:<20s}'.format(
                                        title='Testing',
                                        item_id=item_id,
                                        total=len(item_progress),
                                        item=os.path.split(item['file'])[-1])
                                    )

                            # Load features
                            print(item['file'])                                                      

                            feature_filenames = self._get_feature_filename(
                                audio_file=item['file'],
                                path=self.params.get_path('path.feature_extractor')
                            )
                            

                            feature_list = {}
                            for method, feature_filename in iteritems(feature_filenames):
                                if os.path.isfile(feature_filename):
                                    feature_list[method] = FeatureContainer().load(filename=feature_filename)
                                else:
                                    message = '{name}: Features not found [{file}]'.format(
                                        name=self.__class__.__name__,
                                        method=item['file']
                                    )

                                    self.logger.exception(message)
                                    raise IOError(message)

                            feature_data = model_container.feature_stacker.feature_vector(
                                feature_repository=feature_list)

                            
                            
                            ####################################################### get the CQT data below                            
                            ####################################################### get the CQT data above


                            # Normalize features
                            if model_container.feature_normalizer:
                                feature_data = model_container.feature_normalizer.normalize(feature_data)

                            # Aggregate features
                            if model_container.feature_aggregator:
                                feature_data = model_container.feature_aggregator.process(feature_data)
                            current_result = model_container.predict_UseOutsideModel(
                                feature_data=feature_data,
                                fold = fold,
                                item = item['file'],
                                recognizer_params=self.params.get_path('recognizer')                                
                            )
                            '''
                            current_result = model_container.predict(
                                feature_data=feature_data,
                                recognizer_params=self.params.get_path('recognizer')
                            )
                            '''
                            if self.params.get_path('recognizer.event_post_processing.enable'):
                                # Event list post-processing
                                current_result = current_result.process_events(
                                    minimum_event_length=self.params.get_path('recognizer.event_post_processing.minimum_event_length_seconds'),
                                    minimum_event_gap=self.params.get_path('recognizer.event_post_processing.minimum_event_gap_seconds')
                                )

                            for event in current_result:
                                event.file = self.dataset.absolute_to_relative(item['file'])
                                results.append(event)

                        # Save testing results
                        #print(results)
                        #exit()
                        
                        results.save()

        elif self.params.get_path('recognizer.scene_handling') == 'scene-independent':
            message = '{name}: Scene handling mode not implemented yet [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('recognizer.scene_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

        else:
            message = '{name}: Unknown scene handling mode [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('recognizer.scene_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)


    @before_and_after_function_wrapper
    def system_evaluation(self):
        """System evaluation stage.

        Testing outputs are collected and evaluated.

        Parameters
        ----------

        Returns
        -------
        None

        Raises
        -------
        IOError
            Result file not found

        """
        if not self.dataset.reference_data_present:
            return '  No reference data available for dataset.'
        else:
            output = ''
            if self.params.get_path('evaluator.scene_handling') == 'scene-dependent':
                overall_metrics_per_scene = {}
                for scene_id, scene_label in enumerate(self.dataset.scene_labels):
                    if scene_label not in overall_metrics_per_scene:
                        overall_metrics_per_scene[scene_label] = {}

                    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
                        event_label_list=self.dataset.event_labels(scene_label=scene_label),
                        time_resolution=1.0,
                    )

                    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
                        event_label_list=self.dataset.event_labels(scene_label=scene_label),
                        evaluate_onset=True,
                        evaluate_offset=False,
                        t_collar=0.5,
                        percentage_of_length=0.5
                    )

                    for fold in self._get_active_folds():
                        result_filename = self._get_result_filename(fold=fold,
                                                                    scene_label=scene_label,
                                                                    path=self.params.get_path('path.recognizer'))

                        results = MetaDataContainer().load(filename=result_filename)
                        #print(result_filename)
                        #print(self.dataset.test(1, scene_label=scene_label))

                        for file_id, item in enumerate(self.dataset.test(fold, scene_label=scene_label)):

                            # Select only row which are from current file and contains only detected event
                            #print(self.dataset.absolute_to_relative(item['file']))
                            #exit(0)
                            #print (file_id)  ### a012.wav
                            #print (item)
                            #exit(0)
                            #print(self.dataset.relative_to_absolute_path(item['file']))
                            current_file_results = []
                            for result_item in results.filter(filename=self.dataset.absolute_to_relative(item['file'])):                                
                                #print(result_item)
                                if 'event_label' in result_item and result_item.event_label:
                                    current_file_results.append(result_item)

                              
                        
                            #print(file)
                            #exit(0)
                            meta = [] 
                            
                            #print('processing META--------------------------------------------------------------')   
                            #print(self.dataset.absolute_to_relative(item['file']))
                            #print(self.dataset.absolute_to_relative(item['file']))
                            #print(self.dataset.file_meta('audio/street/a099.wav'))
                            #print(self.dataset.file_meta(self.dataset.absolute_to_relative(item['file'])))
                            #exit(0)     

                            for meta_item in self.dataset.file_meta(self.dataset.absolute_to_relative(item['file'])):                                
                                if 'event_label' in meta_item and meta_item.event_label:
                                    meta.append(meta_item)                           
                            segment_based_metric.evaluate(
                                reference_event_list=meta,
                                estimated_event_list=current_file_results
                            )

                            event_based_metric.evaluate(
                                reference_event_list=meta,
                                estimated_event_list=current_file_results
                            )

                    overall_metrics_per_scene[scene_label]['segment_based_metrics'] = segment_based_metric.results()
                    overall_metrics_per_scene[scene_label]['event_based_metrics'] = event_based_metric.results()
                    if self.params.get_path('evaluator.show_details', False):
                        output += "  Scene [{scene}], Evaluation over {folds:d} folds\n".format(
                            scene=scene_label,
                            folds=self.dataset.fold_count
                        )

                        output += " \n"
                        output += segment_based_metric.result_report_overall()
                        output += segment_based_metric.result_report_class_wise()
                overall_metrics_per_scene = DottedDict(overall_metrics_per_scene)

                output += " \n"
                output += "  Overall metrics \n"
                output += "  =============== \n"
                output += "    {event_label:<17s} | {segment_based_fscore:7s} | {segment_based_er:7s} | {event_based_fscore:7s} | {event_based_er:7s} | \n".format(
                    event_label='Event label',
                    segment_based_fscore='Seg. F1',
                    segment_based_er='Seg. ER',
                    event_based_fscore='Evt. F1',
                    event_based_er='Evt. ER',
                )
                output += "    {event_label:<17s} + {segment_based_fscore:7s} + {segment_based_er:7s} + {event_based_fscore:7s} + {event_based_er:7s} + \n".format(
                    event_label='-' * 17,
                    segment_based_fscore='-' * 7,
                    segment_based_er='-' * 7,
                    event_based_fscore='-' * 7,
                    event_based_er='-' * 7,
                )
                avg = {
                    'segment_based_fscore': [],
                    'segment_based_er': [],
                    'event_based_fscore': [],
                    'event_based_er': [],
                }
                for scene_id, scene_label in enumerate(self.dataset.scene_labels):
                    output += "    {scene_label:<17s} | {segment_based_fscore:<7s} | {segment_based_er:<7s} | {event_based_fscore:<7s} | {event_based_er:<7s} | \n".format(
                        scene_label=scene_label,
                        segment_based_fscore="{:4.2f}".format(overall_metrics_per_scene.get_path(scene_label + '.segment_based_metrics.overall.f_measure.f_measure') * 100),
                        segment_based_er="{:4.2f}".format(overall_metrics_per_scene.get_path(scene_label + '.segment_based_metrics.overall.error_rate.error_rate')),
                        event_based_fscore="{:4.2f}".format(overall_metrics_per_scene.get_path(scene_label + '.event_based_metrics.overall.f_measure.f_measure') * 100),
                        event_based_er="{:4.2f}".format(overall_metrics_per_scene.get_path(scene_label + '.event_based_metrics.overall.error_rate.error_rate')),
                    )

                    avg['segment_based_fscore'].append(overall_metrics_per_scene.get_path(scene_label + '.segment_based_metrics.overall.f_measure.f_measure') * 100)
                    avg['segment_based_er'].append(overall_metrics_per_scene.get_path(scene_label + '.segment_based_metrics.overall.error_rate.error_rate'))
                    avg['event_based_fscore'].append(overall_metrics_per_scene.get_path(scene_label + '.event_based_metrics.overall.f_measure.f_measure') * 100)
                    avg['event_based_er'].append(overall_metrics_per_scene.get_path(scene_label + '.event_based_metrics.overall.error_rate.error_rate'))

                output += "    {scene_label:<17s} + {segment_based_fscore:7s} + {segment_based_er:7s} + {event_based_fscore:7s} + {event_based_er:7s} + \n".format(
                    scene_label='-' * 17,
                    segment_based_fscore='-' * 7,
                    segment_based_er='-' * 7,
                    event_based_fscore='-' * 7,
                    event_based_er='-' * 7,
                )
                output += "    {scene_label:<17s} | {segment_based_fscore:<7s} | {segment_based_er:<7s} | {event_based_fscore:<7s} | {event_based_er:<7s} | \n".format(
                    scene_label='Average',
                    segment_based_fscore="{:4.2f}".format(numpy.mean(avg['segment_based_fscore'])),
                    segment_based_er="{:4.2f}".format(numpy.mean(avg['segment_based_er'])),
                    event_based_fscore="{:4.2f}".format(numpy.mean(avg['event_based_fscore'])),
                    event_based_er="{:4.2f}".format(numpy.mean(avg['event_based_er'])),
                )

            elif self.params.get_path('evaluator.scene_handling') == 'scene-independent':
                message = '{name}: Scene handling mode not implemented yet [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('evaluator.scene_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

            else:
                message = '{name}: Unknown scene handling mode [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('evaluator.scene_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

            if self.params.get_path('evaluator.saving.enable'):
                filename = self.params.get_path('evaluator.saving.filename').format(
                    dataset_name=self.dataset.storage_name,
                    parameter_set=self.params['active_set'],
                    parameter_hash=self.params['_hash']
                )

                output_file = os.path.join(self.params.get_path('path.evaluator'), filename)

                output_data = {
                    'overall_metrics_per_scene': overall_metrics_per_scene,
                    'average': {
                        'segment_based_fscore': numpy.mean(avg['segment_based_fscore']),
                        'segment_based_er': numpy.mean(avg['segment_based_er']),
                        'event_based_fscore': numpy.mean(avg['event_based_fscore']),
                        'event_based_er': numpy.mean(avg['event_based_er']),
                    },
                    'parameters': dict(self.params)
                }
                ParameterFile(output_data, filename=output_file).save()

            return output

    @staticmethod
    def _get_feature_filename(audio_file, path, extension='cpickle'):
        """Get feature filename

        Parameters
        ----------
        audio_file : str
            audio file name from which the features are extracted
        path :  str
            feature path
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        feature_filename : str, dict
            full feature filename or dict of filenames

        """

        audio_filename = os.path.split(audio_file)[1]
        if isinstance(path, dict):
            paths = {}
            for key, value in iteritems(path):
                paths[key] = os.path.join(value, 'sequence_' + os.path.splitext(audio_filename)[0] + '.' + extension)
            return paths

        else:
            return os.path.join(path, 'sequence_' + os.path.splitext(audio_filename)[0] + '.' + extension)

    @staticmethod
    def _get_feature_normalizer_filename(fold, path, scene_label='all', extension='cpickle'):
        """Get normalizer filename

        Parameters
        ----------
        fold : int >= 0
            evaluation fold number
        path :  str
            normalizer path
        scene_label : str
            Scene label
            Default value "all"
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        normalizer_filename : str
            full normalizer filename

        """

        if isinstance(path, dict):
            paths = {}
            for key, value in iteritems(path):
                paths[key] = os.path.join(value, 'scale_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)
            return paths
        elif isinstance(path, list):
            paths = []
            for value in path:
                paths.append(os.path.join(value, 'scale_fold' + str(fold) + '_' + str(scene_label) + '.' + extension))
            return paths
        else:
            return os.path.join(path, 'scale_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)

    @staticmethod
    def _get_model_filename(fold, path, scene_label='all', extension='cpickle'):
        """Get model filename

        Parameters
        ----------
        fold : int >= 0
            evaluation fold number
        path :  str
            model path
        scene_label : str
            Scene label
            Default value "all"
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        model_filename : str
            full model filename

        """

        return os.path.join(path, 'model_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)

    @staticmethod
    def _get_result_filename(fold, path, scene_label='all', extension='txt'):
        """Get result filename

        Parameters
        ----------
        fold : int >= 0
            evaluation fold number
        path :  str
            result path
        scene_label : str
            Scene label
            Default value "all"
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        result_filename : str
            full result filename

        """

        if fold == 0:
            return os.path.join(path, 'results' + '_' + str(scene_label) + '.' + extension)
        else:
            return os.path.join(path, 'results_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)


class BinarySoundEventAppCore(SoundEventAppCore):
    def show_eval(self):

        eval_path = self.params.get_path('path.evaluator')

        eval_files = []
        for filename in os.listdir(eval_path):
            if filename.endswith('.yaml'):
                eval_files.append(os.path.join(eval_path, filename))

        eval_data = {}
        for filename in eval_files:
            data = DottedDict(ParameterFile().load(filename=filename))
            set_id = data.get_path('parameters.set_id')
            if set_id not in eval_data:
                eval_data[set_id] = {}
            params_hash = data.get_path('parameters._hash')

            if params_hash not in eval_data[set_id]:
                eval_data[set_id][params_hash] = data

        output = ''
        output += '  Evaluated systems\n'
        output += '  {set_id:<20s} | {er:8s} | {f1:8s} | {desc:32s} | {hash:32s} |\n'.format(
            set_id='Set id',
            desc='Description',
            hash='Hash',
            er='Evt ER',
            f1='Evt F1'
        )

        output += '  {set_id:<20s} + {er:8s} + {f1:8s} + {desc:32s} + {hash:32s} +\n'.format(
            set_id='-' * 20,
            desc='-' * 32,
            hash='-' * 32,
            er='-' * 8,
            f1='-' * 8
        )

        for set_id in sorted(list(eval_data.keys())):
            for params_hash in eval_data[set_id]:
                data = eval_data[set_id][params_hash]
                desc = data.get_path('parameters.description')
                output += '  {set_id:<20s} | {er:<8s} | {f1:<8s} | {desc:32s} | {hash:32s} |\n'.format(
                    set_id=set_id,
                    desc=(desc[:44] + '..') if len(desc) > 44 else desc,
                    hash=params_hash,
                    er='{0:2.2f} %'.format(data.get_path('average.event_based_er')),
                    f1='{0:4.2f} %'.format(data.get_path('average.event_based_fscore'))
                )
        self.ui.line(output)

    @before_and_after_function_wrapper
    def feature_extraction(self, files=None, overwrite=None):
        
        """Feature extraction stage

        Parameters
        ----------
        files : list
            file list

        overwrite : bool
            overwrite existing feature files
            (Default value=False)

        Returns
        -------
        None

        """
        
        if not files:
            files = []
            
            for event_label in self.dataset.event_labels:
                for fold in self._get_active_folds():
                    for item_id, item in enumerate(self.dataset.train(fold, event_label=event_label)):
                        if item['file'] not in files:
                            files.append(item['file'])
                    for item_id, item in enumerate(self.dataset.test(fold, event_label=event_label)):
                        if item['file'] not in files:
                            files.append(item['file'])
            files = sorted(files)
            

        if not overwrite:            
            overwrite = self.params.get_path('general.overwrite', False)

        
        feature_files = []
        feature_extractor = self.FeatureExtractor(overwrite=overwrite, store=True)
        for file_id, audio_filename in enumerate(tqdm(files, desc='           {0:<15s}'.format('Extracting features '),
                                                      file=sys.stdout,
                                                      leave=False,
                                                      disable=self.disable_progress_bar)):

            if self.log_system_progress:
                self.logger.info('  {title:<15s} [{file_id:d}/{total:d}] {file:<30s}'.format(
                    title='Extracting features ',
                    file_id=file_id,
                    total=len(files),
                    file=os.path.split(audio_filename)[-1])
                )

            # Get feature filename
            current_feature_files = self._get_feature_filename(
                audio_file=os.path.split(audio_filename)[1],
                path=self.params.get_path('path.feature_extractor')
            )
            

            if not filelist_exists(current_feature_files) or overwrite:
                feature_extractor.extract(
                    audio_file=self.dataset.relative_to_absolute_path(audio_filename),
                    extractor_params=DottedDict(self.params.get_path('feature_extractor.parameters')),
                    storage_paths=current_feature_files
                )

            feature_files.append(current_feature_files)
        return feature_files

    @before_and_after_function_wrapper
    def feature_normalization(self, overwrite=None):
        """Feature normalization stage

        Calculated normalization factors for each evaluation fold based on the training material available.

        Parameters
        ----------
        overwrite : bool
            overwrite existing normalizers
            (Default value=False)

        Raises
        -------
        ValueError:
            Unknown scene_handling mode
        IOError:
            Feature file not found.

        Returns
        -------
        None

        """

        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        if self.params.get_path('feature_normalizer.enable', False) and self.params.get_path('feature_normalizer.method', 'global') == 'global':
            if self.params.get_path('feature_normalizer.event_handling') == 'event-dependent':

                fold_progress = tqdm(self._get_active_folds(),
                                     desc='           {0:<15s}'.format('Fold '),
                                     file=sys.stdout,
                                     leave=False,
                                     miniters=1,
                                     disable=self.disable_progress_bar)

                for fold in fold_progress:
                    if self.log_system_progress:
                        self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(title='Fold',
                                                                                      fold=fold,
                                                                                      total=len(fold_progress)))
                    for event_label in self.dataset.event_labels:
                        current_normalizer_files = self._get_feature_normalizer_filename(fold=fold,
                                                                                         path=self.params.get_path(
                                                                                            'path.feature_normalizer'),
                                                                                         event_label=event_label
                                                                                         )

                        method_progress = tqdm(current_normalizer_files,
                                               desc='           {0: >15s}'.format('Feature method '),
                                               file=sys.stdout,
                                               leave=False,
                                               miniters=1,
                                               disable=self.disable_progress_bar)

                        for method in method_progress:
                            current_normalizer_file = current_normalizer_files[method]
                            if not os.path.isfile(current_normalizer_file) or overwrite:
                                normalizer = FeatureNormalizer()
                                item_progress = tqdm(self.dataset.train(fold, event_label=event_label),
                                                     desc="           {0: >15s}".format('Collect data '),
                                                     file=sys.stdout,
                                                     leave=False,
                                                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                                     disable=self.disable_progress_bar
                                                     )
                                for item_id, item in enumerate(item_progress):
                                    feature_filename = self._get_feature_filename(audio_file=item['file'],
                                                                                  path=self.params.get_path(
                                                                                     'path.feature_extractor', {})[method])
                                    if self.log_system_progress:
                                        self.logger.info(
                                            '  {title:<15s} [{item_id:d}/{total:d}] {file:<30s}'.format(
                                                title='Item',
                                                item_id=item_id,
                                                total=len(item_progress),
                                                file=os.path.split(feature_filename)[-1])
                                            )

                                    if os.path.isfile(feature_filename):
                                        feature_stats = FeatureContainer(filename=feature_filename)
                                    else:
                                        message = '{name}: Features not found [{file}]'.format(
                                            name=self.__class__.__name__,
                                            method=feature_filename
                                        )

                                        self.logger.exception(message)
                                        raise IOError(message)

                                    # Accumulate statistics
                                    normalizer.accumulate(feature_stats)

                                # Calculate normalization factors
                                normalizer.finalize().save(filename=current_normalizer_file)
            elif self.params.get_path('feature_normalizer.event_handling') == 'event-independent':
                message = '{name}: Event handling mode not implemented yet [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('feature_normalizer.event_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

            else:
                message = '{name}: Unknown event handling mode [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('feature_normalizer.event_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

    @before_and_after_function_wrapper
    def system_training(self, overwrite=None):
        """System training stage

        Parameters
        ----------

        overwrite : bool
            overwrite existing models
            (Default value=False)

        Raises
        -------
        ValueError:
            Unknown event_handling mode
        IOError:
            Feature normalizer not found.
            Feature file not found.

        Returns
        -------
        None

        """
        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        if self.params.get_path('learner.event_handling') == 'event-dependent':
            fold_progress = tqdm(self._get_active_folds(),
                                 desc='           {0:<15s}'.format('Fold '),
                                 file=sys.stdout,
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar)

            for fold in fold_progress:
                if self.log_system_progress:
                    self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(title='Fold',
                                                                                  fold=fold,
                                                                                  total=len(fold_progress)))
                event_progress = tqdm(self.dataset.event_labels,
                                      desc="           {0: >15s}".format('Event '),
                                      file=sys.stdout,
                                      leave=False,
                                      miniters=1,
                                      disable=self.disable_progress_bar)
                for event_label in event_progress:
                    current_model_file = self._get_model_filename(fold=fold,
                                                                  path=self.params.get_path('path.learner'),
                                                                  event_label=event_label)
                    if not os.path.isfile(current_model_file) or overwrite:
                        # Feature stacker
                        feature_stacker = FeatureStacker(recipe=self.params.get_path('feature_extractor.recipe'))

                        feature_aggregator = None
                        if self.params.get_path('feature_aggregator.enable'):
                            feature_aggregator = FeatureAggregator(
                                recipe=self.params.get_path('feature_aggregator.aggregation_recipe'),
                                win_length_frames=self.params.get_path('feature_aggregator.win_length_frames'),
                                hop_length_frames=self.params.get_path('feature_aggregator.hop_length_frames')
                            )

                        feature_normalizer = None
                        if self.params.get_path('feature_normalizer.enable'):
                            # Load normalizers
                            feature_normalizer_filenames = self._get_feature_normalizer_filename(
                                fold=fold,
                                path=self.params.get_path('path.feature_normalizer'),
                                event_label=event_label
                            )

                            normalizer_list = {}
                            for method, feature_normalizer_filename in iteritems(feature_normalizer_filenames):
                                if os.path.isfile(feature_normalizer_filename):
                                    normalizer_list[method] = FeatureNormalizer().load(
                                        filename=feature_normalizer_filename
                                    )

                                else:
                                    message = '{name}: Feature normalizer not found [{file}]'.format(
                                        name=self.__class__.__name__,
                                        method=feature_normalizer_filename
                                    )

                                    self.logger.exception(message)
                                    raise IOError(message)

                            feature_normalizer = self.FeatureNormalizer(feature_stacker.normalizer(
                                normalizer_list=normalizer_list)
                            )

                        # Collect training examples
                        train_meta = self.dataset.train(fold, event_label=event_label)
                        data = {}
                        annotations = {}

                        item_progress = tqdm(train_meta.file_list,
                                             desc="           {0: >15s}".format('Collect data '),
                                             file=sys.stdout,
                                             leave=False,
                                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                             disable=self.disable_progress_bar
                                             )
                        for item_id, audio_filename in enumerate(item_progress):
                            
                            if self.log_system_progress:
                                self.logger.info(
                                    '  {title:<15s} [{item_id:d}/{total:d}] {item:<20s}'.format(
                                        title='Collect data ',
                                        item_id=item_id,
                                        total=len(item_progress),
                                        item=os.path.split(audio_filename)[-1])
                                    )

                            item_progress.set_postfix(file=os.path.splitext(os.path.split(audio_filename)[-1])[0])
                            #item_progress.update()

                            # Load features
                            feature_filenames = self._get_feature_filename(
                                audio_file=audio_filename,
                                path=self.params.get_path('path.feature_extractor')
                            )

                            feature_list = {}
                            for method, feature_filename in iteritems(feature_filenames):
                                if os.path.isfile(feature_filename):
                                    feature_list[method] = FeatureContainer().load(filename=feature_filename)
                                else:
                                    message = '{name}: Features not found [{file}]'.format(
                                        name=self.__class__.__name__,
                                        method=feature_filename
                                    )

                                    self.logger.exception(message)
                                    raise IOError(message)

                            # Stack features
                            feature_data = feature_stacker.feature_vector(feature_repository=feature_list)

                            # Normalize features
                            if feature_normalizer:
                                feature_data = feature_normalizer.normalize(feature_data)

                            # Aggregate features
                            if feature_aggregator:
                                feature_data = feature_aggregator.process(feature_container=feature_data)

                            data[audio_filename] = feature_data
                            annotations[audio_filename] = train_meta.filter(filename=audio_filename)               

                        
                        fc = self._get_learner(method=self.params.get_path('learner.method'),
                                               class_labels=[event_label],
                                               feature_normalizer=feature_normalizer,
                                               feature_stacker=feature_stacker,
                                               feature_aggregator=feature_aggregator,
                                               params=self.params.get_path('learner'),
                                               filename=current_model_file,
                                               disable_progress_bar=self.disable_progress_bar,
                                               log_progress=self.log_system_progress)
                        fc.learn(data=data, annotations=annotations)
                        fc.save()

                    if self.params.get_path('learner.show_model_information'):
                        # Load class model container
                        model_filename = self._get_model_filename(fold=fold,
                                                                  path=self.params.get_path('path.learner'),
                                                                  event_label=event_label)

                        if os.path.isfile(model_filename):
                            model_container = self._get_learner(method=self.params.get_path('learner.method')).load(
                                filename=model_filename
                            )

                        else:
                            message = '{name}: Model file not found [{file}]'.format(
                                name=self.__class__.__name__,
                                method=model_filename
                            )

                            self.logger.exception(message)
                            raise IOError(message)

                        if 'learning_history' in model_container:
                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                            ax1 = fig.add_subplot(211)
                            ax2 = fig.add_subplot(212)

                            ax.spines['top'].set_color('none')
                            ax.spines['bottom'].set_color('none')
                            ax.spines['left'].set_color('none')
                            ax.spines['right'].set_color('none')
                            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

                            # Set common labels
                            ax.set_xlabel('Epochs')

                            ax1.set_title('model accuracy')
                            if 'fmeasure' in model_container['learning_history']:
                                ax1.set_title('model accuracy / fmeasure')
                                ax1.plot(model_container['learning_history']['fmeasure'])
                                ax1.set_ylabel('fmeasure')
                                if 'val_fmeasure' in model_container['learning_history']:
                                    ax1.plot(model_container['learning_history']['val_fmeasure'])
                                    ax1.legend(['train', 'validation'], loc='upper left')
                                else:
                                    ax1.legend(['train'], loc='upper left')
                            elif 'binary_accuracy' in model_container['learning_history']:
                                ax1.set_title('model accuracy / binary_accuracy')
                                ax1.plot(model_container['learning_history']['binary_accuracy'])
                                ax1.set_ylabel('binary_accuracy')
                                if 'val_binary_accuracy' in model_container['learning_history']:
                                    ax1.plot(model_container['learning_history']['val_binary_accuracy'])
                                    ax1.legend(['train', 'validation'], loc='upper left')
                                else:
                                    ax1.legend(['train'], loc='upper left')
                            elif 'categorical_accuracy' in model_container['learning_history']:
                                ax1.set_title('model accuracy / categorical_accuracy')
                                ax1.plot(model_container['learning_history']['categorical_accuracy'])
                                ax1.set_ylabel('categorical_accuracy')
                                if 'val_categorical_accuracy' in model_container['learning_history']:
                                    ax1.plot(model_container['learning_history']['val_categorical_accuracy'])
                                    ax1.legend(['train', 'validation'], loc='upper left')
                                else:
                                    ax1.legend(['train'], loc='upper left')

                            if 'loss' in model_container['learning_history']:
                                ax2.set_title('model loss')
                                ax2.plot(model_container['learning_history']['loss'])
                                ax2.set_ylabel('loss')
                                if 'val_loss' in model_container['learning_history']:
                                    ax2.plot(model_container['learning_history']['val_loss'])
                                    ax2.legend(['train', 'validation'], loc='upper left')
                                else:
                                    ax2.legend(['train'], loc='upper left')
                            plt.show()

        elif self.params.get_path('learner.event_handling') == 'event-independent':
            message = '{name}: Event handling mode not implemented yet [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('learner.event_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

        else:
            message = '{name}: Unknown event handling mode [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('learner.event_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)


    @before_and_after_function_wrapper
    def system_testing(self, overwrite=None):
        """System testing stage

        If extracted features are not found from disk, they are extracted but not saved.

        Parameters
        ----------
        overwrite : bool
            overwrite existing models
            (Default value=False)

        Raises
        -------
        ValueError:
            Unknown event_handling mode
        IOError:
            Model file not found.
            Audio file not found.

        Returns
        -------
        None

        """

        if not overwrite:
            overwrite = self.params.get_path('general.overwrite', False)

        if self.params.get_path('recognizer.event_handling') == 'event-dependent':

            fold_progress = tqdm(self._get_active_folds(),
                                 desc="           {0: >15s}".format('Fold '),
                                 file=sys.stdout,
                                 leave=False,
                                 miniters=1,
                                 disable=self.disable_progress_bar)

            for fold in fold_progress:
                if self.log_system_progress:
                    self.logger.info('  {title:<15s} [{fold:d}/{total:d}]'.format(title='Fold',
                                                                                  fold=fold,
                                                                                  total=len(fold_progress)))

                event_progress = tqdm(self.dataset.event_labels,
                                      desc="           {0: >15s}".format('Event '),
                                      file=sys.stdout,
                                      leave=False,
                                      miniters=1,
                                      disable=self.disable_progress_bar)
                for event_label in event_progress:
                    current_result_file = self._get_result_filename(fold=fold,
                                                                    path=self.params.get_path('path.recognizer'),
                                                                    event_label=event_label
                                                                    )
                    if not os.path.isfile(current_result_file) or overwrite:
                        results = MetaDataContainer(filename=current_result_file)

                        # Load class model container
                        model_filename = self._get_model_filename(fold=fold,
                                                                  path=self.params.get_path('path.learner'),
                                                                  event_label=event_label
                                                                  )

                        if os.path.isfile(model_filename):
                            model_container = self._get_learner(method=self.params.get_path('learner.method')).load(
                                filename=model_filename)
                        else:
                            message = '{name}: Model file not found [{file}]'.format(
                                name=self.__class__.__name__,
                                method=model_filename
                            )

                            self.logger.exception(message)
                            raise IOError(message)

                        item_progress = tqdm(self.dataset.test(fold, event_label=event_label),
                                             desc="           {0: >15s}".format('Testing '),
                                             file=sys.stdout,
                                             leave=False,
                                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ',
                                             disable=self.disable_progress_bar
                                             )
                        for item_id, item in enumerate(item_progress):
                            if self.log_system_progress:
                                self.logger.info(
                                    '  {title:<15s} [{item_id:d}/{total:d}] {item:<20s}'.format(
                                        title='Testing',
                                        item_id=item_id,
                                        total=len(item_progress),
                                        item=os.path.split(item['file'])[-1])
                                    )

                            # Load features
                            feature_filenames = self._get_feature_filename(
                                audio_file=item['file'],
                                path=self.params.get_path('path.feature_extractor')
                            )

                            feature_list = {}
                            for method, feature_filename in iteritems(feature_filenames):
                                if os.path.isfile(feature_filename):
                                    feature_list[method] = FeatureContainer().load(filename=feature_filename)
                                else:
                                    message = '{name}: Features not found [{file}]'.format(
                                        name=self.__class__.__name__,
                                        method=item['file']
                                    )

                                    self.logger.exception(message)
                                    raise IOError(message)

                            feature_data = model_container.feature_stacker.feature_vector(
                                feature_repository=feature_list)
                          

                            # Normalize features
                            if model_container.feature_normalizer:
                                feature_data = model_container.feature_normalizer.normalize(feature_data)

                            # Aggregate features
                            if model_container.feature_aggregator:
                                feature_data = model_container.feature_aggregator.process(feature_data)

                            current_result = model_container.predict(
                                feature_data=feature_data,
                                recognizer_params=self.params.get_path('recognizer')
                            )

                            if self.params.get_path('recognizer.event_post_processing.enable'):
                                # Event list post-processing
                                current_result = current_result.process_events(
                                    minimum_event_length=self.params.get_path('recognizer.event_post_processing.minimum_event_length_seconds'),
                                    minimum_event_gap=self.params.get_path('recognizer.event_post_processing.minimum_event_gap_seconds')
                                )

                            if current_result:
                                for event in current_result:
                                    event.file = self.dataset.absolute_to_relative(item['file'])
                                    results.append(event)
                            else:
                                results.append(MetaDataItem({'file': self.dataset.absolute_to_relative(item['file'])}))

                        # Save testing results
                        results.save()

        elif self.params.get_path('recognizer.event_handling') == 'event-independent':
            message = '{name}: Event handling mode not implemented yet [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('recognizer.event_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

        else:
            message = '{name}: Unknown event handling mode [{mode}]'.format(
                name=self.__class__.__name__,
                mode=self.params.get_path('recognizer.event_handling')
            )

            self.logger.exception(message)
            raise ValueError(message)

    @before_and_after_function_wrapper
    def system_evaluation(self):
        """System evaluation stage.

        Testing outputs are collected and evaluated.

        Parameters
        ----------

        Returns
        -------
        None

        Raises
        -------
        IOError
            Result file not found

        """

        if not self.dataset.reference_data_present:
            return '  No reference data available for dataset.'
        else:
            output = ''
            if self.params.get_path('evaluator.event_handling', 'event-dependent') == 'event-dependent':
                overall_metrics_per_event = {}
                for event_id, event_label in enumerate(self.dataset.event_labels):
                    if event_label not in overall_metrics_per_event:
                        overall_metrics_per_event[event_label] = {}

                    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
                        event_label_list=[event_label],
                        time_resolution=1.0,
                    )

                    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
                        event_label_list=[event_label],
                        evaluate_onset=True,
                        evaluate_offset=False,
                        t_collar=0.5,
                        percentage_of_length=0.5
                    )

                    for fold in self._get_active_folds():
                        result_filename = self._get_result_filename(fold=fold,
                                                                    event_label=event_label,
                                                                    path=self.params.get_path('path.recognizer'))

                        results = MetaDataContainer().load(filename=result_filename)
                        for file_id, item in enumerate(self.dataset.test(fold, event_label=event_label)):
                            # Select only row which are from current file and contains only detected event
                            current_file_results = []
                            for result_item in results.filter(filename=self.dataset.absolute_to_relative(item['file'])):
                                if 'event_label' in result_item and result_item.event_label:
                                    current_file_results.append(result_item)
                            print('META--------------------------------------------------------------')
                            meta = []
                            for meta_item in self.dataset.file_meta(self.dataset.absolute_to_relative(item['file'])):
                                if 'event_label' in meta_item and meta_item.event_label:
                                    meta.append(meta_item)

                            segment_based_metric.evaluate(
                                reference_event_list=meta,
                                estimated_event_list=current_file_results
                            )

                            event_based_metric.evaluate(
                                reference_event_list=meta,
                                estimated_event_list=current_file_results
                            )

                    overall_metrics_per_event[event_label]['segment_based_metrics'] = segment_based_metric.results()
                    overall_metrics_per_event[event_label]['event_based_metrics'] = event_based_metric.results()
                    if self.params.get_path('evaluator.show_details', False):
                        output += "  Event [{event}], Evaluation over {folds:d} folds\n".format(event=event_label,
                                                                                                folds=self.dataset.fold_count)
                        output += " \n"

                        output += "  Event-based metrics \n"
                        output += event_based_metric.result_report_overall()
                        output += event_based_metric.result_report_class_wise()

                        output += "  Segment-based metrics \n"
                        output += segment_based_metric.result_report_overall()
                        output += segment_based_metric.result_report_class_wise()

                overall_metrics_per_event = DottedDict(overall_metrics_per_event)

                output += " \n"
                output += "  Overall metrics \n"
                output += "  =============== \n"
                output += "    {event_label:<17s} | {event_based_fscore:7s} | {event_based_er:7s} | {segment_based_fscore:7s} | {segment_based_er:7s} |\n".format(
                    event_label='Event label',
                    segment_based_fscore='Seg. F1',
                    segment_based_er='Seg. ER',
                    event_based_fscore='Evt. F1',
                    event_based_er='Evt. ER',
                )
                output += "    {event_label:<17s} + {event_based_fscore:7s} + {event_based_er:7s} + {segment_based_fscore:7s} + {segment_based_er:7s} +\n".format(
                    event_label='-'*17,
                    segment_based_fscore='-'*7,
                    segment_based_er='-' * 7,
                    event_based_fscore='-' * 7,
                    event_based_er='-' * 7,
                )
                avg = {
                    'segment_based_fscore': [],
                    'segment_based_er': [],
                    'event_based_fscore': [],
                    'event_based_er': [],
                }
                for event_id, event_label in enumerate(self.dataset.event_labels):
                    output += "    {event_label:<17s} | {event_based_fscore:<7s} | {event_based_er:<7s} | {segment_based_fscore:<7s} | {segment_based_er:<7s} |\n".format(
                        event_label=event_label,
                        segment_based_fscore="{:4.2f}".format(overall_metrics_per_event.get_path(event_label+'.segment_based_metrics.overall.f_measure.f_measure')*100),
                        segment_based_er="{:4.2f}".format(overall_metrics_per_event.get_path(event_label+'.segment_based_metrics.overall.error_rate.error_rate')),
                        event_based_fscore="{:4.2f}".format(overall_metrics_per_event.get_path(event_label + '.event_based_metrics.overall.f_measure.f_measure') * 100),
                        event_based_er="{:4.2f}".format(overall_metrics_per_event.get_path(event_label + '.event_based_metrics.overall.error_rate.error_rate')),
                    )

                    avg['segment_based_fscore'].append(overall_metrics_per_event.get_path(event_label+'.segment_based_metrics.overall.f_measure.f_measure')*100)
                    avg['segment_based_er'].append(overall_metrics_per_event.get_path(event_label+'.segment_based_metrics.overall.error_rate.error_rate'))
                    avg['event_based_fscore'].append(overall_metrics_per_event.get_path(event_label + '.event_based_metrics.overall.f_measure.f_measure') * 100)
                    avg['event_based_er'].append(overall_metrics_per_event.get_path(event_label + '.event_based_metrics.overall.error_rate.error_rate'))

                output += "    {event_label:<17s} + {event_based_fscore:7s} + {event_based_er:7s} + {segment_based_fscore:7s} + {segment_based_er:7s} +\n".format(
                    event_label='-' * 17,
                    segment_based_fscore='-' * 7,
                    segment_based_er='-' * 7,
                    event_based_fscore='-' * 7,
                    event_based_er='-' * 7,
                )
                output += "    {event_label:<17s} | {event_based_fscore:<7s} | {event_based_er:<7s} | {segment_based_fscore:<7s} | {segment_based_er:<7s} |\n".format(
                    event_label='Average',
                    segment_based_fscore="{:4.2f}".format(numpy.mean(avg['segment_based_fscore'])),
                    segment_based_er="{:4.2f}".format(numpy.mean(avg['segment_based_er'])),
                    event_based_fscore="{:4.2f}".format(numpy.mean(avg['event_based_fscore'])),
                    event_based_er="{:4.2f}".format(numpy.mean(avg['event_based_er'])),
                )

            elif self.params.get_path('evaluator.event_handling') == 'event-independent':
                message = '{name}: Event handling mode not implemented yet [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('evaluator.event_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

            else:
                message = '{name}: Unknown event handling mode [{mode}]'.format(
                    name=self.__class__.__name__,
                    mode=self.params.get_path('evaluator.event_handling')
                )

                self.logger.exception(message)
                raise ValueError(message)

            if self.params.get_path('evaluator.saving.enable'):
                filename = self.params.get_path('evaluator.saving.filename').format(
                    dataset_name=self.dataset.storage_name,
                    parameter_set=self.params['active_set'],
                    parameter_hash=self.params['_hash'],
                    )
                output_file = os.path.join(self.params.get_path('path.evaluator'), filename)

                output_data = {
                    'overall_metrics_per_event': overall_metrics_per_event,
                    'average': {
                        'segment_based_fscore': numpy.mean(avg['segment_based_fscore']),
                        'segment_based_er': numpy.mean(avg['segment_based_er']),
                        'event_based_fscore': numpy.mean(avg['event_based_fscore']),
                        'event_based_er': numpy.mean(avg['event_based_er']),
                    },
                    'parameters': dict(self.params)
                }
                ParameterFile(output_data, filename=output_file).save()

            return output

    @staticmethod
    def _get_feature_filename(audio_file, path, extension='cpickle'):
        """Get feature filename

        Parameters
        ----------
        audio_file : str
            audio file name from which the features are extracted
        path :  str
            feature path
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        feature_filename : str, dict
            full feature filename or dict of filenames

        """

        audio_filename = os.path.split(audio_file)[1]
        if isinstance(path, dict):
            paths = {}
            for key, value in iteritems(path):
                paths[key] = os.path.join(value, 'sequence_' + os.path.splitext(audio_filename)[0] + '.' + extension)
            return paths

        else:
            return os.path.join(path, 'sequence_' + os.path.splitext(audio_filename)[0] + '.' + extension)

    @staticmethod
    def _get_feature_normalizer_filename(fold, path, event_label='all', extension='cpickle'):
        """Get normalizer filename

        Parameters
        ----------
        fold : int >= 0
            evaluation fold number
        path :  str
            normalizer path
        event_label : str
            Scene label
            Default value "all"
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        normalizer_filename : str
            full normalizer filename

        """

        if isinstance(path, dict):
            paths = {}
            for key, value in iteritems(path):
                paths[key] = os.path.join(value, 'scale_fold' + str(fold) + '_' + str(event_label) + '.' + extension)
            return paths
        elif isinstance(path, list):
            paths = []
            for value in path:
                paths.append(os.path.join(value, 'scale_fold' + str(fold) + '_' + str(event_label) + '.' + extension))
            return paths
        else:
            return os.path.join(path, 'scale_fold' + str(fold) + '_' + str(event_label) + '.' + extension)

    @staticmethod
    def _get_model_filename(fold, path, event_label='all', extension='cpickle'):
        """Get model filename

        Parameters
        ----------
        fold : int >= 0
            evaluation fold number
        path :  str
            model path
        scene_label : str
            Scene label
            Default value "all"
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        model_filename : str
            full model filename

        """

        return os.path.join(path, 'model_fold' + str(fold) + '_' + str(event_label) + '.' + extension)

    @staticmethod
    def _get_result_filename(fold, path, event_label='all', extension='txt'):
        """Get result filename

        Parameters
        ----------
        fold : int >= 0
            evaluation fold number
        path :  str
            result path
        scene_label : str
            Scene label
            Default value "all"
        extension : str
            file extension
            Default value "cpickle"

        Returns
        -------
        result_filename : str
            full result filename

        """

        if fold == 0:
            return os.path.join(path, 'results' + '_' + str(event_label) + '.' + extension)
        else:
            return os.path.join(path, 'results_fold' + str(fold) + '_' + str(event_label) + '.' + extension)

