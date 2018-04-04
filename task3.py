#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCASE 2017::Sound Event Detection in Real-life Audio / Baseline System


from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import numpy
import argparse
import textwrap

from dcase_framework.application_core import SoundEventAppCore
from dcase_framework.parameters import ParameterContainer
from dcase_framework.utils import *

from IPython import embed

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


class Task3AppCore(SoundEventAppCore):
    pass


def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2017
            Task 3: Sound Event Detection in Real-life Audio
            Baseline System
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Toni Heittola ( toni.heittola@tut.fi )

            System description
                This is an baseline implementation for the D-CASE 2016, task 3 - Sound event detection in real life audio.
                The system has binary classifier for each included sound event class. The GMM classifier is trained with
                the positive and negative examples from the mixture signals, and classification is done between these
                two models as likelihood ratio. Acoustic features are MFCC+Delta+Acceleration (MFCC0 omitted).

        '''))

    # Setup argument handling
    parser.add_argument('-m', '--mode',
                        choices=('dev', 'challenge'),
                        default=None,
                        help="Selector for system mode",
                        required=False,
                        dest='mode',
                        type=str)

    parser.add_argument('-p', '--parameters',
                        help='parameter file override',
                        dest='parameter_override',
                        required=False,
                        metavar='FILE',
                        type=argument_file_exists)

    parser.add_argument('-s', '--parameter_set',
                        help='Parameter set id',
                        dest='parameter_set',
                        required=False,
                        type=str)

    parser.add_argument("-n", "--node",
                        help="Node mode",
                        dest="node_mode",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_sets",
                        help="List of available parameter sets",
                        dest="show_set_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_datasets",
                        help="List of available datasets",
                        dest="show_dataset_list",
                        action='store_true',
                        required=False)

    parser.add_argument("-show_parameters",
                        help="Show parameters",
                        dest="show_parameters",
                        action='store_true',
                        required=False)

    parser.add_argument("-o", "--overwrite",
                        help="Overwrite mode",
                        dest="overwrite",
                        action='store_true',
                        required=False)

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)

    # Parse arguments
    args = parser.parse_args()

    # Load default parameters from a file
    default_parameters_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parameters',
                                               os.path.splitext(os.path.basename(__file__))[0]+'.defaults.yaml')
    if args.parameter_set:
        parameters_sets = args.parameter_set.split(',')
    else:
        parameters_sets = [None]

    for parameter_set in parameters_sets:
        # Initialize ParameterContainer
        params = ParameterContainer(project_base=os.path.dirname(os.path.realpath(__file__)))

        # Load default parameters from a file
        params.load(filename=default_parameters_filename)

        if args.parameter_override:
            # Override parameters from a file
            params.override(override=args.parameter_override)

        if parameter_set:
            # Override active_set
            params['active_set'] = parameter_set

        # Process parameters
        params.process()

        # Force overwrite
        if args.overwrite:
            params['general']['overwrite'] = True

        # Override dataset mode from arguments
        if args.mode == 'dev':
            # Set dataset to development
            params['dataset']['method'] = 'development'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        elif args.mode == 'challenge':
            # Set dataset to training set for challenge
            params['dataset']['method'] = 'challenge_train'
            params['general']['challenge_submission_mode'] = True

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters(section='dataset')

        if args.node_mode:
            params['general']['log_system_progress'] = True
            params['general']['print_system_progress'] = False

        # Setup logging
        setup_logging(parameter_container=params['logging'])
        app = Task3AppCore(name='DCASE 2017::Sound Event Detection in Real-life Audio / Baseline System',
                           params=params,
                           system_desc=params.get('description'),
                           system_parameter_set_id=params.get('active_set'),
                           setup_label='Development setup',
                           log_system_progress=params.get_path('general.log_system_progress'),
                           show_progress_in_console=params.get_path('general.print_system_progress'),
                           )

        # Show parameter set list and exit
        if args.show_set_list:
            params_ = ParameterContainer(
                project_base=os.path.dirname(os.path.realpath(__file__))
            ).load(filename=default_parameters_filename)

            if args.parameter_override:
                # Override parameters from a file
                params_.override(override=args.parameter_override)
            if 'sets' in params_:
                app.show_parameter_set_list(set_list=params_['sets'])

            return

        # Show dataset list and exit
        if args.show_dataset_list:
            app.show_dataset_list()
            return

        # Show system parameters
        if params.get_path('general.log_system_parameters') or args.show_parameters:
            app.show_parameters()

        # Initialize application
        # ==================================================
        #if params['flow']['initialize']:
        #    app.initialize()

        # Extract features for all audio files in the dataset
        # ==================================================
        #if params['flow']['extract_features']:            
        #    app.feature_extraction()

        # Prepare feature normalizers
        # ==================================================
        #if params['flow']['feature_normalizer']:
        #    app.feature_normalization()

        # System training
        # ==================================================
        #if params['flow']['train_system']:
        #    app.system_training()

        # System evaluation in development mode        
        if not args.mode or args.mode == 'dev':

            # System testing
            # ==================================================
            if params['flow']['test_system']:
                app.system_testing()

            # System evaluation
            # ==================================================
            if params['flow']['evaluate_system']:
                app.system_evaluation()

        # System evaluation with challenge data
        elif args.mode == 'challenge':
            # Set dataset to testing set for challenge
            params['dataset']['method'] = 'challenge_test'

            # Process dataset again, move correct parameters from dataset_parameters
            params.process_method_parameters('dataset')

            if params['general']['challenge_submission_mode']:
                # If in submission mode, save results in separate folder for easier access
                params['path']['recognizer'] = params.get_path('path.recognizer_challenge_output')

            challenge_app = Task3AppCore(name='DCASE 2017::Sound Event Detection in Real-life Audio / Baseline System',
                                         params=params,
                                         system_desc=params.get('description'),
                                         system_parameter_set_id=params.get('active_set'),
                                         setup_label='Evaluation setup'
                                         )
            # Initialize application
            #if params['flow']['initialize']:
            #    challenge_app.initialize()

            # Extract features for all audio files in the dataset
            #if params['flow']['extract_features']:
            #   challenge_app.feature_extraction()

            # System testing
            if params['flow']['test_system']:
                if params['general']['challenge_submission_mode']:
                    params['general']['overwrite'] = True

                challenge_app.system_testing()

                if params['general']['challenge_submission_mode']:
                    challenge_app.ui.line(" ")
                    challenge_app.ui.line("Results for the challenge data are stored at ["+params['path']['recognizer_challenge_output']+"]")
                    challenge_app.ui.line(" ")

            # System evaluation
            if params['flow']['evaluate_system']:
                challenge_app.system_evaluation()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
