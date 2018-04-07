# Multi-task based joint learning acoustic event detection 

1. Here, we provided the DNN based joint learning training scripts as an example, two trained acoustic models (DNN-S & CNN-S) and the evaluation metrics/scripts. The metrics from the DCASE Challenge 2017 Task 3 were adopted to evaluate the performance of the acoustic event detection system. 

2. Requirements:
numpy>=1.9.2
scipy>=0.19.0
scikit-learn>=0.18.1
h5py>=2.6.0
matplotlib>=2.0.0
librosa>=0.5.0
keras>=2.0.8
theano>=0.9.0
six>=1.10.0
sed_eval>=0.1.8
soundfile>=0.9.0
coloredlogs>=5.2
tqdm >=4.11.2
pyyaml>=3.11
msgpack-python>=0.4.8
pydot-ng >= 1.0.0
pafy>=0.5.3.1
pandas>=0.19.2
recurrentshop
seq2seq

3. Instructions: 
a. Download and develop the DCASE Challenge 2017 Task 3 baseline system, which can be referred in:
https://github.com/TUT-ARG/DCASE2017-baseline-system

b. Replace the related scripts and the parameter files with the new scripts and the parameters provided here. 

c.  Switch on the related functiones.{line 2119 (DNN-S) or line 2121 (CNN-S) in "learners.py".}

d.  Run "python task3.py -m challenge" and get the detection results.


4. Final detection results for the system with DNN-S are as following:

Overall metrics:
===============

   Event label       | Seg. F1 | Seg. ER | Evt. F1 | Evt. ER |
   ----------------- + ------- + ------- + ------- + ------- +
   street            | 41.29   | 0.81    | 8.74    | 2.21    |
   ----------------- + ------- + ------- + ------- + ------- +
   Average           | 41.29   | 0.81    | 8.74    | 2.21    |



5. Final detection results for the system with CNN-S are as following:

 Overall metrics
 ===============
   Event label       | Seg. F1 | Seg. ER | Evt. F1 | Evt. ER |
   ----------------- + ------- + ------- + ------- + ------- +
   street            | 43.38   | 0.77    | 9.19    | 2.31    |
   ----------------- + ------- + ------- + ------- + ------- +
   Average           | 43.38   | 0.77    | 9.19    | 2.31    |

6. Train the joint learning acoustic model
If you want to train your own multi-task acoustic models jointly, the DNN based training script and also the training data are provided here as an example.
Run "python DNN_JointLearning.py" to train your own joint acoustic models and the training data can be downloaded from the google drive below: 
https://drive.google.com/open?id=10XcbKr8olysb1FA6UDv9QKyJZFYraUjQ
