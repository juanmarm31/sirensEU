First steps
-----------

Important first steps are here: https://redmine.idmt.fraunhofer.de/issues/16976#change-69526

The ALF config file can be found: https://gitserv00.idmt.fraunhofer.de/m2d/smt/regional_siren_classification/-/blob/main/experiments_final/alf/resnet_siren.json?ref_type=heads
The patches from the spectrogram that are currently being fed into the ResNet model are 1.48 seconds long. Perhaps this is still too short.

If you're interested and currently have the capacity for it, you could train a few models with ALF. This way, we could systematically optimize some parameters. Specifically, this involves: (1) Modifying the ALF config file, (2) Starting the training, (3) Doing something else in the meantime, (4) Documenting the results, and (5) Repeating the process with step (1).

Current workspace and excecution ilmetad012, have a clone of the repositories https://gitserv00.idmt.fraunhofer.de/ima/alf/alf and https://gitserv00.idmt.fraunhofer.de/m2d/smt/regional_siren_classification under /mnt/IDMT-WORKSPACE/DATA-STORAGE/abr, and then execute the two lines in the alf package folder as indicated in https://gitserv00.idmt.fraunhofer.de/m2d/smt/regional_siren_classification/-/blob/main/experiments_final/alf/batch_training.sh?ref_type=heads.


# GENERAL 

## Conda env:

    conda env create -f /home/rodrjl/MDS/17063/ilmetad012/alf/env/ALFpaka.yml

    pip install -r /home/rodrjl/MDS/17063/ilmetad012/alf/requirements.txt

    pip install -r /home/rodrjl/MDS/17063/ilmetad012/alf/requirements_extras.txt

    pip install -e /home/rodrjl/MDS/17063/ilmetad012/alf           

## Update libraries ALFpaka:

    pip install --upgrade albumentations

    pip install --upgrade typing-extensions

# NOTE:

    conda uninstall llvmlite

    pip install --force-reinstall llvmlite

    pip uninstall llvmlite

    pip install --upgrade pip setuptools

    conda install -c conda-forge llvmlite


# main .sh:

    cd /home/rodrjl/MDS/17063/ilmetad012/regional_siren_classification/experiments_final/alf

    chmod +x batch_training.sh

    ./batch_training.sh


Create a .docx file that has the results of different configurations for parameters. 

Tests:
--------

1. Increase the value of blocksize from 1.4 to around 2.5 and for that -> do again Training + Evaluation
2. both settings should have pcen_True. There are 4 Configurations:
-- PCEN 2 variations
-- Blocksize 2 variations


# First variation @ ilmetad012:
    "fftsize": 2048,
    "hopsize": 1024,
    "sample_rate": 44100,
    "patchsize": 64,
    "patchhop": 32,
    "log_spec": true,
    "pcen": false,

# main steps please follow:

    conda activate alfpaka_env

    cd /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/
    
    chmod +x batch_training_sr1.sh

    ./batch_training_sr1.sh

# Notes for this file:
## main code is: 
    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/training_main.py

## temp file:
    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp1

## config json file:
    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren1.json

## reults folder:
    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/results1_sr


# Second variation @ ilmetad012:
    "fftsize": 2048,
    "hopsize": 1024,
    "sample_rate": 44100,
    "patchsize": 64,
    "patchhop": 32,
    "log_spec": true,
    "pcen": true,

# main steps please follow:

    conda activate alfpaka_env

    cd /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/

    chmod +x batch_training_sr2.sh

    ./batch_training_sr2.sh

# Notes for this file:

## main code is: 

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/training_main.py

## temp file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp2

## config json file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren2.json

## reults folder:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/results2_sr


# Third variation @ ilmetad012:
    "fftsize": 2048,
    "hopsize": 1024,
    "sample_rate": 44100,
    "patchsize": 108,
    "patchhop": 32,
    "log_spec": true,
    "pcen": false,

# main steps please follow:

    conda activate alfpaka_env

    cd /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/

    chmod +x batch_training_sr3.sh

    ./batch_training_sr3.sh

# Notes for this file:

## main code is:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/training_main.py

## temp file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp3

## config json file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren3.json

## reults folder:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/results3_sr


# Fourth variation @ ilmetad012:
    "fftsize": 2048,
    "hopsize": 1024,
    "sample_rate": 44100,
    "patchsize": 108,
    "patchhop": 32,
    "log_spec": true,
    "pcen": true,

# main steps please follow:

    conda activate alfpaka_env

    cd /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/

    chmod +x batch_training_sr4.sh

    ./batch_training_sr4.sh

# Notes for this file:

## main code is:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/training_main.py

## temp file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp4

## config json file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren4.json

## reults folder:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/results4_sr

Taking as best result from previous variations the last one, fourth variation, now the model parameters related to the first conv is reduced each time to the middle starting with 32 as first value.

# First Experiment @ ilmeatad012 ~/mnt/../rodrjl


main code is:
./mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/batch_training_sr4_filters_2.sh


temp file:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp4_2

config json file:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren4_2.json

reults folder:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/results4_sr_2


# Second Experiment @ ilmeatad012 ~/mnt/../rodrjl


main code is:
./mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/batch_training_sr4_filters_16.sh


temp file:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp4_16

config json file:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren4_16.json

reults folder:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/results4_sr_16


# Third Experiment @ ilmeatad012 ~/mnt/../rodrjl


main code is:
./mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/batch_training_sr4_filters_8.sh


temp file:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp4_8

config json file:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren4_8.json

reults folder:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/results4_sr_8


# Fourth Experiment @ ilmeatad012 ~/mnt/../rodrjl


main code is:
./mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/batch_training_sr4_filters_4.sh


temp file:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp4_4

config json file:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren4_4.json

reults folder:
/mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/results4_sr_4


Table with PCEN experiments in the first sheet and in the second sheet there are experiments related to the different parameters factor (2, 4, 8, 16)
LINK: https://fraunhofer-my.sharepoint.com/:x:/g/personal/juan_manuel_rodriguez_mejia_idmt_fraunhofer_de/EQtmCbzFbphAu9jiUoSyC6kBI7AppTcUsjJutIpBl6t3EQ?e=fY4XLS


"Predictions" folder: 
* predictions_cv{cv_step}.npy: This file contains the predicted results (resultsTest) for the test set in the cross-validation step specified by cv_step. The data type of this file is typically a NumPy array with floating-point numbers.
* targets_cv{cv_step}.npy: This file contains the ground truth or target values (self.__training_data_model.cv_step_data_model.idMatTest) for the test set in the cross-validation step specified by cv_step. The data type of this file is likely a NumPy array with integer or categorical values.
* file_frames_for_predictions_cv{cv_step}.npy: This file contains information about the frames per file in the test set for the cross-validation step specified by cv_step. The data type of this file is likely a NumPy array with integer values.
* file_names_for_prediction_cv{cv_step}.npy: This file contains the data file names associated with the test set for the cross-validation step specified by cv_step. The data type of this file is likely a NumPy array with string values.

In this case "cv_step" is 1.


Creation of a zip files with confusiuon matrixes based on the predictions folder



# Some other experiments:

# e_3_2 for 2.5 s @ ilmetad012:

## main train & test routes:

windows location of datasets: J:\Metadaten\AVdata\xtract\rodrjl\2024_01_siren_classification

    { "dir_train":"/home/avdata/xtract/rodrjl/2024_01_siren_classification/train",
      "dir_test":"/home/avdata/xtract/rodrjl/2024_01_siren_classification/test",
 
## Classifications due to countries:

      "classes": ["canada-ambulance",
                  "canada-firefighters",
                  "canada-police",
                  "china-ambulance",
                  "china-firefighters",
                  "china-police",
                  "france-ambulance",
                  "france-firefighters",
                  "france-police",
                  "germany-ambulance",
                  "germany-firefighters",
                  "germany-police",
                  "india-ambulance",
                  "india-firefighters",
                  "india-police",
                  "italy-ambulance",
                  "italy-firefighters",
                  "italy-police",
                  "japan-ambulance",
                  "japan-firefighters",
                  "japan-police",
                  "spain-ambulance",
                  "spain-firefighters",
                  "spain-police",
                  "usa-ambulance",
                  "usa-firefighters",
                  "usa-police"],

## Configuration with time 2.5s and PCEN as True:

    "fftsize": 2048,
    "hopsize": 1024,
    "sample_rate": 44100,
    "patchsize": 64,
    "patchhop": 32,
    "log_spec": true,
    "pcen": true,

# main steps, please follow:

    screen -S "name"
    conda activate alfpaka_env
    cd /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/
    chmod +x batch_training_sr3_2.sh
    ./batch_training_sr3_2.sh

# Notes for this file:

## main code is:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/training_main.py

## temp file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp3_2

## config json file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren3_2.json

## results folder @abr:

    /home/avdata/xtract/rodrjl/sirens_classification_experiments_3/3-2_5s

# e_3_1 for 1.48 s @ ilmetad012: 

## main train & test routes:

windows location of datasets: J:\Metadaten\AVdata\xtract\rodrjl\2024_01_siren_classification

    { "dir_train":"/home/avdata/xtract/rodrjl/2024_01_siren_classification/train",
      "dir_test":"/home/avdata/xtract/rodrjl/2024_01_siren_classification/test",
 
## Classifications due to countries:

      "classes": ["canada-ambulance",
                  "canada-firefighters",
                  "canada-police",
                  "china-ambulance",
                  "china-firefighters",
                  "china-police",
                  "france-ambulance",
                  "france-firefighters",
                  "france-police",
                  "germany-ambulance",
                  "germany-firefighters",
                  "germany-police",
                  "india-ambulance",
                  "india-firefighters",
                  "india-police",
                  "italy-ambulance",
                  "italy-firefighters",
                  "italy-police",
                  "japan-ambulance",
                  "japan-firefighters",
                  "japan-police",
                  "spain-ambulance",
                  "spain-firefighters",
                  "spain-police",
                  "usa-ambulance",
                  "usa-firefighters",
                  "usa-police"],

## Configuration with time 1.48s and PCEN as True:

    "fftsize": 2048,
    "hopsize": 1024,
    "sample_rate": 44100,
    "patchsize": 108,
    "patchhop": 32,
    "log_spec": true,
    "pcen": true,

# main steps, please follow:

    screen -S "name"
    conda activate alfpaka_env
    cd /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/
    chmod +x batch_training_sr3_1.sh
    ./batch_training_sr3_1.sh

# Notes for this file:

## main code is:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/alf/training_main.py

## temp file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/temp3_1

## config json file:

    /mnt/IDMT-WORKSPACE/DATA-STORE/rodrjl/regional_siren_classification/experiments_final/alf/combinations_ilmetad012/resnet_siren3_1.json

## results folder @abr:

    /home/avdata/xtract/rodrjl/sirens_classification_experiments_3/3-1_48s






