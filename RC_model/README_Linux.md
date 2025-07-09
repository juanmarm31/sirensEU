Siren-classification_EU - SiCaPKF
===================================

This project is a part of VeraAI project (2023) and IdeeMT (2023) and further approaches of 2024

Tasks / Project plan
-------------
- Creation of dataset to use as test and train for the selected model.
- Using of ResNet supervised NN model to analyse EU Countries excerpts.
- Using of one/two other models to analyse EU Countries excerpts.

Datasets:
---------

Location of currently train and test datasets:

    ~/Dataset/train
    ~/Dataset/test

The goals of this projet is to provide a simple, unified, command line interface for using pretrained models that allows classifying sirens.

This is heavily work in progress, currently we are supporting:

 - 🧠 ResNet approaches
 - 🧠 ALSPaka

Structure
--------

Project-structure/
├── README.md
├── requirements.txt
├── scripts/
    ├── dataset0.py
    ├── SICaqPKF.py
    └── main0.py
└── Dataset/
    ├── Test
        ├── france
            ├── ambulance
            ├── firefighter
            └── police
        ├── germany
            ├── ambulance
            ├── firefighter
            └── police
        ├── italy
            ├── ambulance
            ├── firefighter
            └── police
        └── spain
            ├── ambulance
            ├── firefighter
            └── police
    └── train
        ├── france
            ├── ambulance
            ├── firefighter
            └── police
        ├── germany
            ├── ambulance
            ├── firefighter
            └── police
        ├── italy
            ├── ambulance
            ├── firefighter
            └── police
        └── spain
            ├── ambulance
            ├── firefighter
            └── police


Installation
------------

Note: Installing this can be sometimes a bit tricky. Try to follow the steps below:

Create a conda environment + install required packages. The python version can be the last version but recommended is python version 3.11.8

    conda create -n SiClaPAF python=3.11.8
    conda activate SiClaPAF
    pip install -r requirements.txt

    git clone https://github.com/cknd/pyESN.git


Install setup

    pip install -e .

Some keras history

Using Linux:

    TF_ENABLE_ONEDNN_OPTS='0' python
    import os
    print(os.getenv("TF_ENABLE_ONEDNN_OPTS"))
    quit()

Using windows/bash on windows
    export TF_ENABLE_ONEDNN_OPTS="KEY"
    python

To verify this, use:

    echo $TF_ENABLE_ONEDNN_OPTS

First steps
-----------

Download the dataset available on https://gitlab.tu-ilmenau.de/juro7695/Research_Seminar-SirenClassificationEU.git

You can try following command to start with the proyect

    python scripts/SiClaPAF --train_dir J:/Metadaten/AVdata/xtract/rodrjl/DATABASE/train-DW-EU --test_dir J:/Metadaten/AVdata/xtract/rodrjl/DATABASE/test-RSCD-EU --batch_size 2 --epochs 30 --output_model results-ESKmodel

When working with windows

    python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings.zip" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset.zip" --batch_size 2 --epochs 30 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel1"

Now separating two sets of classes: 
1. Set of coutnries: France, Germany, Spain and Italy
2. Set of Sirentypes: Ambulance, Police and Firefighters

    python scripts/SiCaPKF2 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c.zip" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c.zip" --batch_size 2 --epochs 30 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel2"

    python scripts/SiCaPKF --train_dir "../Datasets/train/DW_Sirens_RealRecordings.zip" --test_dir "../Datasets/Test/regional_siren_classification_dataset.zip" --batch_size 2 --epochs 30 --output_model "../results-ESKmodel"


    python scripts/SiCaPKF0 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c" --batch_size 2 --epochs 35 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel0_44"

    python scripts/SiCaPKF0 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c_1" --batch_size 2 --epochs 35 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec"


Other option, is in developing

    python scripts/SiCaPKF0 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/train/" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/test/" --batch_size 2 --epochs 1 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_S000"

Separation into countries displaying confusion matrix of tru labelos vs predictions of siren types (still developing)

    python scripts/SiCaPKF1 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/train/" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/test/" --batch_size 2 --epochs 1 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_SICaPKF1"

COnfusion matrix for countries 

every class in average:

    python scripts/SiCaPKF2 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/train/" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/test/" --batch_size 2 --epochs 35 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_TOTAL"

ambulance

    python scripts/SiCaPKF2 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/traina/" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/testa/" --batch_size 2 --epochs 35 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_ambulance"

FIrefighter

    python scripts/SiCaPKF2 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/trainf/" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/testf/" --batch_size 2 --epochs 35 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_firefighter"

Police

    python scripts/SiCaPKF2 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/trainp/" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/testp/" --batch_size 2 --epochs 35 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_police"


version 3:

    python scripts/SiCaPKF3 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/train/" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/test/" --batch_size 2 --epochs 2 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_TOTAL_3"


    python scripts/SiCaPKF1 --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/train/" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/est/" --batch_size 2 --epochs 2 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_TOTAL_3"


/c/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset.7z

/c/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings.7z



Common errors when configuring tensorflow

2024-06-05 15:46:04.724600: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the 
environment variable `TF_ENABLE_ONEDNN_OPTS=0`.


python scripts/SiCaPKF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings.zip" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset.zip" --batch_size 2 --epochs 30 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel1000"


python scripts/SiCaPKF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/train/" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Dataset/test/" --batch_size 2 --epochs 1 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_SICaPKF1"


python scripts/SiCaPKF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c_1" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_all"



General

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c_1" --batch_size 2 --epochs 30 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification
_EU/results-ESKmodel_test_5_sec_120_epochs_all"

All:

    python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c1" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c1" --batch_size 2 --epochs 30 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_all_final"

    python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c1" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c1" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_all_final_balance"

France:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c_france/france" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c_1_france/france" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_france"

Germany:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c_germany/germany" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c_1_germany/germany" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_germany"

Italy:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c_italy/italy" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c_1_italy/italy" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_italy"

Spain:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train/DW_Sirens_RealRecordings_st_c_spain/spain" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/Test/regional_siren_classification_dataset_st_c_1_spain/spain" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_spain"


BALANCE:

All:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train-DW-EU_balance" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/test-RSCD-EU_balance" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Results-ESKmodel_test_5_sec_120_epochs_all_final_balance_2"

France:

python scripts/SICaPKF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train-country+sirentype_balance/france" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/test-country+sirentype_balance/france" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_france_balance_2"

Germany:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train-country+sirentype_balance/germany" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/test-country+sirentype_balance/germany" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_germany_balance_2"

Italy:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train-country+sirentype_balance/italy" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/test-country+sirentype_balance/italy" --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_italy_balance_2"

Spain:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/train-country+sirentype_balance/spain" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/Datasets/test-country+sirentype_balance/spain" --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/SoSe2024/Sirens-clasification_EU/results-ESKmodel_test_5_sec_120_epochs_spain_balance_2"






BALANCE - mixup


All:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class-aug_all" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/train-balance-country-class-aug_all" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/FP-Sirens/Sirens-clasification_EU/Results-SiClaPAFmodel_test_5_sec_120_epochs_all_final_balance_data_augmentation_all" 

France:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class-aug_mixup/france" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/train-balance-country-class-aug_mixup/france" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/results-SiClaPAFmodel_test_5_sec_120_epochs_france_final_balance_data_augmentation_mixup" 

Germany:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class-aug_mixup/germany" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class-aug_mixup/germany" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/results-SiClaPAFmodel_test_5_sec_120_epochs_germany_final_balance_data_augmentation_mixup" 

Italy:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class-aug_mixup/italy" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class-aug_mixup/italy" --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/results-SiClaPAFmodel_test_5_sec_120_epochs_italy_final_balance_data_augmentation_mixup" 

Spain:

python scripts/SiClaPAF --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class-aug_mixup/spain" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class-aug_mixup/spain" --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/results-SiClaPAFmodel_test_5_sec_120_epochs_spain_final_balance_data_augmentation_mixup" 



BALANCE - after augmentation

All 12:

time python /home/juro7695/Documents/Forschungsproject/FP-Sirens/RC_model/scripts/RC --train_dir "/home/juro7695/Documents/Forschungsproject/DATASET/train-balance-country-class_augmented_12_upgraded" --test_dir "/home/juro7695/Documents/Forschungsproject/DATASET/test-balance-country-class_augmented_12_upgraded" --batch_size 2 --epochs 120 --output_model "/home/juro7695/Documents/Forschungsproject/Results-Linux/2025/1/Results-RCmodel_test_5_sec_120_epochs_all_final_balance_data_augmentation_12" 

All:

time python /home/juro7695/Documents/Forschungsproject/FP-Sirens/RC_model/scripts/RC --train_dir "/home/juro7695/Documents/Forschungsproject/DATASET/train-balance-country-class_augmented_flat_upgraded" --test_dir "/home/juro7695/Documents/Forschungsproject/DATASET/test-balance-country-class_augmented_flat_upgraded" --batch_size 2 --epochs 120 --output_model "/home/juro7695/Documents/Forschungsproject/Results-Linux/2025/1/Results-RCmodel_test_5_sec_120_epochs_all_final_balance_data_augmentation_flat" 
real    1006m28.202s
user    0m9.499s
sys     0m22.341s

France:

time python /home/juro7695/Documents/Forschungsproject/FP-Sirens/RC_model/scripts/RC --train_dir "/home/juro7695/Documents/Forschungsproject/DATASET/train-balance-country-class_augmented_all_upgraded/france" --test_dir /home/juro7695/Documents/Forschungsproject/DATASET/test-balance-country-class_augmented_all_upgraded/france" --batch_size 2 --epochs 120 --output_model "/home/juro7695/Documents/Forschungsproject/Results-Linux/2025/1/results-RCmodel_test_5_sec_120_epochs_france_final_balance_data_augmentation" 

real    287m59.167s
user    0m2.685s
sys     0m7.014s

Germany:

time python /home/juro7695/Documents/Forschungsproject/FP-Sirens/RC_model/scripts/RC --train_dir "/home/juro7695/Documents/Forschungsproject/DATASET/train-balance-country-class_augmented_all_upgraded/germany" --test_dir "/home/juro7695/Documents/Forschungsproject/DATASET/test-balance-country-class_augmented_all_upgraded/germany" --batch_size 2 --epochs 120 --output_model /home/juro7695/Documents/Forschungsproject/Results-Linux/2025/1/results-RCmodel_test_5_sec_120_epochs_germany_final_balance_data_augmentation" 

Italy:

time python /home/juro7695/Documents/Forschungsproject/FP-Sirens/RC_model/scripts/RC --train_dir "/home/juro7695/Documents/Forschungsproject/DATASET/train-balance-country-class_augmented_all_upgraded/italy" --test_dir "/home/juro7695/Documents/Forschungsproject/DATASET/test-balance-country-class_augmented_all_upgraded/italy" --batch_size 2 --epochs 120 --output_model "/home/juro7695/Documents/Forschungsproject/Results-Linux/2025/1/results-RCmodel_test_5_sec_120_epochs_italy_final_balance_data_augmentation" 

Spain:

time python /home/juro7695/Documents/Forschungsproject/FP-Sirens/RC_model/scripts/RC --train_dir "/home/juro7695/Documents/Forschungsproject/DATASET/train-balance-country-class_augmented_all_upgraded/spain" --test_dir "/home/juro7695/Documents/Forschungsproject/DATASET/test-balance-country-class_augmented_all_upgraded/spain" --batch_size 2 --epochs 120 --output_model "/home/juro7695/Documents/Forschungsproject/Results-Linux/2025/1/results-RCmodel_test_5_sec_120_epochs_spain_final_balance_data_augmentation" 