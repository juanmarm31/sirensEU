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



BALANCE - after augmentation


All 12:

python RC --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/train-balance-country-class_augmented_12_upgraded" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class_augmented_12_upgraded" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/2025/3/Results-RCmodel_test_5_sec_120_epochs_all_final_balance_data_augmentation_12" 

All:

time python RC --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/train-balance-country-class_augmented_flat_upgraded" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class_augmented_flat_upgraded" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/2025/3/Results-RCmodel_test_5_sec_120_epochs_all_final_balance_data_augmentation_flat" 
real    326m30.012s
user    0m3.873s
sys     0m8.108s

France:

python RC --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/train-balance-country-class_augmented_all_upgraded/france" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class_augmented_all_upgraded/france" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/2025/3/results-RCmodel_test_5_sec_120_epochs_france_final_balance_data_augmentation" 

Germany:

time python RC --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/train-balance-country-class_augmented_all_upgraded/germany" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class_augmented_all_upgraded/germany" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/2025/3-1/results-RCmodel_test_5_sec_120_epochs_germany_final_balance_data_augmentation" 
real    78m14.256s
user    0m1.296s
sys     0m2.513s

Italy:

python RC --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/train-balance-country-class_augmented_all_upgraded/italy" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class_augmented_all_upgraded/italy" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/2025/3/results-RCmodel_test_5_sec_120_epochs_italy_final_balance_data_augmentation" 

Spain:

python RC --train_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/train-balance-country-class_augmented_all_upgraded/spain" --test_dir "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/DATASET/test-balance-country-class_augmented_all_upgraded/spain" --batch_size 2 --epochs 120 --output_model "C:/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/Results/2025/3/results-RCmodel_test_5_sec_120_epochs_spain_final_balance_data_augmentation" 


## Common errors when configuring tensorflow

2024-06-05 15:46:04.724600: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the 
environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
