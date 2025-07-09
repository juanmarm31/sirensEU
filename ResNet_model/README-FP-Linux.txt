First steps
-----------

Important first steps are here: https://redmine.idmt.fraunhofer.de/issues/16976#change-69526

The ALF config file can be found: https://gitserv00.idmt.fraunhofer.de/m2d/smt/regional_siren_classification/-/blob/main/experiments_final/alf/resnet_siren.json?ref_type=heads
The patches from the spectrogram that are currently being fed into the ResNet model are 1.48 seconds long. Perhaps this is still too short.

If you're interested and currently have the capacity for it, you could train a few models with ALF. This way, we could systematically optimize some parameters. Specifically, this involves: (1) Modifying the ALF config file, (2) Starting the training, (3) Doing something else in the meantime, (4) Documenting the results, and (5) Repeating the process with step (1).

Current workspace and excecution ilmetad012, have a clone of the repositories https://gitserv00.idmt.fraunhofer.de/ima/alf/alf and https://gitserv00.idmt.fraunhofer.de/m2d/smt/regional_siren_classification under /mnt/IDMT-WORKSPACE/DATA-STORAGE/abr, and then execute the two lines in the alf package folder as indicated in https://gitserv00.idmt.fraunhofer.de/m2d/smt/regional_siren_classification/-/blob/main/experiments_final/alf/batch_training.sh?ref_type=heads.


# GENERAL 

## Conda env:

    conda env create -f /home/juro7695/Documents/Forschungsproject/FP-Sirens/ResNet_model/alf/env/ALFpaka_env.yml

    pip install -r /home/juro7695/Documents/Forschungsproject/FP-Sirens/ResNet_model/alf/requirements.txt

    #pip install -r /home/juro7695/Documents/Forschungsproject/FP-Sirens/ResNet_model/alf/requirements_extras.txt

    pip install -e /home/juro7695/Documents/Forschungsproject/FP-Sirens/ResNet_model/alf           


# main .sh:

    cd /home/juro7695/Documents/Forschungsproject/FP-Sirens/ResNet_model/regional_siren_classification/experiments_FP-Linux


    #chmod +x batch_training.sh
    #./batch_training.sh

    chmod +x batch_training_EU_all_balance.sh
    ./batch_training_EU_all_balance.sh

    chmod +x batch_training_EU_f_balance.sh
    ./batch_training_EU_f_balance.sh

    chmod +x batch_training_EU_g_balance.sh
    ./batch_training_EU_g_balance.sh

    chmod +x batch_training_EU_i_balance.sh
    ./batch_training_EU_i_balance.sh

    chmod +x batch_training_EU_s_balance.sh
    ./batch_training_EU_s_balance.sh

