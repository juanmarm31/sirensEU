# Forschungsprojekt



## Conda env:

    conda env create -f "/c/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/FP-Sirens/ResNet_model/alf/env/ALFpaka_env.yml"

    pip install -r "/c/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/FP-Sirens/ResNet_model/alf/requirements.txt"

    #pip install -r "/c/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/FP-Sirens/ResNet_model/alf/requirements_extras.txt"

    pip install -e "/c/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/FP-Sirens/ResNet_model/alf"

## Main commands

    cd "/c/Users/ju31/OneDrive - Technische Universität Ilmenau/TU Ilmenau/WiSe2023-24/Forschungsprojekt/Projekt/FP-Sirens/ResNet_model/regional_siren_classification/experiments_FP"

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
