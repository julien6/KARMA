#!/bin/bash

# Script d'installation pour Docker, kubectl, Kind, Helm et Miniconda
# Compatible avec Ubuntu/Debian

set -e  # Arrêter le script si une commande échoue

# Fonction pour afficher les étapes
function print_step {
    echo -e "\n\e[1;34m[STEP] $1\e[0m"
}

# Mettre à jour le système
print_step "Mise à jour du système"
sudo apt-get update && sudo apt-get upgrade -y

# 1. Installation de Docker
print_step "Installation de Docker"
if ! [ -x "$(command -v docker)" ]; then
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker $USER
    print_step "Docker installé avec succès"
else
    echo "Docker est déjà installé."
fi

# 2. Installation de kubectl
print_step "Installation de kubectl"
if ! [ -x "$(command -v kubectl)" ]; then
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/kubectl
    print_step "kubectl installé avec succès"
else
    echo "kubectl est déjà installé."
fi

# 3. Installation de Kind
print_step "Installation de Kind"
if ! [ -x "$(command -v kind)" ]; then
    curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
    print_step "Kind installé avec succès"
else
    echo "Kind est déjà installé."
fi

# 4. Installation de Helm
print_step "Installation de Helm"
if ! [ -x "$(command -v helm)" ]; then
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    print_step "Helm installé avec succès"
else
    echo "Helm est déjà installé."
fi

# 5. Installation de Miniconda
print_step "Installation de Miniconda"
if ! [ -d "$HOME/miniconda" ]; then
    curl -o Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda.sh
    ./Miniconda.sh -b -p $HOME/miniconda
    rm Miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    print_step "Miniconda installé avec succès"
else
    echo "Miniconda est déjà installé."
fi

# Charger Miniconda dans le script
print_step "Chargement de Miniconda"
source ~/miniconda/etc/profile.d/conda.sh

# 6. Création et activation de l'environnement Conda avec Python 3.8
print_step "Création de l'environnement Conda 'karma'"
if ! conda info --envs | grep -q "karma"; then
    conda create -n karma -y python=3.8
    print_step "Environnement 'karma' créé avec succès"
else
    echo "L'environnement 'karma' existe déjà."
fi

# Activer l'environnement
conda activate karma

# 7. Installation de MARLlib et patchs
print_step "Installation de MARLlib et application des patchs"
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install "pip==21.0"
pip install "gym>=0.20.0,<0.22.0"
pip install click
pip install gymnasium
pip install -r requirements.txt --timeout 216000
pip install ray
pip install -e .
python marllib/patch/add_patch.py -y
cd ..
print_step "MARLlib installé avec succès"

# 8. Installation des dépendances Python
print_step "Installation des dépendances Python"
if [ -f "requirements.txt" ]; then
    pip install protobuf==3.20.* # Pour éviter un bug lors de l'éxécution
    pip install setuptools==65.5.0 pip==21
    pip install wheel==0.38.0
    pip install -r requirements.txt --timeout 216000
    pip uninstall numpy ; pip install numpy==1.23.5
    print_step "Dépendances installées avec succès"
else
    echo "Warning: 'requirements.txt' introuvable. Aucune dépendance installée."
fi

# 9. Vérification des installations
print_step "Vérification des outils installés"
echo "Docker version : $(docker --version)"
echo "kubectl version : $(kubectl version --client --short)"
echo "Kind version : $(kind --version)"
echo "Helm version : $(helm version --short)"
echo "Conda version : $(conda --version)"

print_step "Redémarre ton terminal ou exécute 'newgrp docker' pour utiliser Docker sans sudo."

print_step "Installation terminée avec succès ! 🚀"
