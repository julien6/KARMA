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

# 6. Vérification des installations
print_step "Vérification des outils installés"
echo "Docker version : $(docker --version)"
echo "kubectl version : $(kubectl version --client --short)"
echo "Kind version : $(kind --version)"
echo "Helm version : $(helm version --short)"
echo "Conda version : $(conda --version)"


source ~/miniconda3/etc/profile.d/conda.sh
conda create -n karma
conda activate karma
pip install -r requirements.txt


print_step "Redémarre ton terminal ou exécute 'newgrp docker' pour utiliser Docker sans sudo."

print_step "Installation terminée avec succès ! 🚀"
