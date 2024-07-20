#!/bin/bash

# Update package lists
sudo yum update -y

# Install wget if not already installed
sudo yum install -y wget

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest.sh
bash Miniconda3-latest.sh -b -p $HOME/miniconda

# Initialize Miniconda
source $HOME/miniconda/bin/activate
conda init

# Add conda-forge channel
conda config --add channels conda-forge

# Create a conda environment with the required Python version and packages
conda create -y -n ProbeMod python=3.10.14
conda activate ProbeMod

# Install additional packages as needed
conda install -y numpy pandas scipy pymoo yaml 

# Ensure the script is executable
chmod +x setup.sh

git clone https://github.com/your-username/your-repo.git /home/ec2-user/your-repo