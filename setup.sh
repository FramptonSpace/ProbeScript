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

# Download the Python script from GitHub
wget https://raw.githubusercontent.com/FramptonSpace/ProbeModelling/main/AWS%20Files/Moo_All_In_One.py?token=GHSAT0AAAAAACVCSBMSUWYC6QJG3HX73YAKZU3MCLQ -O /home/ec2-user/your-python-script.py

# Run the Python script
python /home/ec2-user/your-python-script.py