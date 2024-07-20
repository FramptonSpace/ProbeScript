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
conda install -y numpy pandas scipy pymoo pyaml git tmux

git clone https://github.com/FramptonSpace/ProbeScript.git "/home/ec2-user/AWS Files"

tmux new -s mysession

python "AWS Files/AWS Files/Moo_All_In_One.py"

git init

git remote add origin https://github.com/FramptonSpace/ProbeScript.git

cd "AWS Files"

cd "Moo_Outputs"

git add .

git commit -m "New Results"

instance_id=$(curl http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 stop-instances --instance-ids $instance_id --region eu-west-2
