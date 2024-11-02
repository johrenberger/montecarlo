#!/bin/bash

# Update package lists
sudo apt-get update
sudo apt-get upgrade -y



# install pip
sudo apt-get install -y python3-pip

#install poetry
pip3 install poetry
pip3 install scipy
pip3 install numpy
pip3 install matplotlib
pip3 install pandas

#upgrade pip
pip3 install --upgrade pip

#update poetry
pip3 install --upgrade poetry

#auto completion of poetry
poetry completions bash >> ~/.bash_completion