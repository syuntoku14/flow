#!/bin/bash
echo "Installing system dependencies for SUMO"
apt-get update
apt-get install -y cmake swig libgtest-dev python-pygame python-scipy
apt-get install -y autoconf libtool pkg-config libgdal-dev libxerces-c-dev
apt-get install -y libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
apt-get install -y build-essential curl unzip flex bison python python-dev
apt-get install -y python3-dev
pip3 install cmake cython

echo "Installing sumo binaries"
mkdir -p $HOME/sumo_binaries/bin
pushd $HOME/sumo_binaries/bin
wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/binaries-ubuntu1604.tar.xz
tar -xf binaries-ubuntu1604.tar.xz
rm binaries-ubuntu1604.tar.xz
chmod +x *
popd
echo 'export PATH=$PATH:$HOME/sumo_binaries/bin' >> ~/.bashrc
echo 'export SUMO_HOME=$HOME/sumo_binaries/bin' >> ~/.bashrc
