#!/bin/bash
wget -c https://drive.google.com/uc?id=1FMfpuNACCcT1CzhTixYvclLGSeGMYJyo -O icra_odom.zip
unzip icra_odom.zip
mkdir -p results/icra21
mv accuracy*.csv results/icra21
rm icra_odom.zip
