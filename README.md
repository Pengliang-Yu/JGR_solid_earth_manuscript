# Whakaari
This Whakaari package implements a time series feature engineering and classification workflow that issues eruption alerts based on real-time tremor data. This is the real-time version for running on a VM with html forecaster output and email alerting.

the __init__.py file in the folder of whakaari is slightly modified to extract the features of Rotokawa and husmuli wells based on the __init__.py file in the folder of https://github.com/ddempsey/whakaari/tree/master/whakaari, main changes on the time intervals to make it match with the injection well data.

the tremor_data.dat file in the folder of data is the processed normalized humuli wells injection data, which could be directly used for feature extraction, but need to change the file name to 'tremor_data.dat' 

## Installation

Ensure you have Anaconda Python 3.7 installed. Then

1. Clone the repo

```bash
git clone https://github.com/Pengliang-Yu/JGR_solid_earth_manuscript
```

2. CD into the repo and create a conda environment

```bash
cd whakaari

conda env create -f environment.yml

conda activate whakaari_env
```

The installation has been tested on Windows, Mac and Unix operating systems. Total install with Anaconda Python should be less than 10 minutes.


