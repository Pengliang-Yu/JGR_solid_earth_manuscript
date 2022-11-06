# Whakaari
This Whakaari package implements a time series feature engineering and classification workflow that issues eruption alerts based on real-time tremor data. This is the real-time version for running on a VM with html forecaster output and email alerting.

## Installation

Ensure you have Anaconda Python 3.7 installed. Then

1. Clone the repo

```bash
git clone https://github.com/ddempsey/whakaari
```

2. CD into the repo and create a conda environment

```bash
cd whakaari

conda env create -f environment.yml

conda activate whakaari_env
```

The installation has been tested on Windows, Mac and Unix operating systems. Total install with Anaconda Python should be less than 10 minutes.


