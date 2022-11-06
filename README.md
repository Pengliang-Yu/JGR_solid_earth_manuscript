# Instruction of Whakaari(used for JGR-solid_earth manuscript feature extraction) and plots generation
This Whakaari package initially implements a time series feature engineering and classification workflow that issues eruption alerts based on real-time tremor data. Here, we used the Whakaari package to do the feature extractions for all wells in the Rotokawa and Husmuli fields, (1)-(3) is one example on how to do the feature extraction for the well injection data. environment.yml contains all python packages used in this study.

## Feature extraction process

(1)the __init__.py file in the folder of whakaari is slightly modified to extract the features of Rotokawa and husmuli wells based on the __init__.py file in the folder of https://github.com/ddempsey/whakaari/tree/master/whakaari, main changes on the time intervals to make it match with the injection well data.

(2)the tremor_data_husmuli_hour_unit_norm2015_CO2_col_name.dat file in the folder of data is the processed normalized humuli wells injection data, which could be directly used for feature extraction, but need to change the file name to 'tremor_data.dat' first. For all other raw well data, all of them need to be processed and normalzied first by converting them to the format as the file of tremor_data_husmuli_hour_unit_norm2015_CO2_col_name.dat

(3) FM_husmuli_norm_HN12_2015.py in the folder of feature_extraction_script is for the feature extraction of HN12 well. Similarly, we could extrac the features for all other wells of Rotokawa and Husmuli.

## Introduction of scripts in manuscript_scripts_figures folder

  The manuscript_scripts_figures folder includes all the scripts that used to generate the plots/tabled results in manuscript, the required data files to produce these plots include the extracted_features files for all the wells of Rotokawa and Husmuli fields, and the seismicity catalogs of Rotokawa and Husmuli fields; The Húsmúli reinjection data and raw earthquake catalog are provided by Reykjavík Energy and the Icelandic Meteorological Office (IMO), Iceland, and are freely available at DOI  https://doi.org/10.5281/zenodo.7041552. Rotokawa injection data are owned by the commercial field operator, Rotokawa Joint Venture. Requests for data access subject to privacy agreements should be directed to Mercury NZ Ltd. The raw earthquake data for Rotokawa geothermal field were obtained from Hopp et al., (2020) shared at DOI https://doi.org/10.17605/OSF.IO/C2M6U.
  
(1)Rotokawa_map_fig1.py is used to plot the fig.3c

(2)lb_vs_t_vs_q_total_remove_lb_kgs_fig2.py is used to plot fig.2

(3) Husmuli_map_0601_fig3.py is used to plot the fig.3c

(4)lb_vs_t_vs_q_total_husmuli_hour_JA_remove_LB_kgs_fig4.py is used to plot the fig.4

(5) EqvsQ_all_wells_part1_normalzed_non_negative_pvalue_lag_JA_1207_fig5.py and EqvsQ_all_wells_part2_normalzed_non_negative_pvalue_lag_JA_1207_fig5.py are used to plot fig.5

(6) EqvsQ_all_wells_part1_normalzed_non_negative_pvalue_lag_vsRMS_fig6.py is used to plot the fig.6

(7) EqvsQ_all_wells_husmuli_normalzed_non_negative_pvalue_3mdti_lag_newF_5wells_KDE_JA_Co2_1207_fig7.py is used to plot the fig.7

(8) sig_features_Qtotal_bigdti_undersample_newdata_median_4W_gridDis_RK20_downsample_fig9.py is used to plot the fig.9

(9) sig_features_Qtotal_bigdti_undersample_newdata_median_4W_fig10.py is used to plot the fig.10

(10) sig_features_Qtotal_bigdti_undersample_newdata_median_4W_WHP_fig11.py is used to plot fig.11

(11) sig_features_Qtotal_bigdti_undersample_newdata_median_4W_dqdt_fig12.py is used to plot fig.12

(12) sig_features_ALL_pvalue_undersample1_fast1_newdata_2subp_part1_norm_fig13.py and sig_features_ALL_pvalue_undersample1_fast1_newdata_2subp_part2_norm_fig13.py are used to plot fig.13

(13) sig_features_husmuli_undersample_Qtotal_decluster_2015_allWell_fig14.py is used to plot fig.14

(14) sig_features_husmuli_undersample_All_decluster2015_2subp_45JA_CO2_58_1027_fig15.py is used to plot fig.15

(15) sig_features_RK24_pvalue_undersample1_QvspvsQp_newdata_intqdt_norm1_JA_dqdt_fig16.py and sig_features_husmuli_pvalue_undersample_norm_QvsdiffQvsintQ_HN12_fig16.py is used to plot fig16


## Installation of Whakaari package

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

# Acknowledgements

We thank Manuel Rivera at Mercury for discussions that improved our understanding of the Rotokawa geothermal field. P Yu acknowledges Dr.Alberto Ardid for assistance producing Figs. 1 & 3. D. Dempsey acknowledges funding from MBIE Endeavour “Empowering Geothermal” research grant. The authors wish to acknowledge the Centre for eResearch at the University of Auckland for their help in facilitating this research on Research Virtual Machine. http://www.eresearch.auckland.ac.nz.  A. P. Rinaldi and V. A. Ritz are funded through the COSEISMIQ and DEEP projects. The DEEP project (http://deepgeothermal.org) is funded through the ERANET Cofund GEOTHERMICA (Project No. 200320-4001) from the European Commission. The DEEP project benefits from an exploration subsidy of the Swiss federal office of energy for the EGS geothermal project in Haute-Sorne, canton of Jura (contract number MF-021-GEO-ERK), which is gratefully acknowledged. The COSEISMIQ project (http://www.coseismiq.ethz.ch) is funded through the ERANET Cofund GEOTHERMICA (Project No. 731117) from the European Commission, and Geological Survey Ireland (GSI; Project No. 170167-44011).

