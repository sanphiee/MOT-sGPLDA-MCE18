# MOT-sGPLDA-MCE18

Multiobjective Optimization Training of PLDA for Speaker Verification \
environment : anaconda3, python3, require sklearn.

1. prepare data, make directory ./data and ./temp \
put MCE18 offical uncompressed data on "./data/", \
there are "bl_matching.csv, trn_blacklist.csv, trn_background.csv, dev_blacklist.csv, dev_background.csv, dev_bl_id.ndx, tst_evaluation.csv, tst_evaluation_keys.csv" \
PS: open the website (http://www.mce2018.org/) for data requirement.


2. system: length-normalization + LDA + PLDA (MotPLDA) + score-normalization. 

A. run ./python/mce18_plda_preprocess.py \
It will generate "./temp/mce18.mat"

Option 1: \
B1. run ./matlab/gplda_demo.m for PLDA \
The script will read "./temp/mce18.mat", and it will generate "./temp/mce18_result.mat"

C1. run ./python/mce18_plda_eval.py \
The script will read "./temp/mce18.mat", and the results are \
Test set score using training and development set : \
Top S detector EER is 6.75% \
Top 1 detector EER is 9.39% (Total confusion error is 270)


Option 2: \
B2. run ./matlab/moplda_demo.m for MoPLDA \
The script will read "./temp/mce18.mat", and it will also generate "./temp/mce18_result.mat"

C2. run ./python/mce18_plda_eval.py \
The script will read "./temp/mce18.mat", and the results are \
Test set score using training and development set : \
Top S detector EER is 5.41% \
Top 1 detector EER is 7.32% (Total confusion error is 204)


Reference:
[1] Suwon Shon, Najim Dehak, Douglas Reynolds, and James Glass, “Mce 2018: The 1st multi-target speaker detection and identification challenge evaluation (mce) plan, dataset and baseline system,” in ArXiv e-prints arXiv:1807.06663, 2018.

[2] L. He, X. Chen, C. Xu, and J. Liu, “Multiobjective Optimization Training of PLDA for Speaker Verification,”
ArXiv e-prints arXiv:1808.08344, Aug. 2018.

[3] L. He, X. Chen, C. Xu, and J. Liu, “Multiobjective Optimization Training of PLDA for Speaker Verification,” submitted to ICASSP 2019.

He Liang, heliang@mail.tsinghua.edu.cn \
Oct. 30, 2018

