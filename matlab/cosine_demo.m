
clc;
clear all;
close all;

load '../temp/sre14.mat';

COSINE_scores = enrol_ivec * test_ivec';
COSINE_scores_col = reshape(COSINE_scores',length(COSINE_scores(:)),1);

figure;
score_pos = find(test_mask < 2.5);
all_scores = COSINE_scores_col(score_pos);
all_key = test_key(score_pos);
[PLDA_eer,PLDA_dcf08,PLDA_dcf10,PLDA_dcf14] = compute_eer(all_scores,all_key,true);

figure;
prog_pos = find(test_mask==1);
prog_score = COSINE_scores_col(prog_pos);
prog_key = test_key(prog_pos);
[prog_PLDA_eer, prog_PLDA_dcf08, prog_PLDA_dcf10, prog_PLDA_dcf14] = compute_eer(prog_score,prog_key,true);

figure;
eval_pos = find(test_mask==2);
eval_score = COSINE_scores_col(eval_pos);
eval_key = test_key(eval_pos);
[eval_PLDA_eer, eval_PLDA_dcf08, eval_PLDA_dcf10, eval_PLDA_dcf14] = compute_eer(eval_score,eval_key,true);

fid=fopen('sre14_test.result','a');
fprintf(fid, '%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f\n', PLDA_eer,prog_PLDA_eer,eval_PLDA_eer,PLDA_dcf14,prog_PLDA_dcf14,eval_PLDA_dcf14);
fclose(fid);

% end