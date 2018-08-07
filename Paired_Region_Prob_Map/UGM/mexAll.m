fprintf('Compiling KPM files...\n');
mex -IKPM KPM/repmatC.c
mex -IKPM KPM/max_mult.c
fprintf('Compiling UGM decode files...\n');
mex -IUGM/mex UGM/mex/UGM_Decode_ExactC.c
mex -IUGM/mex UGM/mex/UGM_Decode_ICMC.c
fprintf('Compiling UGM infer files...\n');
mex -IUGM/mex UGM/mex/UGM_Infer_ExactC.c
mex -IUGM/mex UGM/mex/UGM_Infer_ChainC.c
mex -IUGM/mex UGM/mex/UGM_Infer_MFC.c
mex -IUGM/mex UGM/mex/UGM_Infer_LBPC.c
fprintf('Compiling UGM sample files...\n');
mex -IUGM/mex UGM/mex/UGM_Sample_GibbsC.c
fprintf('Compiling UGM estimation files...\n');
mex -IUGM/mex UGM/mex/UGM_makeNodePotentialsC.c
mex -IUGM/mex UGM/mex/UGM_makeEdgePotentialsC.c
mex -IUGM/mex UGM/mex/UGM_PseudoLossC.c
mex -IUGM/mex UGM/mex/UGM_MRFLoss_subC.c
mex -IUGM/mex UGM/mex/UGM_updateGradientC.c
mex -IUGM/mex UGM/mex/UGM_Loss_subC.c
