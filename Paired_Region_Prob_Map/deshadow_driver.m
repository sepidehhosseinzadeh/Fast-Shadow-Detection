addpath('meanshift');
addpath('used_desc');
addpath('svmlight_mex');
addpath('utils');
addpath(genpath('UGM'));

unix('rm data/binary/* data/mask/* data/cache/* data/matting/* data/removal/* data/unary/*');
opt = {};
imagePath = '/home/sepideh/Documents/illuminChngeLrning/data/Lucia/';
images = dir([imagePath '/' '*.jpg']);

opt.dir = 'data/';

for i=1:length(images)
    opt.fn = images(i).name;
    opt.save = 1;
    opt.binaryClassifier = 'models/model_pair.mat';
    opt.unaryClassifier = 'models/unary_model.mat';
    opt.resize = 1;
    opt.adjecent = 0;
    opt.pairwiseonly = 0;
    opt.linearize = 0;
    h = findshadowcut_cross(opt);
end

