addpath(genpath('../../GeometricContext/src'))
load('../cache/traintestfnlist.mat');
classifiers = load('../../GeometricContext/data/ijcvClassifier.mat'); 

% for training image
for i = numel(trainfnlist) : numel(trainfnlist)
    fprintf('processing %d of %d training\n', i , numel(trainfnlist));
    filename = trainfnlist{i};
    %filename = 'DSC01528';
    im = im2double(imread(['../Tappen/img/' filename '.jpg']));
    load(['../annt/annt_' filename '.mat'], 'seg');
    imseg = struct('imname','','imsize',[0 0]);
    imseg.imname = [filename '.jpg'];
    imseg.imsize = size(seg);
    imseg.segimage = seg;
    imseg.nseg = length(unique(seg));
    imseg = APPgetSpStats(imseg);
    try
        [pg, data, imsegs] = ijcvTestImage(im, imseg, classifiers);
        save(['../Tappen/geo/' filename '.mat'], 'pg', 'data', 'imsegs');
    catch exception
        fprintf('Error with file %s\n', filename);
   end
end

% for testing image
for i = 1 : numel(testfnlist)
    fprintf('processing %d of %d testing\n', i , numel(testfnlist));
    filename = testfnlist{i};
    im = im2double(imread(['../Tappen/img/' filename '.jpg']));
    load(['../Tappen/img/' filename '_seg.mat'], 'seg');    
    imseg = struct('imname','','imsize',[0 0]);
    imseg.imname = [filename '.jpg'];
    imseg.imsize = size(seg);
    imseg.segimage = seg;
    imseg.nseg = length(unique(seg));
    imseg = APPgetSpStats(imseg);
    try
        [pg, data, imsegs] = ijcvTestImage(im, imseg, classifiers);
        save(['../Tappen/geo/' filename '.mat'], 'pg', 'data', 'imsegs');
    catch exception
        fprintf('Error with file %s\n', filename);
    end
end
