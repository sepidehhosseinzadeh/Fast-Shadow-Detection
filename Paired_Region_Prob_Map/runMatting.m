addpath('matting');
addpath('utils');

if (~exist('thr_alpha','var'))
 thr_alpha=[];
end
if (~exist('epsilon','var'))
 epsilon=[];
end
if (~exist('win_size','var'))
 win_size=[];
end

if (~exist('levels_num','var'))
 levels_num=1;
end
if (~exist('active_levels_num','var'))
 active_levels_num=1;
end

for i =1:numel(testfnlist)  
    fprintf('processing %d out of %d\n', i, numel(testfnlist));   

    load(['data/cache/' testfnlist{i} '_finalpair.mat']);

    load(['data/cache/' testfnlist{i} '_everything.mat']);
    img = im;
    I=double(img)/255;

    load(['data/cache/' testfnlist{i} '_detect.mat']);
 
    hard_map=hardmap;

    hard_map_1=conv2(hard_map,fspecial('average',5),'same');
    hard_map_2=conv2(hard_map,fspecial('average',5),'same');

    consts_map=hard_map_2>1-1e-8 | hard_map_1==0;

    consts_vals=hard_map_1==0;

    alpha=solveAlpha(I,consts_map,consts_vals,epsilon,win_size);

    soft_map=alpha;

    ratio = cal_ratio(seg,pair,hard_map,soft_map,I);

    red=I(:,:,1);
    green=I(:,:,2);
    blue=I(:,:,3);

    red_val=ratio(1);
    green_val=ratio(2);
    blue_val=ratio(3);

    red_recover=(ones(size(soft_map))*(red_val+1))./(soft_map*red_val+1).*red;
    green_recover=(ones(size(soft_map))*(green_val+1))./(soft_map*green_val+1).*green;
    blue_recover=(ones(size(soft_map))*(blue_val+1))./(soft_map*blue_val+1).*blue;

    shadow_free=zeros(size(I));

    shadow_free(:,:,1)=red_recover;
    shadow_free(:,:,2)=green_recover;
    shadow_free(:,:,3)=blue_recover;

    shadow_free(shadow_free(:)>1) = 1;
    
    imwrite(soft_map, ['data/matting/'  testfnlist{i}  '_soft.jpg'], 'jpg');
    imwrite(shadow_free, ['data/removal/'  testfnlist{i}  '_recovery.jpg'], 'jpg');

end
