function desc  = calcDSiftHist(rgb_im, seg, numRegion)
    load('dsiftCluster.mat');
    numWord = 200;
    [desc grid_x grid_y ] = dense_sift(rgb_im, 16, 2);
    word = assignWords(desc, wc);
    desc = assignRegion(word , numWord, ...
        grid_x(:), grid_y(:), seg, numRegion );
end