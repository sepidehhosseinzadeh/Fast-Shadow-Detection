function desc  = calcTextonHistBig(rgb_im, seg, numRegion)
    load('textonCluster.mat');
    numWord = 200;
    [desc grid_x grid_y ] = calcTextonHist(rgb_im, 16, 2);
    word = assignWords(desc, wc);
    desc = assignRegion(word , numWord, ...
        grid_x(:), grid_y(:), seg, numRegion );
end