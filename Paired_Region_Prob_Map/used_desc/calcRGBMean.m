function [desc] = calcRGBMean(rgb_im, seg, numRegion)
 
    desc = zeros([numRegion size(rgb_im,3)]);    
    ind={};
    for iReg=1:numRegion
        ind{iReg} = seg(:)==iReg;
    end
    im = rgb_im;

    for ch=1:size(rgb_im,3)
        I = im(:,:,ch);
        for iReg=1:numRegion
            desc(iReg, ch) = mean(I(ind{iReg}));
        end
    end
end
