function [regions] = assignRegion(data, numWord, grid_x, grid_y, ...
                                    new_seg, numRegion)

    hist = zeros(1, numWord);
    ind = sub2ind(size(new_seg), grid_y, grid_x);
    map = new_seg(ind);
    regions = zeros(numRegion, numWord);
    for iRegion = 1:numRegion
        ind = find(map==iRegion);
        region_data = data(ind);
        for iWord = 1:numWord
            hist(iWord) = sum(region_data==iWord);
        end
        if (sum(hist)>=10)
            hist = hist ./ (sum(hist)+1e-5);
        else
            hist = zeros(1, numWord);
        end

        regions(iRegion, :) = hist;
    end
end
