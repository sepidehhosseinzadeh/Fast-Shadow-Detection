function dist = distHistInt(x, c)
    [ndata, dimx] = size(x);
    [ncentres, dimc] = size(c);
    dist = zeros(ndata,ncentres);
    for p = 1:ndata
        nonzero_ind = find(x(p,:)>0);
        tmp_x = repmat(x(p,nonzero_ind), [ncentres 1]);
        dist(p,:) = sum(min(tmp_x,c(:,nonzero_ind)),2)';
    end
end