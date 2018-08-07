function fn_list = readFnlist( fn_fnlist )
    fid = fopen(fn_fnlist, 'r');
    if (fid==0)
        return;
    end
    cnt = 0;
    fn_list = {};
    while (~feof(fid))
        cnt = cnt + 1;
        tmp = fscanf(fid, '%s', [1 1]);
        if ~strcmp(tmp, '')
            fn_list{cnt} = tmp;
        end
    end
    fclose(fid);
end