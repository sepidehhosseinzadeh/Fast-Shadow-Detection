function ratio = cal_ratio(seg,pair,hard_map,soft_map,im)
    hard_map_2=conv2(hard_map,fspecial('average',20),'same');
    imcanny=edge(hard_map_2,'canny');
    %imcanny=edge(hard_map,'canny');
    [x,y,t]=bdry_extract_3(imcanny);
    
    ps = 4;
    [height width] = size(double(imcanny));
    npt = length(x);
    
    pair_ratio=[];
    
    nonpair_ratio=[];

    for n=1:npt
        skip=false;
        x0 = x(n); y0 = y(n); t0 = t(n);
        x1 = round(x0 + 1.5*ps*cos(t0+pi/2));
        y1 = round(y0 + 1.5*ps*sin(t0+pi/2));
        x2 = round(x0 + 1.5*ps*cos(t0-pi/2));
        y2 = round(y0 + 1.5*ps*sin(t0-pi/2));
        %hist{1} = ones(3, 1);        hist{2} = ones(3, 1);
        val{1} = zeros(3, 1);        val{2} = zeros(3, 1);
        k=zeros(2,1);
        %cnt1 = 0; cnt2 = 0;
        ix = max(1, x1 - ps):min(width, x1 + ps);
        iy = max(1, y1 - ps):min(height, y1 + ps);
        for ch=1:3
            temp = im(iy, ix, ch);
            val{1}(ch) = mean(temp(:));

            %% find the label for this tmp area

            if isnan(val{1}(ch))
                skip=true;
            end
        end
       
        label=seg(iy, ix);
        label1=mode(double(label(:)));

        temp=soft_map(iy,ix);
        k(1)=mean(temp(:));
        if isnan(k(1))
            skip=true;
        end
        
        temp=hard_map(iy,ix);
        if mean(temp(:))<0.8 && mean(temp(:))>0.2
            skip=true;
        end
        
        %region1=sum(sum(hard_map(iy,ix)));
        
        ix = max(1, x2 - ps):min(width, x2 + ps);
        iy = max(1, y2 - ps):min(height, y2 + ps);
        for ch=1:3
            temp = im(iy, ix, ch);
            val{2}(ch) = mean(temp(:));
            if isnan(val{2}(ch))
                skip=true;
            end 
        end

        label=seg(iy, ix);
        label2=mode(double(label(:)));

        temp=soft_map(iy,ix);
        k(2)=mean(temp(:));
        if isnan(k(2))
            skip=true;
        end
        
        temp=hard_map(iy,ix);
        if mean(temp(:))<0.8 && mean(temp(:))>0.2
            skip=true;
        end
        
        if abs(k(1)-k(2))<0.5
            skip = true;
        end

        tmp_ratio = []; 

        if ~skip
            for ch=1:3
                tmp_ratio=[tmp_ratio,(val{2}(ch)-val{1}(ch))/(val{1}(ch)*k(2)-val{2}(ch)*k(1))];
                if isnan(tmp_ratio(ch))
                    skip = true;
                end
            end
        end
    
        if ~skip
           % test to see if label1 and label2 make a pair
           if ~isempty(pair) && sum((pair(:,1) == label1 & pair(:,2) == label2) | (pair(:,2) == label2 & pair(:,2) == label1))~=0
              pair_ratio = [pair_ratio; tmp_ratio];
           else
              nonpair_ratio = [nonpair_ratio; tmp_ratio];
           end
        end

    end
    
    if ~isempty(pair_ratio)
      pair_ratio(pair_ratio(:,1)<=0|pair_ratio(:,2)<=0|pair_ratio(:,3)<=0,:)=[];
    end
    if ~isempty(nonpair_ratio)
      nonpair_ratio(nonpair_ratio(:,1)<=0|nonpair_ratio(:,2)<=0|nonpair_ratio(:,3)<=0,:)=[];
    end
    if size(pair_ratio,1) == 0 && size(nonpair_ratio,1)>0
       ratio=nonpair_ratio;
    else
       ratio = pair_ratio;
    end


    step_size = 0.1;
    
    pt_num = size(ratio,1);
    
    hash_mode = 99991;
    hash_map = zeros(4, hash_mode);
    
   
%% voting

 
    for i = 1:pt_num
        if mod(i, 100)==0
            fprintf(1, '%d\n', i);
        end
        ir = floor(ratio(i,1)/step_size); ir = [ir ir+1];
        ig = floor(ratio(i,2)/step_size); ig = [ig ig+1];
        ib = floor(ratio(i,3)/step_size); ib = [ib ib+1];
        if isinf(ir) | isinf(ig) | isinf(ib)
            continue
        end
        for i1=ir
        for i2=ig
        for i3=ib
            hashins([i1, i2, i3]);
        end
        end
        end
        
    end
       
    [dummy,ind] = max(hash_map(1,:));
    
    ratio = hash_map(2:4,ind);
    ratio = ratio*step_size;
    
    function ind = hashins(entry)
        h = round(entry(1)*1e2 + entry(2)*10 + entry(3));
        hm = h;
        cnt = 0;
        while 1
            hm = mod(hm, hash_mode) + 1;
            if hash_map(1, hm) == 0
                hash_map(1, hm) = 1;
                hash_map(2:4, hm) = entry;
                break;
            elseif sum ( hash_map(2:4, hm) == entry' ) == 3
                hash_map(1, hm) = hash_map(1, hm) + 1;
                
                break;
            else
                hm = hm+2*cnt+1;
                cnt = cnt + 1;
            end
        end
        ind = hm;
    end

end
