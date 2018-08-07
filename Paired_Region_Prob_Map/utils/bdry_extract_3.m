function [x,y,t,c]=bdry_extract_3(V)
% [x,y,t,c]=bdry_extract_3(V);
% compute (r,theta) histograms for points along boundary of single 
% region in binary image V 

% extract a set of boundary points w/oriented tangents
Vg=V;
[height, width] = size(Vg);
[c1 c2] = find(Vg~=0);
c=[c2 c1]';

fz=c(1,:)~=0.5;
c(:,find(~fz))=NaN;
B=c(:,find(fz));

npts=size(B,2);
t=zeros(npts,1);
for n=1:npts
   x0=round(B(1,n));
   y0=round(B(2,n));
   sx = 0; sy = 0;
   for ix=max(x0-3, 1):min(x0+3, width)
   for iy=max(y0-3, 1):min(y0+3, height)
       if Vg(iy, ix)
           dx=ix-x0; dy=iy-y0;
           if dx<0, dx=-dx; dy=-dy; end
           if dx==0 && dy<0, dx=-dx; dy=-dy; end
           sx=sx+dx; sy=sy+dy;
       end
   end
   end
   t(n)=atan2(sy, sx);
end

x=B(1,:)';
y=B(2,:)';

