function [word] = assignWords(data, clusters)
% function [map] = assignTextons(fim,textons)

d2 = distSqr(data' , clusters');
[y,map] = min(d2,[],2);
word = map;
