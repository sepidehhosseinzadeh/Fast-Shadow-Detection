function dist = calcDistance(i, j, areas, centroids)
  raw_dist = sqrt((centroids(i,1) - centroids(j,1))^2 
     + (centroids(i,2) - centroids(j,2))^2);
  dist = raw_dist/sqrt(sqrt(areas(i) * areas(j)));
end
