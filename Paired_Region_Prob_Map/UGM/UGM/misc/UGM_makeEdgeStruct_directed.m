function [edgeStruct] = UGM_makeEdgeStruct_directed(adj,nStates,useMex,maxIter)
% [edgeStruct] = UGM_getEdgeStructure(adj,nStates,useMex,maxIter)
%
% adj - nNodes by nNodes adjacency matrix (0 along diagonal)
%

if nargin < 3
    useMex = 1;
end
if nargin < 4
    maxIter = 100;
end

nNodes = length(adj);
[i j] = ind2sub([nNodes nNodes],find(adj));
nEdges = length(i);
edgeEnds = zeros(nEdges,2);
eNum = 0;
for e = 1:nEdges
       edgeEnds(eNum+1,:) = [i(e) j(e)];
       eNum = eNum+1;
end

[V,E] = UGM_makeEdgeVE(edgeEnds,nNodes);


edgeStruct.edgeEnds = edgeEnds;
edgeStruct.V = V;
edgeStruct.E = E;
edgeStruct.nNodes = nNodes;
edgeStruct.nEdges = size(edgeEnds,1);

% Handle other arguments
if isscalar(nStates)
   nStates = repmat(nStates,[nNodes 1]);
end
edgeStruct.nStates = nStates(:);
edgeStruct.useMex = useMex;
edgeStruct.maxIter = maxIter;


