function dist = CUM_LOG(sample, model)

% Returns CUM_LOG distance of 2 input histograms .
% Sample and model histograms must be in the right order
%
%	dist = CUM_LOG(sample, model)
% 	 sample (vector)
%	 model (vector)
%

n = sum(model);
model((model==0))=0.00001;
model=model/n;
sample=sample/sum(sample);

dist=-sum(sample .* log(model));


