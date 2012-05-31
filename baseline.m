function [ Rp, offset_row, offset_col ] = baseline(R_train, shrinkage_factor)
%BASELINE predictor: uses avg movie rating & avg user offset to predict.
% source: http://sifter.org/~simon/journal/20061211.html

mu = mean(R_train(R_train > 0));

n = shrinkage_factor;

avg_row = (n * mu + sum(R_train,1)) ./ (n + sum(R_train > 0,1));
avg_col = (n * mu + sum(R_train,2)) ./ (n + sum(R_train > 0,2));

offset_row = avg_row - mu;
offset_col = avg_col - mu;

Rp = repmat(avg_col, 1, size(R_train,2)) + repmat(offset_row, size(R_train,1),1);

end
