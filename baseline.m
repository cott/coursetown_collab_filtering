function [ Rp ] = baseline(R_train)
%BASELINE predictor: uses avg movie rating & avg user offset to predict.
% source: http://sifter.org/~simon/journal/20061211.html

n = 3;

avg_row = (n * mu + sum(R_train,1)) ./ (n + sum(R_train > 0,1));
avg_col = (n * mu + sum(R_train,2)) ./ (n + sum(R_train > 0,2));

offset_row = avg_row - mu;

Rp = repmat(avg_col, 1, size(R_train,2)) + repmat(offset_row, size(R_train,1),1);

end
