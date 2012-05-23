function [ Rp ] = baseline(R_train)
%BASELINE Summary of this function goes here
%   Detailed explanation goes here

sumTotal = sum(R_train(:));
countTotal = sum(R_train(:) > 0);
mu = sumTotal / countTotal;

dev = mean(abs(R_train(R_train > 0) - mu));

nonzero = R_train > 0;

countX = sum(nonzero, 1);
countY = sum(nonzero, 2);

sumX = sum(R_train,1) - mu * countX;
sumY = sum(R_train,2) - mu * countY;

n = 3;

avgX = (n * dev + sumX) ./ (countX + n);
avgY = (n * dev + sumY) ./ (countY + n);

Rp = repmat(avgX,size(R_train,1),1) + repmat(avgY,1,size(R_train,2)) + mu;

end
