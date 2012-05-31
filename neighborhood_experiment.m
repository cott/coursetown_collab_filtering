% NEIGHBORHOOD EXPERIMENT
% WARNING: make sure you run the setup from booktest first. this is not a
% standalone script. I know, poor style, etc.

%% neighborhood-based approach

% WARNING: SUPER SLOW! ALSO: doesn't work! Does demonstrably worse at the
% few areas where it differs from the baseline! jk that's not true for big
% datasets

% TODO: debug.

% QUESTION: WHY does it not seem to matter what I choose for lambda8?
% this might happen because every collision collides in only 1 place
shrinkage = 1;
B = baseline(train,shrinkage);
[Rp, Rp_sq] = neighborhood(train, 10, shrinkage, 100);

%%

raw_diff = Rp - R;
diff = zeros(size(R));
diff(test_indices) = raw_diff(test_indices);
diff(Rp == B) = 0;

raw_diff_sq = Rp_sq - R;
diff_sq = zeros(size(R));
diff_sq(test_indices) = raw_diff_sq(test_indices);
diff_sq(Rp_sq == B) = 0;

%%

neighbor_d = diff(diff > 0);
numel(neighbor_d)

base_diff = B - R;
base_d = base_diff(diff > 0);

display(sprintf('improvement: %d', mean(abs(base_d)) - mean(abs(neighbor_d))));


d_sq = diff_sq(diff_sq > 0);
numel(d_sq)
display(sprintf('sq. improvement: %d', mean(abs(base_d))-mean(abs(d_sq))));

base_err = rms_error(B,R,test_indices);
neighbor_err = rms_error(Rp,R,test_indices);
neighbor_sq_err = rms_error(Rp_sq,R,test_indices);
display(sprintf('ERRORS: base %d, neighbor %d, neighbor_sq %d', base_err, neighbor_err, neighbor_sq_err));
