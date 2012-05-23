function e = mae_error(Rp, Rs, test_indices)
% Mean Average Error: incredibly simple error metric

    diff_sq = abs(Rp(test_indices) - Rs(test_indices));
    e = sum(diff_sq(:)) / length(test_indices);
end