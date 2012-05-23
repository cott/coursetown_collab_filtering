function e = rms_error(Rp, Rs, test_indices)
    diff_sq = (Rp(test_indices) - Rs(test_indices)) .^ 2;
    diff = sum(diff_sq(:));
    
    e = sqrt(diff / length(test_indices));
end