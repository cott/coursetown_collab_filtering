function e = rounded_error(Rp, Rs, test_indices)
    rp = Rp(test_indices);
    rs = Rs(test_indices);
    
    differ = round(rp) ~= round(rs);
    diff = abs(rp - rs) - 0.5;
    diff = diff(diff > 0);
    
    e = sum(diff .^ 2);
    e = e / length(test_indices);
    e = sqrt(e);
end