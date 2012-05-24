% trivial utility function
function R = clip_to_range(R, max_value)
    R(R < 1) = 1;
    R(R > max_value) = max_value;
end