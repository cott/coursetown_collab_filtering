% trivial utility function
function R = clip_to_range(R, max_value)
    R = max(1, min(max_value, R));
end