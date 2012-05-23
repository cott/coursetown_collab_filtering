% Rs(test_indices) = values to be guessed
% nonzeros: nonzero indices in r_train
function [train, test_indices, mu] = prep_data(Rs, k)

    
    flatRs = Rs(:);
    nonzero_is = find(flatRs); % get all nonzero indices (can't test on a zero)
    
    sample_size = round(length(nonzero_is) * k);
    
    is = randsample(length(nonzero_is), sample_size); % pick k of the eligible indices
    test_indices = nonzero_is(is); % 1-dimensional index of test points
    
    % training set = whole dataset w/ test points zero'd out
    train = Rs;
    train(test_indices) = 0;
    
    % mu = avg. rating. so sum everything and divide by num of nonzero
    % entries that remain in the test set
    mu = sum(train(:)) / (length(nonzero_is) - sample_size);
end
