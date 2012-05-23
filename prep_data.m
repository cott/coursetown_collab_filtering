% Rs(test_indices) = values to be guessed
% nonzeros: nonzero indices in r_train
function [train, test_indices, mu] = prep_data(Rs, k)

    flatRs = Rs(:);
    nonzero_is = find(flatRs); % get all nonzero indices (can't test on a zero)
    if k >= length(nonzero_is)
       disp 'ERROR: Your k is too large. You will not have any data left';
    end
    is = randsample(length(nonzero_is), k); % pick k of the eligible indices
    test_indices = nonzero_is(is); % 1-dimensional index of test points
    
    % training set = whole dataset w/ test points zero'd out
    train = Rs;
    train(test_indices) = 0;
    
    % mu = avg. rating. so sum everything and divide by num of nonzero
    % entries that remain in the test set
    mu = sum(train(:)) / (length(nonzero_is) - k);
end
