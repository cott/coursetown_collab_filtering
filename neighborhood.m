function [ Rp, Rp2 ] = neighborhood( R, k, shrinkage_factor, lambda8, test_indices )
%NEIGHBORHOOD Summary of this function goes here
%   Detailed explanation goes here

% shrinkage_factor = how much to shrink the baseline
% lambda8 = another shrink factor. how much to shrink pearson score.

    B = baseline(R, shrinkage_factor);
    S = shrunk_pearson_score(R, B, lambda8);
    
    display('pearson score: calculated.');
    
    Rp = predict_only_answers(R, B, S, k, test_indices);
    
    % alternate prediction
    Rp2 = predict_only_answers(R, B, S .^ 2, k, test_indices);
end

function S = shrunk_pearson_score(R, B, lambda8)
    offset = R - B;
    [I, U] = size(R);
    % don't need to zero the entries that should be zero here because we'll
    % end up ignoring them in the loop anyways
    offset(R == 0) = 0;
    offset_sq = offset .^ 2;

    S = zeros(I,I);
    
    % sum ratings by users that rated both i and j
    for i=1:I
        for j=(i+1):I
            
            mask = R(i,:) & R(j,:);
            n = sum(mask(:));
            if n == 0  
                continue
            end
            
            num = offset(i,:) * offset(j,:)';
            
            denom = sqrt(mask * offset_sq(i,:)' + mask * offset_sq(j,:)');
            
            p = num / denom;
            
            S(i,j) = p * (n - 1) / (n - 1 + lambda8);
            S(j,i) = S(i,j);
        end
    end
end

% only predicts entry at test_indices. FOR SPEED.
function Rp = predict_only_answers(R, B, S, k, test_indices)

    [I, U] = size(R);
    Rp = B;
    
    offset = R - B;
    
    for j=1:size(test_indices,2)
        
        v = test_indices(j);
        i = mod(v-1,I) + 1;
        u = ceil(v / I);
        
        if R(i,u) > 0
           continue;
        end
        
        ratings = R(:,u);
        if sum(ratings > 0) == 0
            continue;
        end
        
        s_row = S(i,:);
        if sum(s_row) == 0
            continue;
        end
        
        Rp(i,u) = B(i,u) + s_row * offset(:,u) / sum(s_row(:));
    end
end


function Rp = predict(R, B, S, k)

    [I, U] = size(R);
    Rp = B;
    
    offset = R - B;
    
    found_count = 0;
        
    % find k neighbors
    for u=1:U
        for i=1:I
            
            if R(i,u) > 0
                continue;
            end
        
            % items rated by u
            ratings = R(:,u);
            
            if sum(ratings > 0) == 0
                continue;
            end
            
%             [~, ix] = sort(ratings,1,'descend');
%             ix = ix(1:k);
%             Sk = S(i, ix);
%             
%             if sum(Sk) == 0
%                 continue;
%             end
%             
%             found_count = found_count + 1;
%                         
%             s_row = zeros(1,I);
%             s_row(ix) = Sk;
            
            s_row = S(i,:);
            if sum(s_row) == 0
                continue;
            end
            
            % TODO there's no good reason to limit to k nearest neighbors,
            % since we're doing this the computationally inefficient way
            % anyways... and odds are we won't find k neighbors anyways.
            
            Rp(i,u) = B(i,u) + s_row * offset(:,u) / sum(Sk(:));
        end
    end
    
    display(sprintf('this alg made a difference at %d locations', found_count));
end