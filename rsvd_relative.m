%   f:  # latent variables
%   mu: constant
%   Bu: U item vector. User baseline.
%   Bi: I item vector. Item baseline.
%   Q:  fxI matrix. row i = item i's affinity
%   P:  fxU matrix. row u = user u's affinity
%   Rp: IxU matrix. Predicted ratings.

% RSVD algorithm s.t. Rp = P * Q'
% rather than offset  Rp = P * Q' + B

function Rp = rsvd_relative(f, lambda, gamma, max_iter, Rtrain, min_value, max_value, noise, momentum, stop_threshold_ratio)
    
    
    % HACK! this whole file was written assuming min_value's an integer.
    % but now we're making it the baseline matrix. lulz.
%     min_value = baseline(Rtrain, 2);
    mu = mean(Rtrain(Rtrain > 0));
    min_value = mu * ones(size(Rtrain));
    
    min_value = mu;

%     [Q, P] = bootstrap(Rtrain, f);
    % shrinkage_value (last param of bootstrap) DOES matter. alg might
    % never recover from really bad initial values!
    [U, I] = bootstrap(Rtrain, f, min_value, noise, 2);
        
    errors = zeros(1,max_iter);
    
    nonzero_entries = find(Rtrain > 0);
    nonzero_entries = nonzero_entries(:);
    perm_vector = randperm(numel(nonzero_entries));
    nonzero_entries(perm_vector(:)) = nonzero_entries(:);
    chunk_count = 5; % how many chunks we break nonzero_entries into
    chunk_size = floor(length(nonzero_entries) / chunk_count);
    
    deltaU = zeros(size(U));
    deltaI = zeros(size(I));
       
    % main loop
    for i = 1:max_iter  
        % does clipping during training actually hurt instead of help?
%         Rp = clip(predict_ratings(Q, P, Rbase), 1, max_value);
        Rp = clip(predict_ratings(U,I,min_value), min_value, max_value);
                
        oldU = U;
        oldI = I;
        
        % THIS IS THE ACTUAL IMPORTANT PART
        [U, I, deltaU, deltaI] = update(U, I, Rp, Rtrain, gamma, lambda, momentum, deltaU, deltaI);
        gamma = gamma * 0.9; % decrease step size
        errors(i) = rms_error(Rp, Rtrain, find(Rtrain > 0));
        
        oldU = abs(oldU - U);
        oldI = abs(oldI - I);
        amt_changed = mean(oldU(:)) + mean(oldI(:));
        if abs(amt_changed / (mean(U(:)) + mean(I(:)))) < stop_threshold_ratio
            display(sprintf('converged after %d iterations :)', i));
            break;
        end
        
        ustat = mean(mean(abs(U)));
        istat = mean(mean(abs(I)));
        if isnan(errors(i)) || errors(i) == Inf || errors(i) == -Inf
            display('ERROR: break b/c error rate is Inf or NaN');
            break;
        end
               
%         display(sprintf('[%d] error rate: \t %d. \t u/i : %d \t / \t %d', i, errors(i), ustat, istat));
        
        if istat > 10 || ustat > 10
            display('uh oh. our matrices are getting out of hand. expect the error to EXPLODE!');
        end
    end
end

% WHY!?!? WHY do the initial values matter so much? Why are Bu and Bi
% always positive? Shouldn't they reflect the deviation from the mean (mu)?

function [U, I] = bootstrap(Rs, f, min_value, noise, shrink_coeff)
    [i, u] = size(Rs);
        
    % chih-chao ma (p. 4) recommends these start values:
%     mu = sum(Rs(:)) / sum(sum(Rs > 0));
%     start_term = sqrt((mu - min_value) / f);
%     U = start_term + rand(f,u) * noise - noise/2;
%     I = start_term + rand(f,i) * noise - noise/2;
    
    
    % these seem like smarter initial values...
    avg = mean(Rs(Rs > 0));
    
    % a lot like in the non-relative version, but w/ avg subtracted out
    % NOTE: don't worry about the loss of the negative sign. it'll work
    % itself out. when all the numbers are big and positive, it's easy to
    % get negative ones from them.
    inside = ((shrink_coeff * avg + sum(Rs, 2)) ./ (shrink_coeff + sum(Rs > 0, 2)) - avg) / f;
    Q_col = sqrt(abs(inside));
    Q = repmat(Q_col', f, 1) + rand(f,i) .* noise - noise/2;
        
    inside = ((shrink_coeff * avg + sum(Rs, 1)) ./ (shrink_coeff + sum(Rs > 0, 1)) - avg) / f;
    P_row = sqrt(abs(inside));
    P = repmat(P_row, f, 1) + rand(f,u) .* noise - noise/2;    
          
    U = P;
    I = Q;
end


function R = clip(R, min_value, max_value)
    R(R < 1) = 1;
    R(R > max_value) = max_value;
end


function Rp = predict_ratings(U, I, base)
    Rp = I' * U + base;
%     Rp = sqrt(Rp .* conj(Rp));
end


function [nU, nM, dU, dM] = update(U, M, Rp, Rs, lrate, lambda, momentum, deltaU, deltaM)
    num_m = size(M,2);
    
    E = Rs - Rp;
        
    gradM = zeros(size(M));
    gradU = zeros(size(U));
    
    % only update using KNOWN entries (nonzero in Rs)
    nonzero_is = find(Rs > 0);
%     for j = 1:length(which_entries)
    for j = 1:length(nonzero_is)
        v = nonzero_is(j);
        m = mod(v-1, num_m) + 1;
        u = ceil(v / num_m);
        
        % notes leave it ambiguous whether you subtract the lambda * U
        % within the loop or outside. definitely within, otherwise you can
        % end up subtracting from an entry without otherwise touching it,
        % which is really bad
        gradU(:,u) = gradU(:,u) + E(m,u) * M(:,m) - lambda * U(:,u);
        gradM(:,m) = gradM(:,m) + E(m,u) * U(:,u) - lambda * M(:,m);
    end
    
    dU = deltaU * momentum + lrate * gradU;
    dM = deltaM * momentum + lrate * gradM;
    
    nU = U + dU;
    nM = M + dM;
end

