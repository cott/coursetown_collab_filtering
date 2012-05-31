%   f:  # latent variables
%   mu: constant
%   Bu: U item vector. User baseline.
%   Bi: I item vector. Item baseline.
%   Q:  fxI matrix. row i = item i's affinity
%   P:  fxU matrix. row u = user u's affinity
%   Rp: IxU matrix. Predicted ratings.

function Rp = collab_svd(f, lambda, gamma, max_iter, max_value, Rtrain, noise, shrinkage_factor)
    
    mu = mean(Rtrain(Rtrain > 0));
    [Bu, Bi, Q, P] = bootstrap(Rtrain, f, noise, shrinkage_factor);
    
    % main loop
    for i = 1:max_iter  
        Rp = clip_to_range(predict_ratings(mu, Bu, Bi, Q, P), max_value);
        [Bu, Bi, Q, P] = update(Bu, Bi, Q, P, Rp, Rtrain, gamma, lambda);
        gamma = gamma * 0.9; % decrease step size
    end
end

function [Bu, Bi, Q, P] = bootstrap(Rs, f, noise, shrinkage_factor)
    [U, I] = size(Rs);
    [x, Bu, Bi] = baseline(Rs, shrinkage_factor);
    Bi = Bi';
    
    % NOTE: cols of Q and P always update as simply a sum of other cols in
    % Q and P, so if we start them all at the same value, for instance,
    % it's impossible for different features to have different values
    % within a single column. ATM just randomize, but this is bad!
        
    Q = rand(f,U) .* (noise * 2) - noise;
    P = rand(f,I) .* (noise * 2) - noise;    
end


function Rp = predict_ratings(mu, Bu, Bi, Q, P)
    % the paper says: r_ui = mu + b_i + b_u + q_i^T * p_u
    U = size(P,2);
    I = size(Q,2);
    
    Rp = Q' * P;
%     Rp = zeros(I,U); % TODO REMOVE THIS. then FIX THIS ISSUE.
        
    % benchmarks say the best way to do this is by looping along this
    % (small) dimension. cuts runtime in 1/3
    for u = 1:U
         Rp(:,u) = Rp(:,u) + Bu(u) + Bi' + mu;
    end
end

function [nBu, nBi, nQ, nP] = update(Bu, Bi, Q, P, Rp, Rs, gamma, lambda)
    U = size(P,2);
    I = size(Q,2);
    
    E = Rs - Rp;
%     Eproper = E .* (Rs > 0);
    
    % as an easy way to set the dimensions, copy the matrices
    % NOTE: matlab does a deep copy here
    nBu = Bu;
    nBi = Bi;
    nQ = Q;
    nP = P;
    
    % only update using KNOWN entries (nonzero in Rs)
    nonzero_is = find(Rs);
    for j = 1:length(nonzero_is)
        v = nonzero_is(j);
        i = mod(v-1,I) + 1;
        u = ceil(v / I);
        
        nBu(u) = nBu(u) + gamma * (E(i,u) - lambda * nBu(u)); % TODO
        nBi(i) = nBi(i) + gamma * (E(i,u) - lambda * nBi(i));
        nQ(:,i) = nQ(:,i) + gamma * (E(i,u) * P(:,u) - lambda * nQ(:,i));
        nP(:,u) = nP(:,u) + gamma * (E(i,u) * Q(:,i) - lambda * nP(:,u));
    end
            
    % look for fishy trends w/ P and Q
    P_mag = mean(abs(nP(:))) - mean(abs(P(:)));
    Q_mag = mean(abs(nQ(:))) - mean(abs(Q(:)));
    
    diff = mean(abs(nP(:) - P(:)));
end

