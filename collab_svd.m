%   f:  # latent variables
%   mu: constant
%   Bu: U item vector. User baseline.
%   Bi: I item vector. Item baseline.
%   Q:  fxI matrix. row i = item i's affinity
%   P:  fxU matrix. row u = user u's affinity
%   Rp: IxU matrix. Predicted ratings.

function Rp = collab_svd(f, lambda, gamma, max_iter, max_value, Rtrain)
    
    mu = mean(Rtrain(Rtrain > 0));
    [Bu, Bi, Q, P] = bootstrap(Rtrain, f);
    
    % main loop
    for i = 1:max_iter  
        Rp = clip_to_range(predict_ratings_easy(mu, Bu, Bi, Q, P), max_value);
        [Bu, Bi, Q, P] = update_procedural(Bu, Bi, Q, P, Rp, Rtrain, gamma, lambda);
        gamma = gamma * 0.9; % decrease step size
    end
end

% WHY!?!? WHY do the initial values matter so much? Why are Bu and Bi
% always positive? Shouldn't they reflect the deviation from the mean (mu)?

% TODO what's a smart way to init these?
function [Bu, Bi, Q, P] = bootstrap(Rs, f)
    [U, I] = size(Rs);
    [x, Bu, Bi] = baseline(Rs);
    Bi = Bi';
    
    % NOTE: cols of Q and P always update as simply a sum of other cols in
    % Q and P, so if we start them all at the same value, for instance,
    % it's impossible for different features to have different values
    % within a single column. ATM just randomize, but this is bad!
    Q = rand(f,U) .* 0.5 - 0.25;
    P = rand(f,I) .* 0.5 - 0.25;
%     Q = zeros(f,U) + 0.01;
%     P = zeros(f,I) + 0.01;


% 
%     sumTotal = sum(Rs(:));
%     countTotal = sum(Rs(:) > 0);
%     mu = sumTotal / countTotal;
% 
%     dev = mean(abs(Rs(Rs > 0) - mu));
% 
%     nonzero = Rs > 0;
% 
%     countX = sum(nonzero, 1);
%     countY = sum(nonzero, 2);
% 
%     sumX = sum(Rs,1) - mu * countX;
%     sumY = sum(Rs,2) - mu * countY;
% 
%     n = 3;
% 
%     avgX = (n * dev + sumX) ./ (countX + n);
%     avgY = (n * dev + sumY) ./ (countY + n);
%     
%     Bu = avgX;
%     Bi = avgY';
    
end


function Rp = predict_ratings_easy(mu, Bu, Bi, Q, P)
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


% PROBLEM: we're only supposed to update the vectors for the KNOWN vertices
% right now we're subtracting too much! or too little.
function [nBu, nBi, nQ, nP] = update(Bu, Bi, Q, P, Rp, Rs, gamma, lambda)
    % NOTE: only look at error on the KNOWN entries in Rs
    % even though Rs predicts a value for EVERY entry
    
    f = size(P,1);
    
    % foundEntries: mark nonzero entries in Rs
    exist = Rs > 0;
    E = (Rs - Rp) .* exist; % UxI
    uCount = sum(exist,1); % 1xU. actually sums along i's...
    iCount = sum(exist,2); % Ix1
    uRep = repmat(uCount, f,1); % fxU. num u's s.t. e_i,u > 0
    iRep = repmat(iCount, 1,f); % Ixf.
    
    nBu = Bu + gamma * (sum(E,1)  - uCount .* (lambda * Bu)); % sum E along i's
    nBi = Bi + gamma * (sum(E,2)' - iCount' .* (lambda * Bi));
    
    nP = P + gamma * (Q * E  - uRep  .* (lambda * P));
    nQ = Q + gamma * (P * E' - iRep' .* (lambda * Q));
end

function [nBu, nBi, nQ, nP] = update_procedural(Bu, Bi, Q, P, Rp, Rs, gamma, lambda)
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
        
        nBu(u) = nBu(u) + gamma * (E(i,u) - lambda * nBu(u));
        nBi(i) = nBi(i) + gamma * (E(i,u) - lambda * nBi(i));
        nQ(:,i) = nQ(:,i) + gamma * (E(i,u) * nP(:,u) - lambda * nQ(:,i));
        nP(:,u) = nP(:,u) + gamma * (E(i,u) * nQ(:,i) - lambda * nP(:,u));
    end
            
    % look for fishy trends w/ P and Q
    P_mag = mean(abs(nP(:))) - mean(abs(P(:)));
    Q_mag = mean(abs(nQ(:))) - mean(abs(Q(:)));
    
    diff = mean(abs(nP(:) - P(:)));
end

