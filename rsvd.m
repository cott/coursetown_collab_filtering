%   f:  # latent variables
%   mu: constant
%   Bu: U item vector. User baseline.
%   Bi: I item vector. Item baseline.
%   Q:  fxI matrix. row i = item i's affinity
%   P:  fxU matrix. row u = user u's affinity
%   Rp: IxU matrix. Predicted ratings.

function Rp = rsvd(f, lambda, gamma, max_iter, max_value, Rtrain)
    
    mu = mean(Rtrain(Rtrain > 0));
    [Bu, Bi, Q, P] = bootstrap(Rtrain, f);
    
    % main loop
    for i = 1:max_iter  
        Rp = clip_to_range(predict_ratings(mu, Q, P), max_value);
        [Q, P] = update(Q, P, Rp, Rtrain, gamma, lambda);
        gamma = gamma * 0.9; % decrease step size
    end
end

% WHY!?!? WHY do the initial values matter so much? Why are Bu and Bi
% always positive? Shouldn't they reflect the deviation from the mean (mu)?

% TODO what's a smart way to init these?
function [Bu, Bi, Q, P] = bootstrap(Rs, f)
    [U, I] = size(Rs);    
    
    % NOTE: cols of Q and P always update as simply a sum of other cols in
    % Q and P, so if we start them all at the same value, for instance,
    % it's impossible for different features to have different values
    % within a single column. ATM just randomize, but this is bad!
    Q = rand(f,U) .* 0.5 - 0.25;
    P = rand(f,I) .* 0.5 - 0.25;
  
end


function R = clip_to_range(R, max_value)
    R(R < 0) = 0;
    R(R > max_value) = max_value;
end


function Rp = predict_ratings(Q, P)
    Rp = Q' * P;
end


function [nQ, nP] = update(Q, P, Rp, Rs, lrate, lambda)
    U = size(P,2);
    I = size(Q,2);
    
    E = Rs - Rp;
    
    nQ = Q;
    nP = P;
    
    % only update using KNOWN entries (nonzero in Rs)
    nonzero_is = find(Rs);
    for j = 1:length(nonzero_is)
        v = nonzero_is(j);
        i = mod(v-1,I) + 1;
        u = ceil(v / I);
        
        nQ(:,i) = (1 - lrate * lambda) * nQ(:,i) + lrate * (E(i,u) * nP(:,u) - lambda * nQ(:,i));
        nP(:,u) = nP(:,u) + lrate * (E(i,u) * nQ(:,i) - lambda * nP(:,u));
    end
            
    % look for fishy trends w/ P and Q
    P_mag = mean(abs(nP(:))) - mean(abs(P(:)))
    Q_mag = mean(abs(nQ(:))) - mean(abs(Q(:)))
    
    diff = mean(abs(nP(:) - P(:)))
end

