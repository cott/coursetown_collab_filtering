%   f:  # latent variables
%   mu: constant
%   Bu: U item vector. User baseline.
%   Bi: I item vector. Item baseline.
%   Q:  fxI matrix. row i = item i's affinity
%   P:  fxU matrix. row u = user u's affinity
%   Rp: IxU matrix. Predicted ratings.

function Rp = rsvd(f, lambda, gamma, max_iter, Rbase, Rtrain, Qseed, Pseed, max_value)
    
%     [Q, P] = bootstrap(Rtrain, f);
    Q = Qseed;
    P = Pseed;
        
    % main loop
    for i = 1:max_iter  
        % does clipping during training actually hurt instead of help?
%         Rp = clip(predict_ratings(Q, P, Rbase), 1, max_value);
        Rp = predict_ratings(Q,P,Rbase);
        
        [Q, P] = update(Q, P, Rp, Rtrain, gamma, lambda);
        gamma = gamma * 0.9; % decrease step size
    end
end

% WHY!?!? WHY do the initial values matter so much? Why are Bu and Bi
% always positive? Shouldn't they reflect the deviation from the mean (mu)?

% TODO what's a smart way to init these?
function [Q, P] = bootstrap(Rs, f)
    [I, U] = size(Rs);
    
    % NOTE: cols of Q and P always update as simply a sum of other cols in
    % Q and P, so if we start them all at the same value, for instance,
    % it's impossible for different features to have different values
    % within a single column. ATM just randomize, but this is bad!
    Q = rand(f,I) .* 0.5 - 0.25;
    P = rand(f,U) .* 0.5 - 0.25;
end


function R = clip(R, min_value, max_value)
    R(R < min_value) = min_value;
    R(R > max_value) = max_value;
end


function Rp = predict_ratings(Q, P, base)
    Rp = Q' * P + base;
end


function [nQ, nP] = update(Q, P, Rp, Rs, lrate, lambda)
    U = size(P,2);
    I = size(Q,2);
    
    E = Rs - Rp;
    
    nQ = Q;
    nP = P;
    
    % only update using KNOWN entries (nonzero in Rs)
    nonzero_is = find(Rs > 0);
    for j = 1:length(nonzero_is)
        v = nonzero_is(j);
        i = mod(v-1,I) + 1;
        u = ceil(v / I);
        
        % update using old P & Q.
        % WAIT this is weird. we could just use P and Q instead of nP and
        % nQ... but instead we're using semi-new data?
        old_q = nQ(:,i);
        nQ(:,i) = (1 - lrate * lambda) * Q(:,i) + lrate * (E(i,u) * P(:,u));
        nP(:,u) = (1 - lrate * lambda) * P(:,u) + lrate * (E(i,u) * Q(:,i));
    end
            
    % look for fishy trends w/ P and Q
%     P_mag = mean(abs(nP(:)));
%     Q_mag = mean(abs(nQ(:)));
%     diff = mean(abs(nP(:) - P(:)));
%     display(sprintf('deltas: %f , %f , %f', P_mag, Q_mag, diff));
end

