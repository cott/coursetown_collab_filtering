%   f:  # latent variables
%   mu: constant
%   Bu: U item vector. User baseline.
%   Bi: I item vector. Item baseline.
%   Q:  fxI matrix. row i = item i's affinity
%   P:  fxU matrix. row u = user u's affinity
%   Rp: IxU matrix. Predicted ratings.

function Rp = collab_svd_plus(f, lambda5, lambda6, gamma, max_iter, max_value, Rtrain)
    
    [Bu, Bi, Q, P, Y] = bootstrap(Rtrain, f);
    
    rated = Rtrain > 0;
    mu = mean(Rtrain(rated));
    % num_rated_mtx is a diag matrix s.t. if you num_rated_mtx * A for a
    % f x U matrix A will multiply each column (user) by its appropriate
    % value of |R(u)|^(-0.5)
    num_rated = sum(rated, 1) .^ -0.5;
    num_rated_mtx = diag(num_rated);
    
    % main loop
    for i = 1:max_iter  
        Rp = clip_to_range(predict_ratings(mu, Bu, Bi, Q, P, Y, rated, num_rated_mtx), max_value);
        [Bu, Bi, Q, P, Y] = update(Bu, Bi, Q, P, Y, Rp, Rtrain, gamma, lambda5, lambda6, num_rated);
        gamma = gamma * 0.9; % decrease step size
    end
end


% TODO what's a smart way to init these?
function [Bu, Bi, Q, P, Y] = bootstrap(Rs, f)
    [U, I] = size(Rs);    
    Bu = mean(Rs, 1) + 0.001; % TODO: i think these are basically what they're supposed to be?
    Bi = mean(Rs, 2)' + 0.001;
    Q = zeros(f,U) + 0.01;
    P = zeros(f,I) + 0.01;
    Y = Q;
end


function R = clip_to_range(R, max_value)
    R(R < 0) = 0;
    R(R > max_value) = max_value;
end


function Rp = predict_ratings(mu, Bu, Bi, Q, P, Y, rated, num_rated_mtx)
    % r_ui = mu + b_i + b_u + q_i^T * p_u
    U = size(P,2);
      
    % modify P with this new factor 
    % a = |R(u)|^(-0.5) * sum_(j in R(u)) * y[j]
    newP = P + (Y * rated * num_rated_mtx);
    
    Rp = Q' * newP;
        
    % benchmarks say the best way to do this is by looping along this
    % (small) dimension. cuts runtime in 1/3
    for u = 1:U
         Rp(:,u) = Rp(:,u) + Bu(u) + Bi' + mu;
    end
end

function [nBu, nBi, nQ, nP, nY] = update(Bu, Bi, Q, P, Y, Rp, Rs, gamma, lambda5, lambda6, num_rated)
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
    nY = Y;
    
    % only update using KNOWN entries (nonzero in Rs)
    nonzero_is = find(Rs);
    for j = 1:length(nonzero_is)
        v = nonzero_is(j);
        i = mod(v-1,I) + 1;
        u = ceil(v / I);
        
        nBu(u) = nBu(u) + gamma * (E(i,u) - lambda5 * nBu(u));
        nBi(i) = nBi(i) + gamma * (E(i,u) - lambda5 * nBi(i));
        nQ(:,i) = nQ(:,i) + gamma * (E(i,u) * P(:,u) - lambda6 * nQ(:,i));
        nP(:,u) = nP(:,u) + gamma * (E(i,u) * Q(:,i) - lambda6 * nP(:,u));
        
        % update Y
        y_indices = find(Rs(:,u));
        for j2 = 1:length(y_indices)
           yj = y_indices(j2);
           nY(:,yj) = (1 - gamma * lambda6) * nY(:,yj) + gamma * (E(i,u) * num_rated(u) * Q(:,i));
        end
    end
end