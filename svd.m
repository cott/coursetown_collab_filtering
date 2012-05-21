%   f:  # latent variables
%   mu: constant
%   Bu: U item vector. User baseline.
%   Bi: I item vector. Item baseline.
%   Q:  fxI matrix. row i = item i's affinity
%   P:  fxU matrix. row u = user u's affinity
%   Rp: IxU matrix. Predicted ratings.

function [e, Bu, Bi, Q, P] = svd(f, lambda, gamma, mu, Rs, max_iter)
    
    [Bu, Bi, Q, P] = bootstrap(f, Rs);
    
    % main loop
    for i = 1:max_iter 
        Rp = predict_ratings(mu, Bu, Bi, Q, P);
        [Bu, Bi, Q, P] = update(Bu, Bi, Q, P, Rp, Rs, gamma, lambda);
    end
    
    e = reg_sq_error(Bu, Bi, Q, P, Rp, Rs, lambda);
end


% % HELPER FUNCTIONS

% TODO what's a smart way to init these?
function [Bu, Bi, Q, P] = bootstrap(f, Rs)
    [U, I] = size(Rs);
    Bu = zeros(U);
    Bi = zeros(I);
    Q = zeros(f,I);
    P = zeros(f,U);
end


function Rp = predict_ratings(mu, Bu, Bi, Q, P)
    % r_ui = mu + b_i + b_u + q_i^T * p_u
    U = size(P,2);
    I = size(Q,2);

    BuBlock = repmat(Bu, I,1);
    BiBlock = repmat(Bi',1,U);
    Rp = mu + BuBlock + BiBlock + Q' * P;
end


function [nBu, nBi, nQ, nP] = update(Bu, Bi, Q, P, Rp, Rs, gamma, lambda)
    E = Rs - Rp; % IxU
    nBu = Bu + gamma * (sum(E,1) - lambda * Bu); % sum E along i's
    nBi = Bi + gamma * (sum(E,2)' - lambda * Bi);
    nQ = Q + gamma * (P * E' - lambda * Q);
    nP = P + gamma * (Q * E  - lambda * P);
end


function e = reg_sq_error(Bu, Bi, Q, P, Rp, Rs, lambda)
    I = size(Q,2);
    U = size(P,2);

    diff_matrix = (Rp - Rs) .^ 2;
    lhs = sum(diff_matrix(:));

    % TODO might have mixed up *U and *I !!!
    Q_norm_sum = sum(Q(:) .^ 2) * U;
    P_norm_sum = sum(P(:) .^ 2) * I;
    Bi_sum = sum(Bi .^ 2) * U;
    Bu_sum = sum(Bu .^ 2) * I;
    rhs = lambda * (Q_norm_sum + P_norm_sum + Bi_sum + Bu_sum);

    e = lhs + rhs;
end