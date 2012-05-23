%   f:  # latent variables
%   mu: constant
%   Bu: U item vector. User baseline.
%   Bi: I item vector. Item baseline.
%   Q:  fxI matrix. row i = item i's affinity
%   P:  fxU matrix. row u = user u's affinity
%   Rp: IxU matrix. Predicted ratings.

function Rp = collab_svd(f, lambda, gamma, max_iter, max_value, Rtrain, mu)
    
    [Bu, Bi, Q, P] = bootstrap(Rtrain, f);
    
    % main loop
    for i = 1:max_iter  
        Rp = clip_to_range(predict_ratings_easy(mu, Bu, Bi, Q, P), max_value);
        [Bu, Bi, Q, P] = update_procedural(Bu, Bi, Q, P, Rp, Rtrain, gamma, lambda);
    end
end


% TODO what's a smart way to init these?
function [Bu, Bi, Q, P] = bootstrap(Rs, f)
    [U, I] = size(Rs);    
    Bu = mean(Rs, 1) + 0.001; % TODO: i think these are basically what they're supposed to be?
    Bi = mean(Rs, 2)' + 0.001;
    Q = zeros(f,U) + 0.01;
    P = zeros(f,I) + 0.01;
end


function R = clip_to_range(R, max_value)
    R(R < 0) = 0;
    R(R > max_value) = max_value;
end


function Rp = predict_ratings(mu, Bu, Bi, Q, P)
    % r_ui = mu + b_i + b_u + q_i^T * p_u
    U = size(P,2);
    I = size(Q,2);

    BuBlock = repmat(Bu, I,1);
    BiBlock = repmat(Bi',1,U);
    Rp = mu + BuBlock + BiBlock + Q' * P;
end

function Rp = predict_ratings_easy(mu, Bu, Bi, Q, P)
    % r_ui = mu + b_i + b_u + q_i^T * p_u
    U = size(P,2);
    I = size(Q,2);
    
    % easy way
    
    Rp = mu + Q' * P;

    % I don't know what's faster...
    %  creating a repmat matrix of Bu/Bi
    %  looping & adding Bu and Bi directly
    %  looping and adding scalar Bu(u) and Bi(i) to each row/col
    for i = 1:I
        Rp(i,:) = Rp(i,:) + Bu;
    end
    for u = 1:U
        Rp(:,u) = Rp(:,u) + Bi';
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
    
%     aBu = Bu;
    
    for i = 1:I
        
%         count = sum(Rs(i,:) > 0);
%         aBu = aBu + gamma * (sum(Eproper(i,:),1) - count .* (lambda * aBu));
        
        for u = 1:U
           
            % only update KNOWN entries
            if Rs(i,u) == 0
                continue
            end
            
            nBu(u) = nBu(u) + gamma * (E(i,u) - lambda * nBu(u));
%             nBu(u) = nBu(u) + gamma * (E(i,u));
            nBi(i) = nBi(i) + gamma * (E(i,u) - lambda * nBi(i));
            nQ(:,i) = nQ(:,i) + gamma * (E(i,u) * P(:,u) - lambda * nQ(:,i));
            nP(:,u) = nP(:,u) + gamma * (E(i,u) * Q(:,i) - lambda * nP(:,u));
        end
    end

%     % check aBu = nBu
%     if ~isequal(aBu,nBu)
%         'OH NO'
%         aBu(1:10)
%         nBu(1:10)
%     end
end

function [nBu, nBi, nQ, nP] = update_semi_procedural(Bu, Bi, Q, P, Rp, Rs, gamma, lambda)
    U = size(P,2);
    I = size(Q,2);
    
    E = Rs - Rp;
    
    % as an easy way to set the dimensions, copy the matrices
    % NOTE: matlab does a deep copy here
    nBu = Bu;
    nBi = Bi;
    nQ = Q;
    nP = P;
    
    for i = 1:I
        
        
        for u = 1:U
           
            % only update KNOWN entries
            if Rs(i,u) == 0
                continue
            end
            
            nBi(i) = nBi(i) + gamma * (E(i,u) - lambda * nBi(i));
            nQ(:,i) = nQ(:,i) + gamma * (E(i,u) * P(:,u) - lambda * nQ(:,i));
            nP(:,u) = nP(:,u) + gamma * (E(i,u) * Q(:,i) - lambda * nP(:,u));
        end
    end

end


% function e = reg_sq_error(Bu, Bi, Q, P, Rp, Rs, lambda)
%     I = size(Q,2);
%     U = size(P,2);
% 
%     diff_matrix = (Rp - Rs) .^ 2;
%     lhs = sum(diff_matrix(:));
% 
%     % TODO might have mixed up *U and *I !!!
%     Q_norm_sum = sum(Q(:) .^ 2) * U;
%     P_norm_sum = sum(P(:) .^ 2) * I;
%     Bi_sum = sum(Bi .^ 2) * U;
%     Bu_sum = sum(Bu .^ 2) * I;
%     rhs = lambda * (Q_norm_sum + P_norm_sum + Bi_sum + Bu_sum);
% 
%     e = lhs + rhs;
% end

