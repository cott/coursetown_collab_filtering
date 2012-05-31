
%   f:  # latent variables
%   mu: constant
%   Bu: U item vector. User baseline.
%   Bi: I item vector. Item baseline.
%   Q:  fxI matrix. row i = item i's affinity
%   P:  fxU matrix. row u = user u's affinity
%   Rp: IxU matrix. Predicted ratings.

% RSVD algorithm s.t. Rp = P * Q'
% rather than offset  Rp = P * Q' + B

% this implementation is intended to match that outlined in the Koren &
% Bell paper ? learning both the factor matrices AND the baselines.

function Rp = rsvd_incremental(f, reg, reg_b, lrate, max_iter, Rtrain, min_value, max_value, noise)
    
    avg = mean(Rtrain(Rtrain > 0));
    
    [BU, BI, U, I] = bootstrap(Rtrain, f, min_value, noise, 2);
    
    nonzeros = find(Rtrain);
    
    tis = Rtrain > 0;
    
%     errs = 1:max_iter;
    
%     lrate_scaling_factor = nthroot(.01, max_iter);
%     lrate_scaling_factor = .93;
    lrate_scaling_factor = (.93 + nthroot(.01, max_iter)) / 2;
    
    % main loop
    for i = 1:max_iter
        
%         Rp = predict_ratings(U, I, BU, BI, avg, min_value, max_value);
%         errs(i) = rms_error(Rp, Rtrain, tis);
         
        [U, I, BU, BI] = update(U, I, BU, BI, avg, Rtrain, lrate, reg, reg_b, min_value, max_value);
        lrate = lrate * lrate_scaling_factor; % decrease step size        
% 
%         % warn if matrices are blowing up
%         ustat = mean(mean(abs(U)));
%         istat = mean(mean(abs(I)));
%         if istat > max_value || ustat > max_value
%             display('uh oh. our matrices are getting out of hand. expect the error to EXPLODE!');
%         end
    end
    
%     errs
    
    % predict all ratings for output
    Rp = predict_ratings(U, I, BU, BI, avg, min_value, max_value);
end

% WHY!?!? WHY do the initial values matter so much? Why are Bu and Bi
% always positive? Shouldn't they reflect the deviation from the mean (mu)?

function [BU, BI, U, I] = bootstrap(Rs, f, min_value, noise, shrink_coeff)
    [i, u] = size(Rs);
        
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
    
    BU = zeros(1,u);
    BI = zeros(1,i);
end

% predicts entire rating matrix
function Rp = predict_ratings(U, I, BU, BI, avg, min_value, max_value)
    Rp = I' * U;
    
    for u=1:size(U,2)
       Rp(:,u) = Rp(:,u) + BI' + BU(u) + avg;
    end
    
    Rp = clip(Rp, min_value, max_value);
end

function r = clip(r, min_value, max_value)
    r = min(max_value, max(min_value, r));
end

% predicts a column of ratings
function r = predict_rating(u, i, U, I, BU, BI, avg, min_value, max_value)
    r = U(:,u)' * I + BU(u) + BI + avg;
    r = clip(r, min_value, max_value);
end

function r = predict_single_rating(u, i, U, I, BU, BI, avg, min_value, max_value)
    r = U(:,u)' * I(:,i) + BU(u) + BI(i) + avg;
    r = clip(r, min_value, max_value);
end

% iterative update
function [U, I, BU, BI] = update(U, I, BU, BI, avg, Rs, lrate, reg, reg_b, min_value, max_value)
    num_i = size(I,2);
    num_u = size(U,2);
% 
%     backup_U = U;
%     backup_I = I;
%     backup_BU = BU;
%     backup_BI = BI;
%     
    b_reg_term = 1 - lrate * reg_b;
    reg_term = 1 - lrate * reg;

    % only update using KNOWN entries (nonzero in Rs)
    nonzero_is = find(Rs > 0);
    for j = 1:length(nonzero_is)
        v = nonzero_is(j);
        i = mod(v-1, num_i) + 1;
        u = ceil(v / num_i);
        e = Rs(i,u) - predict_single_rating(u, i, U, I, BU, BI, avg, min_value, max_value);
%         
%         old_bs = [BI(i), BU(u)];
%         old_os = [U(:,u),I(:,i)];

        BI(i) = b_reg_term * BI(i) + lrate * e;
        BU(u) = b_reg_term * BU(u) + lrate * e;
        u_col = U(:,u);
        U(:,u) = reg_term * U(:,u) + e * lrate * I(:,i);
        I(:,i) = reg_term * I(:,i) + e * lrate * u_col;
        
%         new_bs = [BI(i), BU(u)];
%         new_os = [U(:,u),I(:,i)];
%         
%         diff_bs = new_bs - old_bs
%         diff_os = new_os' - old_os'
    end
% 
%     old_I = I;
%     old_U = U;
%     old_BU = BU;
%     old_BI = BI;
%     
%     U = backup_U;
%     I = backup_I;
%     BU = backup_BU;
%     BI = backup_BI;

%     % only update using KNOWN entries (nonzero in Rs)
%     for u = 1:num_u
%         e = Rs(:,u) - predict_rating(u, i, U, I, BU, BI, avg, min_value, max_value)';
%         nonzeros_logical = Rs(:,u) > 0;
%         nonzeros = find(nonzeros_logical);
%         num_nonzeros = length(nonzeros);
%         
%         if (num_nonzeros == 0)
%             continue;
%         end
% 
%         esum = e' * nonzeros_logical;
%         
%         BU(u) = ((1 - lrate * reg_b) ^ num_nonzeros) * BU(u) + lrate * esum;
%         BI = (1 - lrate * reg_b * nonzeros_logical') .* BI + lrate * e';
%                 
%         u_col = U(:,u);
%         U(:,u) = ((1 - lrate * reg) ^ num_nonzeros) * U(:,u) + I * (nonzeros_logical .* e) * lrate;
%         
%         I(:,nonzeros) = (1 - lrate * reg) * I(:,nonzeros) + lrate * u_col * e(nonzeros)';
%     end
% 
%     % INCOMPLETE
%     % only update using KNOWN entries (nonzero in Rs)
%     for u = 1:num_u
%         e = Rs(:,u) - predict_rating(u, i, U, I, BU, BI, avg, min_value, max_value)';
%         nonzeros_logical = Rs(:,u) > 0;
%         nonzeros = find(nonzeros_logical);
%         num_nonzeros = length(nonzeros);
%         
%         if (num_nonzeros == 0)
%             continue;
%         end
% 
%         esum = e' * nonzeros_logical;
%         
%         dU = I * (nonzeros_logical .* e) - reg * U(:,u);
%         dI = I .* 
%         
%         BU(u) = ((1 - lrate * reg_b) ^ num_nonzeros) * BU(u) + lrate * esum;
%         BI = (1 - lrate * reg_b * nonzeros_logical') .* BI + lrate * e';
%                 
%         u_col = U(:,u);
%         U(:,u) = ((1 - lrate * reg) ^ num_nonzeros) * U(:,u) + I * (nonzeros_logical .* e) * lrate;
%         
%         I(:,nonzeros) = (1 - lrate * reg) * I(:,nonzeros) + lrate * u_col * e(nonzeros)';
%     end
    
%     dU = abs(old_U - U);

end

