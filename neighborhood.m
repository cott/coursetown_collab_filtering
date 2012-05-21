classdef neighborhood
    %NEIGHBORHOOD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Q: []
        P: []
    end
    
    methods
        % INPUTS
        %   mu: constant
        %   Bu: U item vector. User baseline.
        %   Bi: I item vector. Item baseline.
        %   Q:  UxI matrix?
        %   P:  UxI matrix?
        % OUTPUTS
        %   R: UxI matrix.
        function R = predict_ratings(mu, Bu, Bi, Q, P)
            % r_ui = mu + b_i + b_u + q_i^T * p_u
            I = size(Bi)
            U = size(Bu)
            BuBlock = repmat(Bu,I,1)
            BiBlock = repmat(Bi',1,U)
            R = mu + BuBlock + BiBlock + Q' * P
        end
        
        
        function E = reg_sq_error(mu, Bu, Bi, Q, P, Rpred, Rsol, lambda4)
            diff_matrix = (Rpred - Rsol) .^ 2
            diff = sum(diff_matrix(:))
            
        end
    end
    
end

