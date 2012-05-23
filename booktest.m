%% INIT DATA
% generate sparse matrix from input data

% just pick the first N points
% NOTE: convert all to double so sparse matrix doesn't complain
% data = double(int0(:,1:20000));
data = int0(:,1:20000);

% num_nonzeros = sum(data(3,:) > 0);
nonzero_rows = data(3,:) > 0;
length(find(nonzero_rows))
data = data(:,nonzero_rows);

% data format: userid, bookid, rating
% want: Rs = IxU matrix

num_users = length(unique(data(1,:)));
num_books = length(unique(data(2,:)));

% note: max(user) and max(books) are both smaller than 10k

collapse_factor = 1;

R = zeros(round(num_books / collapse_factor), num_users);
for j = 1:size(data,2)

    col = data(:,j);
    % note: matlab doesn't like indexing from 0
    R(round(col(2) / collapse_factor + 1), col(1)+1) = col(3);
    
end

% R = sparse(data(1,:) + 1,data(2,:) + 1,data(3,:));

%% PREP DATA

k = 0.10;
[train, test_indices, mu] = prep_data(R, k);

%% INIT PARAMS

gamma = 0.005;
lambda = 0.02;
max_iter = 10;
max_value = 10;

%% RUN IT

R_svd = collab_svd(f, lambda, gamma, max_iter, max_value, train, mu);

%%
R_base = baseline(train);

'R_svd error:'
rms_error(R_svd,R,test_indices)
mae_error(R_svd,R,test_indices)
'R_base error:'
rms_error(R_base,R,test_indices)
mae_error(R_base,R,test_indices)

% check if it moves in the right/wrong direction
both_up = (R_base > R) .* (R_base >= R_svd);
both_down = (R_base < R) .* (R_base <= R_svd);
right_side = sum(both_up(R > 0)) + sum(both_down(R > 0))
nonzero = R > 0;
total = sum(nonzero(:))
% NOTEWORTHY: svd does way better than neighbor-based approach from
% published paper here. NOTE: condensed the dataset though, so it's easier


