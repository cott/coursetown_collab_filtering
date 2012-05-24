%% INIT DATA
% generate sparse matrix from input data

% just pick the first N points
% NOTE: convert all to double so sparse matrix doesn't complain
% data = double(int0(:,1:20000));
data = int0(:,1:2000);

% num_nonzeros = sum(data(3,:) > 0);
nonzero_rows = data(3,:) > 0;
length(find(nonzero_rows));
data = data(:,nonzero_rows);

% data format: userid, bookid, rating
% want: Rs = IxU matrix

% num_users = length(unique(data(1,:)));
% num_books = length(unique(data(2,:)));

num_users = max(data(1,:));
num_books = max(data(2,:));

% note: max(user) and max(books) are both smaller than 10k

collapse_factor = 1;

R = zeros(ceil(num_books / collapse_factor + 1), num_users);

for j = 1:size(data,2)

    col = data(:,j);
    % note: matlab doesn't like indexing from 0
    R(floor(col(2) / collapse_factor + 1), col(1)+1) = col(3);
    
end

%% PREP THE LOOP

k = 10;

svd_err = zeros(k,2); % row 1: rms, row 2: mae
base_err = zeros(k,2);

% permute the indices before partitioning for cross-validation
nonzero_indices = find(R(:));
i_count = length(nonzero_indices);
indices = nonzero_indices(randperm(i_count));

%% PREP DATA

gamma = 0.007;
lambda = 0.1;
max_iter = 30;
max_value = 10;
f = 10;

%% RUN THE LOOP

display(sprintf('starting %d-fold cross-validation.',k));

i = 6;
% for gamma = [0.001, 0.005, 0.007, 0.01, 0.02, 0.05];
for lambda = [0.001, 0.003, 0.009, 0.02, 0.06, 0.1, 0.2, 0.3, 0.5, 1, 2, 4, 8];

% loop for k-fold cross-validation
% for i = 1:k
    %% prep the loop
    
    display(sprintf('%f',lambda));
    
    train = R;
    low = 1 + floor((i_count * (i-1))/k);
    hi = 1 + floor((i_count * i)/k);
    test_indices = indices(low:hi);
    train(test_indices) = 0; % zero the test indices
    
    %% run & evaluate
    
%     R_svd = collab_svd(f, lambda, gamma, max_iter, max_value, train);   
    R_base = baseline(train);
    
    svd_err(i,1) = rms_error(R_svd, R, test_indices);
    svd_err(i,2) = mae_error(R_svd, R, test_indices);
    base_err(i,1) = rms_error(R_base, R, test_indices);
    base_err(i,2) = mae_error(R_base, R, test_indices);
    
    display(sprintf('loop %d. SVD {rms: %f, mae: %f}, BASE {rms: %f, mae: %f}',i,svd_err(i,1),svd_err(i,2),base_err(i,1),base_err(i,2)));
    
end

%% run svd++

train = R;
low = 1 + floor((i_count * (i-1))/k);
hi = 1 + floor((i_count * i)/k);
test_indices = indices(low:hi);
train(test_indices) = 0; % zero the test indices

%%
R_svd_plus = collab_svd_plus(f, 0.005, 0.015, 0.007, max_iter, max_value, train);
R_svd = collab_svd(f, lambda, gamma, max_iter, max_value, train);   
R_base = baseline(train);

svd_err(i,1) = rms_error(R_svd, R, test_indices);
svd_err(i,2) = mae_error(R_svd, R, test_indices);
base_err(i,1) = rms_error(R_base, R, test_indices);
base_err(i,2) = mae_error(R_base, R, test_indices);

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
right_side = sum(both_up(R > 0)) + sum(both_down(R > 0));
nonzero = R > 0;
total = sum(nonzero(:));
percent_right_side = right_side / total
% NOTEWORTHY: svd does way better than neighbor-based approach from
% published paper here. NOTE: condensed the dataset though, so it's easier


