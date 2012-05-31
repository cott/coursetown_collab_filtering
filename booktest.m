%% pick dataset

% int0: books
data_raw = int0;
max_value = 10;

%% use movielens dataset

data_raw = movielens(:,[1 2 3])';
max_value = 5;

%% my course_review data

data_raw = reviews(:,[1 2 4])';
max_value = 5;

%% normalize my course_review data
data_raw = normalize_reviews(reviews');
data_raw = data_raw([1 2 4], :);
max_value = 5;

%% make a super dense dataset!

limit = 4000;
user_limit = limit;
item_limit = limit / 2;

R = dense_R(data_raw, item_limit, user_limit, 100, 5);
display(sprintf('%d items, %d users. avg %f reviews per item, %f per user', size(R,1), size(R,2), mean(sum(R > 0,2)), mean(sum(R > 0,1))));

%% INIT DATA
% generate sparse matrix from input data

% just pick the first N points
% NOTE: convert all to double so sparse matrix doesn't complain
% data = double(int0(:,1:20000));
limit = 4000;
user_limit = limit;
item_limit = limit / 2;

under_limit = (data_raw(2,:) < item_limit) & (data_raw(1,:) < user_limit);
data = data_raw(:,under_limit);

num_results = size(data,2);
collision_ratio = num_results / limit;


display(sprintf('found %d results w/ limit %d. collision ratio: %f', num_results, limit, collision_ratio));

% num_nonzeros = sum(data(3,:) > 0);
nonzero_rows = data(3,:) ~= 0;
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

I = ceil(num_books / collapse_factor + 1);
U = num_users + 1;

R = zeros(I, U);

for j = 1:size(data,2)
    col = data(:,j);
    
    row_i = floor(col(2) / collapse_factor + 1); % item
    col_i = col(1) + 1; % user

    % note: matlab doesn't like indexing from 0
    R(row_i, col_i) = col(3);
    
end

display(sprintf('avg reviews per item: %f, avg reviews per user: %f', mean(sum(R > 0,2)), mean(sum(R > 0,1))));

%%

% 
% P = [1 1 2 2 3 3 4 4; 1 2 1 2 1 2 1 2];
% Q = [1 1 1 1 1 1 1 1 1; 0 1 0 1 0 1 0 1 0];
% 
% R = P' * Q;
% max_value = 6;
% f = 2;
% 
% indices = 1:numel(R);

%% PREP THE LOOP

k = 10; % for k-fold cross-validation

svd_err = zeros(k,2); % row 1: rms, row 2: mae
base_err = zeros(k,2);

% permute the indices before partitioning for cross-validation
nonzero_indices = find(R(:));
i_count = length(nonzero_indices);
indices = nonzero_indices(randperm(i_count));

%% PREP DATA

gamma = 0.007;
lambda = 0.1;
max_iter = 10;
f = 20; % number of latent dimensions
i = 1;

%% RUN THE LOOP

display(sprintf('starting %d-fold cross-validation.',k));

% for gamma = [0.001, 0.005, 0.007, 0.01, 0.02, 0.05];
for lambda = [0.0005, 0.001, 0.003, 0.009, 0.02, 0.06, 0.1, 0.2, 0.3, 0.5, 1, 2];

% loop for k-fold cross-validation
% for i = 1:k
    %% prep the loop
       
    train = R;
    low = 1 + floor((i_count * (i-1))/k);
    hi = 1 + floor((i_count * i)/k);
    test_indices = indices(low:hi);
    train(test_indices) = 0; % zero the test indices
       
    %% run & evaluate
    
    R_svd = collab_svd(f, lambda, gamma, max_iter, max_value, train, 0.25, 3);   
    R_base = baseline(train, 3);
    
    svd_err(i,1) = rms_error(R_svd, R, test_indices);
    svd_err(i,2) = mae_error(R_svd, R, test_indices);
    base_err(i,1) = rms_error(R_base, R, test_indices);
    base_err(i,2) = mae_error(R_base, R, test_indices);
    
    display(sprintf('lambda %d. SVD {rms: %f, mae: %f}, BASE {rms: %f, mae: %f}',lambda,svd_err(i,1),svd_err(i,2),base_err(i,1),base_err(i,2)));
    
end



%%

i = 5;
mu = mean(train(train ~= 0));
smu = sqrt(mu);
[I,U] = size(train);
noise = 0.5;
Q = rand(f,I) .* noise - noise/2;
P = rand(f,U) .* noise - noise/2;

% for lambda = [0.001, 0.003, 0.009, 0.02, 0.06, 0.1, 0.2, 0.3, 0.5, 1, 2, 4, 8];
lambda = 0.02;
gamma = 0.001;

max_iter = 30;

momentum = 0.4; % adding a momentum allows you to decrease the learning rate (gamma)

%%

display(sprintf('starting loop. max_iter = %d. momentum = %d', max_iter, momentum));
% for max_iter = [2, 5, 10, 20, 40, 100]
% 

fake_test_is = find(train > 0);
fake_test_is(randperm(length(fake_test_is))) = fake_test_is;
fake_test_is = fake_test_is(1:100);

% for gamma = [0.002, 0.004, 0.01] % large gamma => divergence (papers agree)
%     for lambda = [0.0005, 0.001, 0.01, 0.1]
    %%
    
for gamma = [0.001, 0.005, 0.01] % large gamma => divergence (papers agree)
% for gamma = [0.005, 0.01, 0.02] % large gamma => divergence (papers agree)
    for lambda = [0.0001, 0.001, 0.01, 0.02]
    %%
    base = baseline(train, 2);
%     prepped_train = zeros(size(train,1),size(train,2));
%     prepped_train(train ~= 0) = train(train ~= 0) - base(train ~= 0);
%     R_svd = rsvd(f, lambda, gamma, max_iter, base, train, Q, P, max_value);
%     R_svd = rsvd_absolute(f, lambda, gamma, max_iter, train, 1, 10, bound);
%     R_svd = rsvd_momentum(f, lambda, gamma, max_iter, train, 1, 10, bound, momentum, 0.00005);
%     R_svd = rsvd_relative(f, lambda, gamma, max_iter, train, 1, 10, noise, momentum, -1);
    R_svd = rsvd_standard(f, lambda, lambda/2, gamma, max_iter, train, 1, max_value, noise, momentum, -1);
%     R_svd = rsvd_incremental(f, lambda, lambda*2, gamma, max_iter, train, 1, max_value, noise);
   
    
%     R_svd2 = real(R_svd) ; 
    
    svd_err(i,1) = rms_error(R_svd, R, test_indices);
    svd_err(i,2) = mae_error(R_svd, R, test_indices);
    base_err(i,1) = rms_error(base, R, test_indices);
    base_err(i,2) = mae_error(base, R, test_indices);
    
    R_err = rms_error(R_svd, train, fake_test_is);
  
%     svd_2_err = rms_error(R_svd2, R, test_indices);
%     svd_2_mae_err = mae_error(R_svd2, R, test_indices);
%     display(sprintf('loop %d. lambda %f. gamma %f. SVD {rms: %f, mae: %f}, SVD2 {rms: %f, mae: %f}, BASE {rms: %f, mae: %f}',i,lambda,gamma,svd_err(i,1),svd_err(i,2),svd_2_err, svd_2_mae_err,base_err(i,1),base_err(i,2)));   
    
    display(sprintf('loop %d. lambda %f. gamma %f. SVD {rms: %f, mae: %f}, BASE {rms: %f, mae: %f} (overfitting? %d)',i,lambda,gamma,svd_err(i,1),svd_err(i,2),base_err(i,1),base_err(i,2), R_err));

    end
end


%% incremental SVD
display(sprintf('starting loop. max_iter = %d', max_iter));
for gamma = [0.005, 0.01, 0.03] % large gamma => divergence (papers agree)
    for lambda = [0.001, 0.01, 0.02, 0.04, 0.1]
    %%
    base = baseline(train, 2);
%     R_svd = rsvd_standard(f, lambda, lambda/2, gamma, max_iter, train, 1, max_value, noise, momentum, -1);
    R_svd = rsvd_incremental(f, lambda, lambda/2, gamma, max_iter, train, 1, max_value, noise);
      
    svd_err(i,1) = rms_error(R_svd, R, test_indices);
    svd_err(i,2) = mae_error(R_svd, R, test_indices);
    base_err(i,1) = rms_error(base, R, test_indices);
    base_err(i,2) = mae_error(base, R, test_indices);
    
    R_err = rms_error(R_svd, train, fake_test_is);
    display(sprintf('loop %d. lambda %f. gamma %f. SVD {rms: %f, mae: %f}, BASE {rms: %f, mae: %f} (overfitting? %d)',i,lambda,gamma,svd_err(i,1),svd_err(i,2),base_err(i,1),base_err(i,2), R_err));

    end
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
R_base = baseline(train, 3);

svd_err(i,1) = rms_error(R_svd, R, test_indices);
svd_err(i,2) = mae_error(R_svd, R, test_indices);
base_err(i,1) = rms_error(R_base, R, test_indices);
base_err(i,2) = mae_error(R_base, R, test_indices);

%% RUN IT

R_svd = collab_svd(f, lambda, gamma, max_iter, max_value, train, mu);

%%
R_base = baseline(train, 3);

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


