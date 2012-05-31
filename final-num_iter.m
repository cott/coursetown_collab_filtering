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






%% PREP THE LOOP

k = 10; % for k-fold cross-validation

svd_err = zeros(k,2); % row 1: rms, row 2: mae
base_err = zeros(k,2);

% permute the indices before partitioning for cross-validation
nonzero_indices = find(R(:));
i_count = length(nonzero_indices);
indices = nonzero_indices(randperm(i_count));

f = 20; % number of latent dimensions
i = 1;

file_counter = 0;

%% RUN THE LOOP

% gamma = [0.007, 0.01, 0.014] % large gamma => divergence (papers agree)
% lambda = [0.04, 0.1, 0.2, 0.3]

max_iter = 50;


noise = 0.5;

 
% params: lambda & gamma
gammas = [.002 .004 .008 .01];
num_iters = [10 30 50 100 200];
lambda = 0.01;
errors = zeros(numel(gammas), numel(num_iters)); % IGNORES k right now
base_errors = zeros(1, k);
overfitting = zeros(numel(gammas), numel(num_iters));
%%
display(sprintf('starting loop. k = %d, f = %d, max_iter = %d',k, f, max_iter));

% for gamma = [0.001, 0.005, 0.007, 0.01, 0.02, 0.05];
for i = 1:1
    %% prep the loop
       
    train = R;
    low = 1 + floor((i_count * (i-1))/k);
    hi = min(i_count, floor((i_count * i)/k));
    nonzero_train_indices = find(train > 0);
    test_indices = indices(low:hi);
    train(test_indices) = 0; % zero the test indices
    R_base = baseline(train, 2);
    %%
    rmse_b = rms_error(R_base, R, test_indices);
    %%
    maee_b = mae_error(R_base, R, test_indices);
    base_errors(i) = rmse_b;
       
    display(sprintf('  -- i = %d, BASE { rms: %f, mae: %f }', i, rmse_b, maee_b));
    
    %%
    for G = 1:numel(gammas)
        gamma = gammas(G);
        display(sprintf('gamma %f', gamma));
    for L = 1:numel(num_iters)
        %% run & evaluate
        
        max_iter = num_iters(L);
        
        lambda_b = lambda;

        R_svd = rsvd_incremental(f, lambda, lambda_b, gamma, max_iter, train, 1, max_value, noise);

        rmse = rms_error(R_svd, R, test_indices);
        maee = mae_error(R_svd, R, test_indices);
        
        errors(G, L) = rmse;
        overfitting(G,L) = rms_error(R_svd, train, nonzero_train_indices);
        this_overfit = overfitting(G,L);
%%
        display(sprintf('%f iterations. improvements: ( %f , %f ), overfit: %f -- SVD {rms: %f, mae: %f}',max_iter,rmse_b - rmse, maee_b - maee, this_overfit, rmse,maee));
    end
    end
end

%%

file_counter = file_counter + 1;
spec_filename = sprintf('./run_max_iter_%d.mat', file_counter );
filename = './run_max_iter.mat';
% run 'load(filename);' to reload lost data

%%
baseline = berrors(1);

%%
berrors = base_errors;
rerrors = errors;

% backup to disk
save(filename, 'max_iters', 'gammas', 'berrors', 'rerrors');
save(spec_filename, 'max_iters', 'gammas', 'berrors', 'rerrors');

%% average errors along the k instances
avg_errors = mean(rerrors,1);
baseline = mean(berrors) * ones(size(max_iters));
plot(max_iters, [avg_errors; baseline]);


%%
% calculate and plot avg offset (bigger is better)
e_offset = repmat(berrors', 1, size(rerrors,2)) - rerrors;
plot(lambdas, [e_offset; mean(e_offset,1)]);
xlabel('lambda (regularization term)');
ylabel('RMSE improvement');
