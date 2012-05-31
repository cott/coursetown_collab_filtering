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

clear 'data_raw'; % remove data from memory. maybe this is a bad idea?
clear 'data';
clear 'nonzero_rows';




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

noise = 0.4;

% params: lambda & gamma
% gammas = [.004 .008 .01];
gammas = [.008 .01 .02];
gammas = [0.2 0.3]
lambdas = [.00001 .0001 .001 .01 .1 .2];
errors = zeros(numel(gammas), numel(lambdas)); % IGNORES k right now
base_errors = zeros(1, numel(lambdas));
%%
display(sprintf('starting loop. k = %d, f = %d, max_iter = %d',k, f, max_iter));

% for gamma = [0.001, 0.005, 0.007, 0.01, 0.02, 0.05];
for i = 1:1
    %% prep the loop
       
    train = R;
    low = 1 + floor((i_count * (i-1))/k);
    hi = min(i_count, floor((i_count * i)/k));
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
        %%
        display(sprintf('gamma %f', gamma));
    for L = 1:numel(lambdas)
        %% run & evaluate
        
        lambda = lambdas(L);
        lambda_b = lambda * 2;
%%
        R_svd = rsvd_incremental(f, lambda, lambda_b, gamma, max_iter, train, 1, max_value, noise);

        rmse = rms_error(R_svd, R, test_indices);
        maee = mae_error(R_svd, R, test_indices);
        
        errors(G, L) = rmse;
%%
        display(sprintf('lambda %f. improvements: ( %f , %f ) -- SVD {rms: %f, mae: %f}',lambda,rmse_b - rmse, maee_b - maee, rmse,maee));
    end
    end
end

%%

file_counter = file_counter + 1;
spec_filename = sprintf('./run_data_%d.mat', file_counter );
filename = './run_data.mat';
% run 'load(filename);' to reload lost data

%%
berrors = base_errors;
rerrors = errors;

% backup to disk
save(filename, 'lambdas', 'gammas', 'berrors', 'rerrors');
save(spec_filename, 'lambdas', 'gammas', 'berrors', 'rerrors');

%%

baseline = berrors(1);


%% average errors along the k instances
avg_errors = mean(rerrors,1);
baseline = mean(berrors) * ones(size(lambdas));
plot(lambdas, [avg_errors; baseline]);

%% plot the results (raw!)

err_offset = rerrors - baseline;

plot(lambdas, rerrors);

%%
% calculate and plot avg offset (bigger is better)
e_offset = repmat(berrors', 1, size(rerrors,2)) - rerrors;
plot(lambdas, [e_offset; mean(e_offset,1)]);
xlabel('lambda (regularization term)');
ylabel('RMSE improvement');
