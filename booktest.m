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

R = zeros(num_books, num_users);
for j = 1:size(data,2)

    col = data(:,j);
    % note: matlab doesn't like indexing from 0
    R(col(2)+1, col(1)+1) = col(3);
    
end

% R = sparse(data(1,:) + 1,data(2,:) + 1,data(3,:));

%% PREP DATA

[train, test_indices, mu] = prep_data(R, k);

%% INIT PARAMS

gamma = 0.005;
lambda = 0.02;
max_iter = 10;
k = 10;
max_value = 10;

%% RUN IT

collab_svd(f, lambda, gamma, max_iter, max_value, R, train, test_indices, mu)