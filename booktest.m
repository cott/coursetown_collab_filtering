%% INIT
% generate sparse matrix from input data

% just take 10k points
data = int0(:,1:20000);
num_nonzeros = sum(data(3,:) > 0)

% data format: userid, bookid, rating
% want: Rs = IxU matrix

num_users = length(unique(data(1,:)));
num_books = length(unique(data(2,:)));

% note: max(user) and max(books) are both smaller than 10k

R = zeros(num_books, num_users/5);
for j = 1:length(data)

    col = data(:,j);
    % note: matlab doesn't like indexing from 0
    R(col(2)/5+1, col(1)+1) = col(3);
    
end

%% RUN THE ALGORITHM!

gamma = 0.005;
lambda = 0.02;
max_iter = 2;
k = 10;

f = 10;

collab_svd(f, lambda, gamma, R', max_iter, k)