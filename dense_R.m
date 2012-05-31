function [ R ] = dense_R( data_raw, item_limit, user_limit, max_item_reviews, min_user_reviews )
%DENSE_R takes the input in the standard format and outputs an R matrix
%  with the most-reviewed items and users who have reviewed those items
%  many times (at least 3)

    num_items = max(data_raw(2,:)) + 1;
    num_users = max(data_raw(1,:)) + 1;

    % count items

    item_counts = zeros(1,num_items);
    for col_i=1:size(data_raw,2)
       col = data_raw(:,col_i);
       item = col(2) + 1;
       item_counts(item) = item_counts(item) + 1;
    end
    %
    % grab top [item_limit] items w/ between 1 and 50 reviews

    itemp = [1:num_items ; item_counts];
    itemp = itemp(:,itemp(2,:) > 1);
    itemp = itemp(:,itemp(2,:) <= max_item_reviews);
    [~, ix] = sort(itemp(2,:),2,'descend');
    ix = itemp(1,ix); % ix = ids of top users
    
    item_limit = min(item_limit, numel(ix));

    % item_map(id) = 0 if we're not keeping that item, [new id] if we are
    item_map = zeros(1,numel(item_counts));
    item_map(ix(1:item_limit)) = 1:item_limit; % keep [limit] items w/ the most reviews

    % count users (for the items we now care about)

    user_counts = zeros(1, num_users);
    for col_i=1:size(data_raw,2)
       col = data_raw(:,col_i);
       user = col(1) + 1;
       item = col(2) + 1;

       if item_map(item) == 0
           continue;
       end

       user_counts(user) = user_counts(user) + 1;
    end

    % grab random [user_limit] users w/ at least 5 reviews

    itemp = [1:num_users ; user_counts];
    itemp = itemp(:,itemp(2,:) >= min_user_reviews);
    itemp = itemp(:,randperm(size(itemp,2)));
    ix = itemp(1,:);

    final_user_count = min(numel(ix), user_limit);
    user_map = zeros(1, numel(user_counts));
    user_map(ix(1:final_user_count)) = 1:final_user_count;

    % create R using this filtered data

    num_ditched = 0;
    R = zeros(item_limit, final_user_count);
    for col_i=1:size(data_raw,2)
       col = data_raw(:,col_i);
       user = col(1) + 1;
       item = col(2) + 1;
       new_user = user_map(user);
       new_item = item_map(item);
       if new_item == 0 || new_user == 0
           continue
       end

       R(new_item, new_user) = col(3);
    end
end

