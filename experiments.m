
%% find # users that rated two different items

U = size(R,1);
overlap = zeros(U);

for i=1:U
    for j=i:U
        overlap(i,j) = sum(R(i,:) & R(j,:));
        overlap(j,i) = overlap(i,j);
    end
end

max_overlap = max(overlap(:));
min_overlap = min(overlap(:));
avg_overlap = mean(overlap(:));

display(sprintf('max: %d, min: %d, avg: %f, zeros/nonzeros: %d/%d', max_overlap, min_overlap, avg_overlap, length(overlap(overlap == 0)), length(overlap(overlap > 0))));


%% find # users that rated two different items

I = size(R,2);
i_overlap = zeros(I);

for i=1:I
    for j=i:I
        i_overlap(i,j) = sum(R(:,i) & R(:,j));
        i_overlap(j,i) = i_overlap(i,j);
    end
end

min_overlap = min(i_overlap(:));
avg_overlap = mean(i_overlap(:));

display(sprintf('max: %d, min: %d, avg: %f, zeros/nonzeros: %d/%d', max(i_overlap(:)), min_overlap, avg_overlap, length(i_overlap(i_overlap == 0)), length(i_overlap(i_overlap > 0))));
