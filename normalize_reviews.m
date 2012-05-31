function [ normalized_reviews ] = normalize_reviews( reviews )
%NORMALIZE_REVIEWS Summary of this function goes here
%   Detailed explanation goes here
    
    % reviews: [ user ; course ; course_prof ; rating ; workload_rating ]
    
    % average by course_prof and course
    % remove course_prof bias from everyone, so it's as if everyone that
    % took course X took it with the same prof! in this way we can act like
    % each course is the same regardless of who taught it, which means
    % DENSER DATA
    
    num_courses = max(reviews(2,:));
    num_course_profs = max(reviews(3,:));
    
    % compute averages
    
    course_avgs = zeros(1, num_courses);
    course_counts = course_avgs;
    course_prof_avgs = zeros(1, num_course_profs);
    course_prof_counts = course_prof_avgs;
    
    for i=1:size(reviews,2)
        
       col = reviews(:,i);
       course = col(2);
       course_prof = col(3);
       rating = col(4);
       
       course_avgs(course) = course_avgs(course) + rating;
       course_counts(course) = course_counts(course) + 1;
       
       course_prof_avgs(course_prof) = course_prof_avgs(course_prof) + rating;
       course_prof_counts(course_prof) = course_prof_counts(course_prof) + 1;
    end
    
    course_avgs = course_avgs ./ course_counts;
    course_prof_avgs = course_prof_avgs ./ course_prof_counts;
    
    % use averages to normalize reviews
    
    normalized_reviews = reviews;
    
    for i=1:size(reviews, 2)
       col = reviews(:,i);
       course = col(2);
       course_prof = col(3);
       rating = col(4);
    
       normalized_reviews(4,i) = rating - (course_prof_avgs(course_prof) - course_avgs(course));
       normalized_reviews(4,i) = min(5, max(1, normalized_reviews(4,i)));
    end
    
end

