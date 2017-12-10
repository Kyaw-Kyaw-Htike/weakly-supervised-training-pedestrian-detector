% Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

dir_save = 'weak_GT';
dir_images = 'D:/Research/Datasets/MIT_Traffic/GT/frames_train';

% So that I know which pedestrians to give weak labels for
dir_gt_bbox = 'D:/Research/Datasets/MIT_Traffic/GT/bboxes_train'; 

[fnames_full, fnames] = dir_imgnames(dir_images);
nimg = length(fnames);

mkdir2(dir_save);

nlabels_total = 0;

% to save the difference between labelled weak supervisions and
% true weak supervisions (computed from bbox GT/supervision)
centroids_diff = []; 

fprintf('Labelling started. Timer started.\n');
fprintf('==================\n');
tic;

% nimg = 3;

for i=1:nimg
    fprintf('Processing image: %s.\n', fnames{i});
    imshow(imread(fnames_full{i}));
    
    S = load(sprintf('%s/%s.mat', dir_gt_bbox, fnames{i}));
    
    hold on;
    
    draw_rect(S.dr);
    
    centroids = [];    
    while 1
        [x, y, button] = ginput(1);
        if button == 32, break; end % space bar to go to next image
        centroids = [centroids; x, y];
        plot(x, y, 'bx', 'MarkerSize', 30, 'LineWidth', 4);
    end
    hold off;
    
    nlabels = rrr(centroids);
    nlabels_total = nlabels_total + nlabels;
    fprintf('# labels in this frame = %d.\n Total # labels so far = %d.\n', nlabels, nlabels_total);
    
    % true centroids (from bbox gt)
    % due to the difference in the order of labelling, I need to align them
    % however, this alignment is just a hack and may be erroneous if the
    % pedestrians are almost overlapping to each other
    centroids_true = dr2centroid(S.dr);  
    idx_knn = knnsearch(centroids_true, centroids);
    centroids_true = centroids_true(idx_knn, :);
    centroids_diff = [ centroids_diff; (centroids - centroids_true) ];
    
%     if nlabels > 0
%         save(sprintf('%s/%s.mat', dir_save, fnames{i}), 'centroids');
%     end

    fprintf('Pausing. Please Ctrl+break to break..\n');
    pause;
    fprintf('Continuing..\n');
    
    fprintf('==================\n');
    
end

timeTaken = toc;
fprintf('To label %d frames [%d objects], it took %.2f secs [%.2f mins].\n', nimg, nlabels_total, timeTaken, timeTaken / 60);

disp(cov(centroids_diff))
