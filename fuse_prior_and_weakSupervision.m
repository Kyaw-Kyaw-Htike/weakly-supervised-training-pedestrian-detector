%{
Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

Given weak supervisions and the pedestrian prior, fuse these two cues to
generate strong (bounding box) supervisions.
%}

%% hyperparameters
winsize = [128, 64];
func_feat_extract = @hog_dollar_new;
img_scale = 6;

min_width = winsize(2)/img_scale;
max_width = min_width * 4;
min_height = min_width * 2;
max_height = min_height * 4;

% %% Visualize some bivariate distributions
% imshow(img); hold on; 
% while 1
%     mu = ginput(1);
%     r = mvnrnd(mu,[3 0; 0 3],100);
%     plot(r(:,1),r(:,2),'+');
%     drawnow;
% end
% hold off;

%% input
dir_weakSupervision = 'weak_GT_simulated';
dir_images = 'D:/Research/Datasets/MIT_Traffic/GT/frames_train';
fnames = dir([dir_weakSupervision, '/*.mat']);

%% output
dir_output = 'strong_GT_gen';
dir_output_noSpatialPenalty = 'strong_GT_gen_noSpatialPenalty';
dir_output_patches = 'patches_strong_GT_gen';
dir_output_patches_noSpatialPenalty = 'patches_strong_GT_gen_noSpatialPenalty';

% be careful!
mkdir2(dir_output);
mkdir2(dir_output_noSpatialPenalty);
mkdir2(dir_output_patches);
mkdir2(dir_output_patches_noSpatialPenalty);

%% load prior detector
load('classifier_prior.mat'); % load w_lin

%% fusion
rehash
counter = 0;
tic;
for i=1:length(fnames)
    
    print_progress(i, length(fnames));
    S = load(fullfile(dir_weakSupervision, fnames(i).name));
    imgname = remove_mat_ext( fnames(i).name);
    img = imread(fullfile(dir_images, imgname));
    img_single = im2single(img);
    centroids = S.centroids;
    ncen = rrr(centroids);  
    
    dr_from_centroids = zeros(ncen, 4);
    dr_from_centroids_noSpatialPenalty = zeros(ncen, 4);
    dr_from_centroids_noPedPrior = zeros(ncen, 4);
    roiOuter = zeros(ncen, 4);
    
    for j=1:ncen 
        
        centroid_cur = centroids(j, :);
        
        % get the surrounding ROI fence for the current centroid supervision
        roi_outer = widthHeightCentroid_To_rect(max_width, max_height, centroid_cur);   
        roiOuter(j, :) = roi_outer;
                
        % run sliding window detector in the outer_roi
        [~, ~, r1, r2, c1, c2] = roi_img_rc(img, roi_outer);
        if r1<0 || c1<0 || r2>rrr(img) || c2>ccc(img) % outer roi protruding image boundaries
            dr = [1,1,1,1];
            ds = 1;
        else            
            [dr, ds] = slidewin_detect_linear_roi2(img_single, w_lin, true, true, roi_outer, img_scale); 
        end
                
%         %%%%%%%%%%%%% Generate from the prior object distribution and score using distance to centroid %%%%%%%%%
%         nsamples = 100;
%         idx_samples = randsample(rrr(dr), nsamples, true, ds); % according to unsupervised learnt prior distribution
% %         idx_samples = randsample(rrr(dr), nsamples, true); % just uniform prior
%         dr_prior = dr(idx_samples, :);             
%         c = dr2centroid(dr_prior); % centroids corresponding to dr_prior
%         % d = pdist2(c, centroid_cur); % distance of each dr_prior's centroid to centroid_cur
%         % w = 1 ./ (1 + d); % convert to similarity
%         w = mvnpdf(c, centroid_cur, [5 0; 0 5]); % higher weight to ones close to groundtruth centroid
%         w = w / sum(w); % make sum to one
%         
%         dr_prior = dr;
%         w = ds / sum(ds);
%         
%         dr = dr_prior;
%         ds = w;        
        
        %%%%%%%%%%%%%
        
%         %%%%%  Bayesian algorithm %%%%%%%%       
%         % define number of samples
%         nsamples = 1000;        
%         
%         % define prior dr distributions
%         centroids_sampled = mvnrnd(centroid_cur, [3 0; 0 3], nsamples);
%         widths_sampled = randi([floor(min_width), ceil(max_width)], [nsamples, 1]);
%         heights_sampled = widths_sampled * 2;
%         dr = widthHeightCentroid_To_rect(widths_sampled, heights_sampled, centroids_sampled);
%         
%         if 0
%             % save patches of sampled dr
%             folder_name = sprintf('proposed_patches/frame%05d_ped%02d', i, j);
%             mkdir(folder_name);        
%             for jj=1:nsamples
%                 r = roi_img(img, dr(jj, :));
%                 imwrite(r, sprintf('%s/%04d.jpg', folder_name, jj));
%             end        
%         end
%         
%         % extract features from the sampled dr (prior)
%         feats_cur = slidewin_get_feats(img_single, dr, func_feat_extract, winsize);
%         
%         % score each prior sample to get likelihood
%         % ds = [feats_cur, ones(nsamples, 1)] * liblinear_model.w(:);
%         ds = logsig_transfer( [feats_cur, ones(nsamples, 1)] * liblinear_model.w(:) );
% 
%         
%        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        %%%%%% Fuse the two cues %%%%        
        
        c = dr2centroid(dr);
%         d = pdist2(c, centroid_cur);
        p = mvnpdf(c, centroid_cur, [3 0; 0 3]); 
%         d = 1 ./ (d + 1); % convert distance to similarity
%         s1 = normalise_dataset_01(ds);
%         s2 = normalise_dataset_01(d);
%         s2 = d;
%         s1 = p;
%         s2 = ds;

        s1 = ds;
        s2 = p;
        
        s1 = s1 / sum(s1);
        s2 = s2 / sum(s2);    
        ds = mean([s1, s2], 2); 
        ds_noSpatialPenalty = s1;
        ds_noPedPrior = s2;
        %%%%%%%%%%%%%%%%%%%%
        
        % get the posterior score: take max score
        [ds_max, idx_max] = max(ds);
        dr_from_centroids(j, :) =  dr(idx_max, :);      
        
        [ds_max, idx_max] = max(ds_noSpatialPenalty);
        dr_from_centroids_noSpatialPenalty(j, :) =  dr(idx_max, :); 
        
        [ds_max, idx_max] = max(ds_noPedPrior);
        dr_from_centroids_noPedPrior(j, :) =  dr(idx_max, :);  
                
        % get the posterior score: take mean score
        % ds = ds - min(ds);
        % if sum(ds>0)>0, ds(ds<0)=0; end
        % ds = normalise_dataset_01(ds);       
        % ds = ds / sum(ds);
        % dr_from_centroids(j, :) =  wmean(dr, repmat(ds, 1, 4));       
        
%         % get the posterior score: take mean score of the top N
%         tN = 3; % top N
%         [ds_top, idx_sort] = sort(ds, 'descend');       
%         dr_top = dr(idx_sort, :);
%         ds_top = ds_top(1:tN);
%         dr_top = dr_top(1:tN, :);
%         ds_top = normalise_dataset_01(ds_top);        
%         dr_from_centroids(j, :) =  wmean(dr_top, repmat(ds_top, 1, 4));       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
%         if ds_max < 0.5, isDiscarded(j) = true; end
    end    
    
%     dr_from_centroids = remove_padding_dalal(dr_from_centroids);
%     dr_from_centroids = add_padding_dalal(dr_from_centroids);    %%%%%%%%%% NOTE THIS %%%%%%%%%%%%%%
        
    %%%%%%%%%%%%%%% Visualization of the resulting strong supervision %%%%%%%%%%%%%%
    if 0
        imshow(img);
        hold on;
        draw_centroid(centroids);
    
        draw_rect(dr_from_centroids_noSpatialPenalty, 'red', 3);
        draw_rect(dr_from_centroids_noPedPrior, 'cyan', 3);
        draw_rect(dr_from_centroids, 'blue', 3);

        draw_rect(roiOuter, 'magenta', 3);
        title('Red: No spatial penalty; Cyan: No Ped Prior; Blue: Fusion of ped prior and spatial penalty.');
        hold off;    
        pause;    
    end
    
    %%%%%%%%%%%%%%% Saving the resulting strong supervision %%%%%%%%%%%%%%
    if 1
        % save dr
        dr = round(dr_from_centroids);
        save(sprintf('%s/%s', dir_output, fnames(i).name), 'dr', 'ds');
        
        dr = round(dr_from_centroids_noSpatialPenalty);
        save(sprintf('%s/%s', dir_output_noSpatialPenalty, fnames(i).name), 'dr', 'ds');
        
        patches = bbApply('crop', img, dr_from_centroids);
        patches_noSpatialPenalty = bbApply('crop', img, dr_from_centroids_noSpatialPenalty);
                       
        % save patches
        for ii = 1:rrr(dr)
            counter = counter + 1;
            imwrite(patches{ii}, sprintf('%s/%05d.jpg', dir_output_patches, counter));
            imwrite(patches_noSpatialPenalty{ii}, sprintf('%s/%05d.jpg', dir_output_patches_noSpatialPenalty, counter));
        end        
    end    
    
end
toc


%% compute the bonding box overlap cover
res_dir = 'strong_GT_gen'; 
% res_dir = 'strong_GT_gen_noSpatialPenalty'; 
gt_dir  = 'D:/Research/Datasets/MIT_Traffic/GT/bboxes_train';

fnames = dir(sprintf('%s/*.mat', res_dir));
ovps = [];

for i=1:length(fnames)
    S1 = load(fullfile(res_dir, fnames(i).name)); % results to evaluate
    S2 = load(fullfile(gt_dir, fnames(i).name)); % ground truth
    assert(rrr(S1.dr)==rrr(S2.dr));
    if rrr(S1.dr)==0, continue; end
    t = overlap_percent_rects(S1.dr, S2.dr);
    t = max(t, [], 2);
    ovps = [ovps; t];    
end

median_overlap_score = median(ovps)

%% Make a histogram
figure, bar([], 'EdgeColor', [0.1,0,0],'LineWidth',1);
xlabel('Datasets');
ylabel('Median overlap score');
set(gca,'XTickLabel', {'MIT Traffic', 'CUHK Square'}, 'TickLabelInterpreter','none','XTickLabelRotation', 45);
legend('Fused system', 'Ped Prior only (No spatial penalty)', 'Location', 'best');
export_fig('figs/bar_overlapScores', '-pdf', '-transparent');
close;

%% Make table 
mat_table = [];
mat_table = [mat_table; mean(mat_table, 1)];
fname = 'figs/table_scores.tex';
rowLabels = {'CUHK Square', 'MIT Traffic', 'Mean'};
colLabels = {'No spatial penalty', 'Fused System'};
matrix2latex(mat_table, fname, 'rowLabels', ...
    rowLabels, 'columnLabels', colLabels, ...
    'alignment', 'c', 'format', '%.4f'); 



