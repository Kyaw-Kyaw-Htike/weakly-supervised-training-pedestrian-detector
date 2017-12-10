%{
Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

Given foreground masks for each frame in the video (the first half, 
i.e. training data), train a pedestrian prior (in the form of a classifier)
by using HOG features extracted from foreground ROIs as positive samples
and samples from non-foreground areas as negative samples.

%}

% read MIT video 
vobj = VideoReader('D:/Research/Datasets/MIT_Traffic/vids/mv2.avi');

%% visualize min and max allowable rectangles to filter out 
% rectangles from bg sub (i.e. foreground detections) that are impossible 
% to be pedestrians.
% Therefore, these "filter" conditions can be considered as a type of 
% prior knowledge on pedestrians. 

width_min = 5;
width_max = 25;
height_min = 15;
height_max = 50;

% get a sample frame from video
img = vobj.read(1);
imshow(img);

% min rectangle
dr_min = [300, 250, width_min, height_min];
% max rectangle
dr_max = [500, 350, width_max, height_max];

draw_rect(dr_min, 'green', 1);
draw_rect(dr_max, 'red', 1);


%% Get and filter foreground detections and extract HOG features
dir_bgsub = 'D:/Research/Datasets/MIT_Traffic/BGsub/PBAS/FG';

nframes = vobj.NumberOfFrames;
nframes_proc = floor(nframes/2);

bool_visualize = false; % visualize foreground detections (after filtering)
bool_savefeat = true; % save HOG features from each of the generated ROIs
bool_savepatch = true; % save patches from each of the generated ROIs

dir_patches = 'patches';

if bool_savepatch, mkdir2(dir_patches); end
if bool_savefeat, delete('feats_FG.mat'); end
    
nsamples_save = 10000; % only to be concerned when bool_savefeat = true or when saving patches

if bool_savefeat    
    func_feat_extract = @hog_dollar_new;
    dim_feat = get_dim_feat(@hog_dollar_new, [128, 64]);
    feats = zeros(nsamples_save, dim_feat);
end

clear sobj; sobj = sample_reservoir(nsamples_save);

counter_obj = 0;
for i=2:nframes_proc
    
   if mod(i, 100) == 0, print_progress(i, nframes_proc); fprintf('# ROIs = %d.\n', counter_obj); end
   img = vobj.read(i);
   img_fg = imread(sprintf('%s/%d.png', dir_bgsub, i));
   imbw = img_fg > 100;
   
    % imbw = medfilt2(imbw, [3, 3]);
    imbw = imdilate(imbw,strel('square', 3));
    % imbw = imerode(imbw,strel('square', 3));
    stats = regionprops(imbw, 'BoundingBox' );
    dr = cell2mat(struct2cell(stats)');
%     dr(dr(:, 4) < 10, :) = []; % remove everything of height less than 10 pixels

    if rrr(dr) == 0, continue; end

    % apply filters
    idx_good = dr(:,3)>=width_min & dr(:,3)<=width_max & ...
        dr(:,4)>=height_min & dr(:,4)<=height_max;
    dr = dr(idx_good, :);   

    if rrr(dr) == 0, continue; end
    
    if  bool_visualize
        imshow(img);     
        draw_rect(dr);
        drawnow;
    end
    
    counter_obj = counter_obj + rrr(dr);   
    
    dr_con = bbApply( 'resize', dr, 128/100, 0, 0.5);
    patches = bbApply('crop', img, dr_con, 0, [64, 128]);
    
    for j=1:length(patches)
        s = sobj.get_index(); if s== 0, continue; end
        if bool_savepatch
            imwrite(patches{j}, sprintf('%s/%08d.png', dir_patches, s));  
        end
        if bool_savefeat 
            feats(s, :) = hog_dollar_new(im2single(patches{j}));
        end
    end    
   
end

if bool_savefeat
    save('feats_FG.mat', 'feats');
end


%% Load negative data samples from INRIA
S = load('D:/Research/PetDet/INRIA/SVM_HOG_saveData/dataset_train_HOG.mat');
feats_neg = S.feats_neg;
clear S;

%% train classifier
labels_train = [ones(size(feats, 1),1); 2*ones(size(feats_neg, 1),1)];
c  = liblinear_find_c_cautious(labels, [feats; feats_neg]);
liblinear_model = train(labels, sparse([feats; feats_neg]), sprintf('-s 0 -c %f -w1 %f -w2 1 -B 1 -q 1', c, sum(labels==2) / sum(labels==1)));
% liblinear_model = train(labels, sparse([feats; feats_neg]), sprintf('-s 1 -c %f -w1 %f -w2 1 -B 1 -q 1', c, sum(labels==2) / sum(labels==1)));

w_lin = logistic_regression_auto([feats; feats_neg], labels_train, 500, true);
visualize_hog(w_lin(1:end-1), 1);

str_readme = 'Classifier trained on bounding boxes obtained from FG ROIs vs negs';
save('classifier_prior.mat', 'w_lin', 'str_readme');

%% try detecting pedestrians with this classifier
img_orig = vobj.read(1);
img_scale = 6;
img = im2single(img_orig);
img = imresize(img, img_scale);
[dr, ds] = slidewin_detect_linear(img, w_lin, 1, 1);
[dr, ds] = nms_wrapper(dr, ds);
fprintf('# of bboxes = %d\n', size(dr, 1));
dr = round(dr / img_scale);
figure, imshow(img_orig); hold on; draw_rect(dr(ds > 0.5, :)); hold off;




