
%% Training process

load('LIVE_data.mat');
% dmos_new_live: 
%       the subjective distortion score of LIVE database
% ind_live: 
%       bool variables, 0 indicates a pristine image and 1 indicates a
%       distorted image, the pristine image will be removed in experiments.
% ref_ind_live:
%       vectors to indicate the index of the reference image for all the
%       image. This will be used in the spliting of the traning set and
%       testing set.
% live_feature: 
%       the NR feature for all the image in LIVE databases, each row for one image.


% preparing the data for traning and test

Ref_number = max(ref_ind_live); % The numbe of reference image

N = 1000; % repeatation times

% splitting the databases acroding to the reference index
REF = round(Ref_number*0.8);
C = zeros(N,REF);
for j = 1:N
    rand_order = randperm(Ref_number);
    C(j,:) = rand_order(1:REF);
end

data = live_feature(ind_live,:); % input data to the SVR model
label = dmos_new_live(ind_live); % output of the SVR model


% the (cost, gamma) parameters for the SVR learning
% the optimal parameters we found for the 3 proposed models on the
% databases.
% (cost, gamma)  = (65536, 2) for LIVE M1
% (cost, gamma)  = (1024,   8) for LIVE M2
% (cost, gamma)  = (16384, 2) for LIVE M3
% (cost, gamma)  = (32768, 4) for CSIQ M1
% (cost, gamma)  = (65536, 2) for CSIQ M2
% (cost, gamma)  = (16384, 2) for CSIQ M3
% (cost, gamma)  = (2048 ,16) for TID2008 M1
% (cost, gamma)  = (2048,   8) for TID2008 M2
% (cost, gamma)  = (128,   16) for TID2008 M3
cost = 16384;
gamma = 2;
c_str = sprintf('%f',cost);
g_str = sprintf('%.2f',gamma);
libsvm_options = ['-s 3 -t 2 -g ',g_str,' -c ',c_str];

spear_results = zeros(N,1);
% Training and Test procedure
for i = 1:N
    % get the index of the distorted image from the index of the reference
    % image for the traning set and the test set.
    train = ismember(ref_ind_live,C(i,:));
    test = ~train;
    
    model = svmtrain(label(train),data(train,:),libsvm_options);
    [predict_score, ~, ~] = svmpredict(label(test), data(test,:), model);    
    spear_results(i) = corr(predict_score, label(test),'type','Spearman');
end
% median Spearman Coefficient (SRC) performance
spear_median = median(spear_results);
spear_std = std(spear_results,0);

%% read image data from file and compute the NR features

im_cell = dir('image\*.bmp');
im_cell = struct2cell(im_cell);

% number of the image in the folder
num_image = size(im_cell,2);

% matrix for the feature 
feature_mat = zeros(num_image, 40);

for k = 1:num_image
    im = imread(['image\',im_cell{1,k}]);
    im  =double(rgb2gray(im));
    feature = Grad_LOG_CP_TIP(im);
    feature_mat(k,:) = feature;
end

%% predict quality from the feature with the trained model

% load the trained model on LIVE database, M3 by default
load('Learned_SVR_model_on_LIVE.mat','LIVE_SVR_M3');

svr_model = LIVE_SVR_M3;
predict_score = svmpredict(zeros(num_image, 1), feature_mat, svr_model);

