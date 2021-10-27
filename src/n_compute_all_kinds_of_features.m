% Compute features for a set of video files from datasets
% 
close all; 
clear;
%addpath(genpath('../include/BRISQUE_release')); 
addpath(genpath('../include/GM-LOG-BIQA')); 
addpath(genpath('../include'));
%addpath(genpath('../include/FRIQUEE_Release/FRIQUEE_Release/src'));     % feature extract
%addpath(genpath('../include/FRIQUEE_Release/FRIQUEE_Release/include')); % matlabPyrTools
%addpath(genpath('../include/CORNIA')); 
%addpath(genpath('../include/HOSA')); 
%addpath(genpath('../include/nr-vqa-consumervideo')); 
%addpath(genpath('../include/VIDEVAL/include')); %420p
%addpath(genpath('../include/higradeRelease/higradeRelease'));
%addpath(genpath('../include/RAPIQUE-main/include'));
%addpath(genpath('../include/FAVER'));
%addpath(genpath('../include/niqe-main'));

%% parameters
%algo_name = 'BRISQUE'; 
%algo_name = 'HIGRADE';
%algo_name = 'RAPIQUE';
algo_name = 'GM-LOG';
%algo_name = 'FRIQUEE'; 
%algo_name = 'CORNIA'; 
%algo_name = 'HOSA'; 
%algo_name = 'TLVQM'; 
%algo_name = 'FAVER_Haar_T';  %haar, db2, bior22
%algo_name = 'NIQE'; 

data_name = 'KoNVid';
data_path = '../../database/KoNViD_1k_videos';
write_file = true;

video_tmp = '../tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
out_mat_name = fullfile(feat_path, [data_name,'_',algo_name,'_STS_feats.mat']);
%feats_mat = zeros( num_videos, 36 );     %480*36

switch(algo_name)
    case 'NIQE'
        
        load modelparameters.mat
 
        blocksizerow    = 96;
        blocksizecol    = 96;
        blockrowoverlap = 0;
        blockcoloverlap = 0;
        feats_mat = zeros(num_videos,1);
    case 'BRISQUE'
        feats_mat = zeros( num_videos, 108 );
    case 'GM-LOG'
        feats_mat = zeros( num_videos, 120 );
    case 'HIGRADE'
        feats_mat = zeros( num_videos, 216 );
    case 'FRIQUEE'
        feats_mat = zeros( num_videos, 560 );

    case 'CORNIA' % Not sure about the size of feats_mat. Need an experiment to figure it out.
        % load codebook
        load('CSIQ_codebook_BS7.mat','codebook0');
        load('LIVE_soft_svm_model.mat','soft_model','soft_scale_param');
        % load whitening parameter
        load('CSIQ_whitening_param.mat','M','P');
        feats_mat = zeros( num_videos, 20000 );

    case 'HOSA' % Not sure about the size of feats_mat. Need an experiment to figure it out.
        load('whitening_param.mat', 'M', 'P');
        load('codebook_hosa', 'codebook_hosa');
        BS = 7; % patch size
        power = 0.2; % signed power normalizaiton param
        feats_mat = zeros( num_videos, 14700);

    case 'TLVQM'
        feats_mat = zeros( num_videos, 75 );
    case 'VIDEVAL'
        feats_mat = zeros( num_videos, 60 );
    case 'RAPIQUE'
	    minside = 512.0;
		log_level = 0; 
		net = resnet50;
		layer = 'avg_pool';
	    feats_mat = zeros( num_videos, 3884);
	case 'FAVER_Haar_T' %haar, db2, bior22
	    minside = 512.0;
		log_level = 0;
	    feats_mat = zeros( num_videos, 476);
end

for i = 1:num_videos
    % get video file name

    %strs = strsplit(filelist.File{i}, '.');
    video_name = fullfile(data_path,[num2str(filelist.Filename(i)),'.mp4']);
    fprintf('Computing features for %d sequence: %s\n', i, video_name);
    
    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));

    % decode video and store in video_tmp dir
    yuv_name = fullfile(video_tmp, [num2str(filelist.Filename(i)), '.yuv']);
    cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
        ' -i ', video_name, ' -pix_fmt ', filelist.pixfmt{i}, ...
        ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
    system(cmd);
    
    % read YUV frame (credit: Dae Yeol Lee)  
     fp_input = fopen(yuv_name, 'r');
     fseek(fp_input, 0, 1);
     file_length = ftell(fp_input);
%      if strcmp(filelist.pixfmt{i}, 'yuv420p')
         nb_frames = floor(file_length/width/height/1.5); % for 8 bit
%      else
%          nb_frames = floor(file_length/width/height/3.0); % for 10 bit
%      end
    %nb_frames = filelist.nb_frames(i);
    fprintf('%d\n',nb_frames);
    scale_coef = nb_frames / framerate;
    fclose(fp_input);
    feats = [];
   
    switch(algo_name)
    case 'NIQE'
        feats(end+1,:)=computequality(RGB,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
mu_prisparam,cov_prisparam);
    case 'BRISQUE'%%
        tempfeat1 = [];
        tempfeat2 = [];
        tempfeat3 = [];
        %tempall = [];
        frames = myYUVread(yuv_name, height, width, nb_frames);
        % XY-T
        for fr = floor(framerate/2):framerate:nb_frames-3
            y_plane = frames(:,:,fr);
            %imshow(y_plane,[]);
            tempfeat1(end+1,:) = brisque_feature(y_plane);
        end
        tempfeat1 = nanmean(tempfeat1);
        % XT-Y
        for fr = floor(height/2/scale_coef):floor(height/scale_coef):height
            y_plane = reshape(frames(fr,:,:),[width nb_frames]);
            %imshow(y_plane,[]);
            tempfeat2(end+1,:) = brisque_feature(y_plane);
        end
        tempfeat2 = nanmean(tempfeat2);
        % YT-X
        for fr = floor(width/2/scale_coef):floor(width/scale_coef):width
            y_plane = reshape(frames(:,fr,:),[height nb_frames]);
            %imshow(y_plane,[]);
            tempfeat3(end+1,:) = brisque_feature(y_plane);
        end
        tempfeat3 = nanmean(tempfeat3);
        feats_mat(i,:) = [tempfeat1,tempfeat2,tempfeat3];
        delete(yuv_name);
        %feats_mat(i,:) = nanmean(tempall);  %求平均
        if write_file
           save(out_mat_name, 'feats_mat');
        end
        
    case 'GM-LOG'%%
        tempfeat1 = [];
        tempfeat2 = [];
        tempfeat3 = [];
        %tempall = [];
        frames = myYUVread(yuv_name, height, width, nb_frames);
        % XY-T
        for fr = floor(framerate/2):framerate:nb_frames-3
            y_plane = frames(:,:,fr);
            %imshow(y_plane,[]);
            tempfeat1(end+1,:) = Grad_LOG_CP_TIP(y_plane);
        end
        tempfeat1 = nanmean(tempfeat1);
        % XT-Y
        for fr = floor(height/2/scale_coef):floor(height/scale_coef):height
            y_plane = reshape(frames(fr,:,:),[width nb_frames]);
            %imshow(y_plane,[]);
            tempfeat2(end+1,:) = Grad_LOG_CP_TIP(y_plane);
        end
        tempfeat2 = nanmean(tempfeat2);
        % YT-X
        for fr = floor(width/2/scale_coef):floor(width/scale_coef):width
            y_plane = reshape(frames(:,fr,:),[height nb_frames]);
            %imshow(y_plane,[]);
            tempfeat3(end+1,:) = Grad_LOG_CP_TIP(y_plane);
        end
        tempfeat3 = nanmean(tempfeat3);
        feats_mat(i,:) = [tempfeat1,tempfeat2,tempfeat3];
        delete(yuv_name);
        %feats_mat(i,:) = nanmean(tempall);  %求平均
        if write_file
           save(out_mat_name, 'feats_mat');
        end

    case 'HIGRADE'%%
        feats(end+1,:) = higrade_1(RGB);

    case 'FRIQUEE'%%
        feats_frtmp = extractFRIQUEEFeatures(RGB);
        feats(end+1,:) = feats_frtmp.friqueeALL;
        
    case 'CORNIA'%
        feats(end+1,:) = CORNIA_Fv(y_plane, codebook0, 'soft', M, P,sqrt(size(codebook0,1)),10000);
    case 'HOSA'%
        fv = hosa_feature_extraction(codebook_hosa.centroid_cb, codebook_hosa.variance_cb, ...
            codebook_hosa.skewness_cb, M, P, BS, power, y_plane);
    case 'TLVQM'%%

        feats(end+1,:) = compute_nrvqa_features(yuv_name,[width height],framerate);

    case 'VIDEVAL'%%
        %feats(end+1,:) = calc_VIDEVAL_feats(yuv_name,width,height,framerate);
        feats(end+1,:) = calc_VIDEVAL_feats_light(yuv_name,width,height,framerate,480,3);
        
    case 'RAPIQUE'%%
        feats(end+1,:) = calc_RAPIQUE_features(yuv_name, width, height, ...
framerate, minside, net, layer, log_level);
    case 'FAVER_Haar_T' %haar, db2, bior22
        feats = calc_FAVER_features_test(video_name,width, height, ...
                                    framerate, minside, log_level);
    end

    % compute frame-level features and average
    % for fr = floor(framerate/2):framerate:nb_frames-3  %取每秒中间的图片
    %     %if strcmp(filelist.pixfmt{i}, 'yuv420p')
    %         YUV = myYUVread(fp_input, [width height], fr, 'yuv420p');
    %         y_plane= YUV(:,:,1);
    %         RGB = reshape(convertYuvToRgb(reshape(YUV, width * height, 3)), ...
    %                      height, width, 3);
        % else 
        %     YUV = myYUVread(fp_input, [width height], fr, filelist.pixfmt{i})./4;
        %     y_plane= YUV(:,:,1);
        %     RGB = reshape(convertYuvToRgb(reshape(YUV, width * height, 3)), ...
        %                   height, width, 3);
        % end
%         imwrite(RGB, 'test.png');
         %imshow(RGB);
		
        %toc
	
%         catch
%             continue
%         end
    %end
    % fclose(fp_input);
    
end
%mean(T(:))
%toc
if write_file
    save(out_mat_name, 'feats_mat');
end
