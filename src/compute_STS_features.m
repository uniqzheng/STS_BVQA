% Compute features for a set of video files from datasets
% 
close all; 
clear;
%addpath(genpath('../include/BRISQUE_release')); 
%addpath(genpath('../include/GM-LOG-BIQA')); 
addpath(genpath('../include'));
%addpath(genpath('../include/higradeRelease/higradeRelease'));
addpath(genpath('../include/RAPIQUE-main/include'));
%addpath(genpath('../include/FAVER'));
%addpath(genpath('../include/niqe-main'));

%% parameters
%algo_name = 'BRISQUE'; 
%algo_name = 'HIGRADE';
algo_name = 'RAPIQUE_spatial';
%algo_name = 'GM-LOG'; 
%algo_name = 'FAVER_spatial';  
%algo_name = 'NIQE'; 
%algo_name = 'resnet50';

data_name = 'Youtube-UGC'; %LIVE_VQC, KoNVid
data_path = '../../YouTube-UGC/original_videos_h264';
write_file = true;

video_tmp = '../tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
out_mat_name = fullfile(feat_path, [data_name,'_',algo_name,'_STS_resize_feats.mat']);
%feats_mat = zeros( num_videos, 36 );     %480*36
minside = 512;
switch(algo_name)
    case 'NIQE'
        load modelparameters.mat
        blocksizerow    = 96;
        blocksizecol    = 96;
        blockrowoverlap = 0;
        blockcoloverlap = 0;
        feats_mat = zeros(num_videos,3);
    case 'BRISQUE'
        feats_mat = zeros( num_videos, 108 );
    case 'GM-LOG'
        feats_mat = zeros( num_videos, 120 );
    case 'HIGRADE'
        feats_mat = zeros( num_videos, 216*3 );
    case 'RAPIQUE_spatial'
	    minside = 512.0;
		log_level = 0; 
	    feats_mat = zeros( num_videos, 680*3);
	case 'FAVER_spatial' %haar, db2, bior22
	    minside = 512.0;
		log_level = 0;
	    feats_mat = zeros(num_videos, 272*3);
    case 'resnet50'
	    net = resnet50;
		layer = 'avg_pool';
	    feats_mat = zeros( num_videos, 2048*3);
end

for i = 1:num_videos
    % get video file name
    % num2str(filelist.Filename(i))
    %strs = strsplit(filelist.File{i}, '.');
    video_name = fullfile(data_path,[filelist.filename{i}, '_crf_10_ss_00_t_20.0.mp4']);
    %video_name = fullfile(data_path,filelist.File{i});
    fprintf('Computing features for %d sequence: %s\n', i, video_name);
    
    % get video meta data
    resolution = filelist.resolution(i);
    switch (resolution)
        case {360}
            width = 480; 
            height = 360; 
        case {480}
            width = 640;
            height = 480;
        case {720}
            width = 1280;
            height = 720;
        case {1080}
            width = 1920;
            height = 1080;
        case {2160}
            width = 3840;
            height = 2160;
    end
%   width = filelist.width(i);
%   height = filelist.height(i);
%   framerate = round(filelist.framerate(i));
%   pixfmt = filelist.pixfmt{i};
    framerate = 30;
    pixfmt = 'yuv420p';
    
    % decode video and store in video_tmp dir
    yuv_name = fullfile(video_tmp, [filelist.filename{i}, '.yuv']);
    cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
        ' -i ', video_name, ' -pix_fmt ', pixfmt, ...
        ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
    system(cmd);
    
    % read YUV frame (credit: Dae Yeol Lee)  
    fp_input = fopen(yuv_name, 'r');
    fseek(fp_input, 0, 1);
    file_length = ftell(fp_input);
    nb_frames = floor(file_length/width/height/1.5); % for 8 bit

    fprintf('%d\n',nb_frames);
    scale_coef = nb_frames / framerate;
    fclose(fp_input);
    feats = [];
   
    switch(algo_name)
    case 'NIQE'
        tempfeat1 = [];
        tempfeat2 = [];
        tempfeat3 = [];
        dim = [width height];
        % XY-T
        for fr = floor(framerate/2):framerate:nb_frames-3
            try
            frame = YUVread(fp_input, dim, fr);
            y_plane = frame(:,:,1);
            tempfeat1(end+1,:) = computequality(y_plane,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
            mu_prisparam,cov_prisparam);
            catch
                continue
            end
            %imshow(y_plane,[]);
        end
        sside = min(width, height);
        ratio = minside / sside;
        if ratio < 1 
           new_reso = imresize(frame, ratio);
           height = size(new_reso,1);
		   width  = size(new_reso,2);
        end
        
        %cubic
        wht_frames = zeros(nb_frames-2, height, width, 3);
        for idx = 1 : nb_frames-2
		    orig_fr = YUVread(fp_input, dim, idx);
			if ratio < 1
                orig_fr = imresize(orig_fr, ratio);
            end
		    wht_frames(idx,:,:,:) = orig_fr;
        end
        %XT---Y
        for fr = floor(height/2/scale_coef):floor(height/scale_coef):height
            try
            frame = reshape(wht_frames(:,fr,:,:),[nb_frames-2, width, 3]);
            y_plane = frame(:,:,1);
            tempfeat2(end+1,:) = computequality(y_plane,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
            mu_prisparam,cov_prisparam);
            catch
                continue
            end
        end
        for fr = floor(width/2/scale_coef):floor(width/scale_coef):width    
            try
            frame = reshape(wht_frames(:,:,fr,:),[nb_frames-2, height, 3]);
            y_plane = frame(:,:,1);
            
            %imshow(y_plane,[]);
            tempfeat3(end+1,:) = computequality(y_plane,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
            mu_prisparam,cov_prisparam);
            catch
                continue
            end
        end
        fclose(fp_input);
        tempfeat1 = nanmean(tempfeat1);
        tempfeat2 = nanmean(tempfeat2);
        tempfeat3 = nanmean(tempfeat3);
        feats_mat(i,:) = [tempfeat1,tempfeat2,tempfeat3];
        delete(yuv_name);
        if write_file
           save(out_mat_name, 'feats_mat');
        end
        
    case 'BRISQUE'%%
        tempfeat1 = [];
        tempfeat2 = [];
        tempfeat3 = [];
        dim = [width height];
        % XY-T
        for fr = floor(framerate/2):framerate:nb_frames-3
            frame = YUVread(fp_input, dim, fr);
            y_plane = frame(:,:,1);
            tempfeat1(end+1,:) = brisque_feature(y_plane);
        end
        sside = min(width, height);
        ratio = minside / sside;
        if ratio < 1 
           new_reso = imresize(frame, ratio);
           height = size(new_reso,1);
		   width  = size(new_reso,2);
        end
        
        % cubic
        wht_frames = zeros(nb_frames-2, height, width, 3);
        for idx = 1 : nb_frames-2
		    orig_fr = YUVread(fp_input, dim, idx);
			if ratio < 1
                orig_fr = imresize(orig_fr, ratio);
            end
		    wht_frames(idx,:,:,:) = orig_fr;
        end
        %XT---Y
        for fr = floor(height/2/scale_coef):floor(height/scale_coef):height
            frame = reshape(wht_frames(:,fr,:,:),[nb_frames-2, width, 3]);
            y_plane = frame(:,:,1);
            tempfeat2(end+1,:) = brisque_feature(y_plane);
        end
        for fr = floor(width/2/scale_coef):floor(width/scale_coef):width    
            frame = reshape(wht_frames(:,:,fr,:),[nb_frames-2, height, 3]);
            y_plane = frame(:,:,1);
            tempfeat3(end+1,:) = brisque_feature(y_plane);
        end
        fclose(fp_input);
        tempfeat1 = nanmean(tempfeat1);
        tempfeat2 = nanmean(tempfeat2);
        tempfeat3 = nanmean(tempfeat3);
        feats_mat(i,:) = [tempfeat1,tempfeat2,tempfeat3];
        delete(yuv_name);
        if write_file
           save(out_mat_name, 'feats_mat');
        end

        
    case 'GM-LOG'%%
        tempfeat1 = [];
        tempfeat2 = [];
        tempfeat3 = [];
        dim = [width height];
        % XY-T
        for fr = floor(framerate/2):framerate:nb_frames-3
            frame = YUVread(fp_input, dim, fr);
            y_plane = frame(:,:,1);
            tempfeat1(end+1,:) = Grad_LOG_CP_TIP(y_plane);
        end
        sside = min(width, height);
        ratio = minside / sside;
        if ratio < 1 
           new_reso = imresize(frame, ratio);
           height = size(new_reso,1);
		   width  = size(new_reso,2);
        end
        
        % cubic
        wht_frames = zeros(nb_frames-2, height, width, 3);
        for idx = 1 : nb_frames-2
		    orig_fr = YUVread(fp_input, dim, idx);
			if ratio < 1
                orig_fr = imresize(orig_fr, ratio);
            end
		    wht_frames(idx,:,:,:) = orig_fr;
        end
        %XT---Y
        for fr = floor(height/2/scale_coef):floor(height/scale_coef):height
            frame = reshape(wht_frames(:,fr,:,:),[nb_frames-2, width, 3]);
            y_plane = frame(:,:,1);
            tempfeat2(end+1,:) = Grad_LOG_CP_TIP(y_plane);
        end
        for fr = floor(width/2/scale_coef):floor(width/scale_coef):width    
            frame = reshape(wht_frames(:,:,fr,:),[nb_frames-2, height, 3]);
            y_plane = frame(:,:,1);
            tempfeat3(end+1,:) = Grad_LOG_CP_TIP(y_plane);
        end
        fclose(fp_input);
        tempfeat1 = nanmean(tempfeat1);
        tempfeat2 = nanmean(tempfeat2);
        tempfeat3 = nanmean(tempfeat3);
        feats_mat(i,:) = [tempfeat1,tempfeat2,tempfeat3];
        delete(yuv_name);
        if write_file
           save(out_mat_name, 'feats_mat');
        end

    case 'HIGRADE'%%
	    tempfeat1 = [];
        tempfeat2 = [];
        tempfeat3 = [];
        dim = [width height];
        % XY-T
        for fr = floor(framerate/2):framerate:nb_frames-3
            YUV = YUVread(fp_input, dim, fr);
			RGB = reshape(convertYuvToRgb(reshape(YUV, width * height, 3)), ...
                          height, width, 3);
            tempfeat1(end+1,:) = higrade_1(RGB);
        end
        sside = min(width, height);
        ratio = minside / sside;
        if ratio < 1 
           new_reso = imresize(YUV, ratio);
           height = size(new_reso,1);
		   width  = size(new_reso,2);
        end
        
        % cubic
        wht_frames = zeros(nb_frames-2, height, width, 3);
        for idx = 1 : nb_frames-2
		    orig_fr = YUVread(fp_input, dim, idx);
			if ratio < 1
                orig_fr = imresize(orig_fr, ratio);
            end
		    wht_frames(idx,:,:,:) = orig_fr;
        end
        %XT---Y
        for fr = floor(height/2/scale_coef):floor(height/scale_coef):height
            YUV = reshape(wht_frames(:,fr,:,:),[nb_frames-2, width, 3]);
			RGB = reshape(convertYuvToRgb(reshape(YUV, width * (nb_frames-2), 3)), ...
                      (nb_frames-2), width, 3);
            tempfeat2(end+1,:) = higrade_1(RGB);
        end
        for fr = floor(width/2/scale_coef):floor(width/scale_coef):width    
            YUV = reshape(wht_frames(:,:,fr,:),[nb_frames-2, height, 3]);
            RGB = reshape(convertYuvToRgb(reshape(YUV, (nb_frames-2) * height, 3)), ...
                          height, (nb_frames-2), 3);
            tempfeat3(end+1,:) = higrade_1(RGB);
        end
        fclose(fp_input);
        tempfeat1 = nanmean(tempfeat1);
        tempfeat2 = nanmean(tempfeat2);
        tempfeat3 = nanmean(tempfeat3);
        feats_mat(i,:) = [tempfeat1,tempfeat2,tempfeat3];
        delete(yuv_name);
        if write_file
           save(out_mat_name, 'feats_mat');
        end
        
    case 'RAPIQUE_spatial'%%
        feats_mat(i,:) = calc_RAPIQUE_features(yuv_name, width, height, ...
framerate, minside, log_level, nb_frames, scale_coef);
        delete(yuv_name);
        if write_file
           save(out_mat_name, 'feats_mat');
        end
    case 'FAVER_spatial' %haar, db2, bior22
        feats_mat(i,:) = calc_FAVER_features(yuv_name,width, height, ...
                                    framerate,log_level,nb_frames,scale_coef);
        delete(yuv_name);
        if write_file
           save(out_mat_name, 'feats_mat');
        end
    case 'resnet50'%% extract deep learning features
		tempfeat1 = [];
        tempfeat2 = [];
        tempfeat3 = [];
        dim = [width height];
        % XY-T
        for fr = floor(framerate/2):framerate:nb_frames-3
            YUV = YUVread(fp_input, dim, fr);
            RGB = reshape(convertYuvToRgb(reshape(YUV, width * height, 3)), ...
                          height, width, 3);
            %imshow(y_plane,[]);
			input_size = net.Layers(1).InputSize;
            im_scale = imresize(RGB, [input_size(1), input_size(2)]);
            feats_spt_deep = activations(net, im_scale, layer, ...
                            'ExecutionEnvironment','cpu');
            tempfeat1(end+1,:) = squeeze(feats_spt_deep);
        end
        sside = min(width, height);
        ratio = minside / sside;
        if ratio < 1 
           new_reso = imresize(YUV, ratio);
           height = size(new_reso,1);
		   width  = size(new_reso,2);
        end
        
        % cubic
        wht_frames = zeros(nb_frames-2, height, width, 3);
        for idx = 1 : nb_frames-2
		    orig_fr = YUVread(fp_input, dim, idx);
			if ratio < 1
                orig_fr = imresize(orig_fr, ratio);
            end
		    wht_frames(idx,:,:,:) = orig_fr;
        end
        %XT---Y
        for fr = floor(height/2/scale_coef):floor(height/scale_coef):height
            YUV = reshape(wht_frames(:,fr,:,:),[nb_frames-2, width, 3]);
            RGB = reshape(convertYuvToRgb(reshape(YUV, width * (nb_frames-2), 3)), ...
                      (nb_frames-2), width, 3);
            input_size = net.Layers(1).InputSize;
            im_scale = imresize(RGB, [input_size(1), input_size(2)]);
            feats_spt_deep = activations(net, im_scale, layer, ...
                            'ExecutionEnvironment','cpu');
            tempfeat2(end+1,:) = squeeze(feats_spt_deep);
        end
        for fr = floor(width/2/scale_coef):floor(width/scale_coef):width    
            YUV = reshape(wht_frames(:,:,fr,:),[nb_frames-2, height, 3]);
            RGB = reshape(convertYuvToRgb(reshape(YUV, (nb_frames-2) * height, 3)), ...
                          height, (nb_frames-2), 3);
            input_size = net.Layers(1).InputSize;
            im_scale = imresize(RGB, [input_size(1), input_size(2)]);
            feats_spt_deep = activations(net, im_scale, layer, ...
                            'ExecutionEnvironment','cpu');
            tempfeat3(end+1,:) = squeeze(feats_spt_deep);
        end
        fclose(fp_input);
        tempfeat1 = nanmean(tempfeat1);
        tempfeat2 = nanmean(tempfeat2);
        tempfeat3 = nanmean(tempfeat3);
        feats_mat(i,:) = [tempfeat1,tempfeat2,tempfeat3];
        delete(yuv_name);
        if write_file
           save(out_mat_name, 'feats_mat');
        end
    end    
end

if write_file
    save(out_mat_name, 'feats_mat');
end

% Read one frame from YUV file
function YUV = YUVread(f, dim, frnum)
    fseek(f, dim(1)*dim(2)*1.5*frnum, 'bof');
    
        % Read Y-component
        Y = fread(f, dim(1)*dim(2), 'uchar');
        if length(Y) < dim(1)*dim(2)
            YUV = [];
            return;
        end
        Y = cast(reshape(Y, dim(1), dim(2)), 'double');
        
        % Read U-component
        U = fread(f, dim(1)*dim(2)/4, 'uchar');
        if length(U) < dim(1)*dim(2)/4
            YUV = [];
            return;
        end
        U = cast(reshape(U, dim(1)/2, dim(2)/2), 'double');
        U = imresize(U, 2.0);
        
        % Read V-component
        V = fread(f, dim(1)*dim(2)/4, 'uchar');
        if length(V) < dim(1)*dim(2)/4
            YUV = [];
            return;
        end    
        V = cast(reshape(V, dim(1)/2, dim(2)/2), 'double');
        V = imresize(V, 2.0);
        
        % Combine Y, U, and V
        YUV(:,:,1) = Y';
        YUV(:,:,2) = U';
        YUV(:,:,3) = V';
end
