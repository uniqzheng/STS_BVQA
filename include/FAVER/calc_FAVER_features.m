function feats_frames = calc_FAVER_features(test_video,width, height, ...
                                            framerate, log_level,nb_frames, scale_coef)
     feats_frames = [];
    % Try to open test_video; if cannot, return
    test_file = fopen(test_video,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        feats_frames = [];
        return;
    end
    % Open test video file
    %fseek(test_file, 0, 1);
    %file_length = ftell(test_file);
    if log_level == 1
        fprintf('Video file size: %d bytes (%d frames)\n',file_length, ...
                floor(file_length/width/height/1.5));
    end
    % get frame number
    %if strcmp(pixfmt, 'yuv420p')
     %   nb_frames = floor(file_length/width/height/1.5); % for 8 bit
    %else
    %    nb_frames = floor(file_length/width/height/3.0); % for 10 bit
    %end
    %fprintf('nb_frames: %d\n',nb_frames);
    % get features for each chunk
        minside = 512.0;
        tempfeat1 = [];
        tempfeat2 = [];
        tempfeat3 = [];
        dim = [width height];
        % XY-T
        for fr = floor(framerate/2):framerate:nb_frames-3
            
            frame = YUVread(test_file, dim, fr);
            tempfeat1(end+1,:) = FAVER_spatial_features(frame);
            %imshow(y_plane,[]);
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
		    orig_fr = YUVread(test_file, dim, idx);
			if ratio < 1
                orig_fr = imresize(orig_fr, ratio);
            end
		    wht_frames(idx,:,:,:) = orig_fr;
        end
        %XT---Y
        for fr = floor(height/2/scale_coef):floor(height/scale_coef):height
            
            frame = reshape(wht_frames(:,fr,:,:),[nb_frames-2, width, 3]);
            tempfeat2(end+1,:) = FAVER_spatial_features(frame);
        end
        for fr = floor(width/2/scale_coef):floor(width/scale_coef):width    
           
            frame = reshape(wht_frames(:,:,fr,:),[nb_frames-2, height, 3]);
            tempfeat3(end+1,:) = FAVER_spatial_features(frame);
        end
        fclose(test_file);
        tempfeat1 = nanmean(tempfeat1);
        tempfeat2 = nanmean(tempfeat2);
        tempfeat3 = nanmean(tempfeat3);
        feats_frames = [tempfeat1,tempfeat2,tempfeat3];
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