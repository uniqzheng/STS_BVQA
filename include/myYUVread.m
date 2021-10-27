function frames = myYUVread(YUVfile, height, width, nb_frames)

    fp_input = fopen(YUVfile, 'r');

    %% Start a file pointer
    fseek(fp_input, 0, 'bof');

    frames = zeros(height, width, nb_frames);
    
    for idx = 1:nb_frames
        %Read Y-component
        Y = fread(fp_input, height * width, 'uchar'); % Frame read for 8 bit
        %if (Y)
          Y = reshape(Y, [width height]);
          frames(:, :, idx) = Y';
          % frame(idx,height,width,chn) = YUV
          % Y = frame(idx,height,width,1);
        %end
        % Read U-component
        U = fread(fp_input, width * height / 4, 'uchar');
        % Read V-component
        V = fread(fp_input, width * height / 4, 'uchar');
    end
    
    fclose(fp_input);
    % online fetch frames