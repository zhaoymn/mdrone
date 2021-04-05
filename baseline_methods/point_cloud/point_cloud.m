%radar parameters
c = 3e8;
start_freq = 60.25e9;
freq_slope = 72.999e12;
adc_samples = 256;
sample_rate = 12000e3;
chirp_loops = 128;
period = 0.1;
idle_time = 7e-6;
ramp_end_time = 51.14e-6;
chirp_time = idle_time + ramp_end_time;
N_fft = 128;
lambda = c / start_freq;
bandwidth = freq_slope * adc_samples / sample_rate;
d_res = c / 2 / bandwidth;
d_max = sample_rate * c / 2 / freq_slope;
v_res = lambda / 2 / N_fft / chirp_time;
frames = 600;
rx_num = 4;
tx_num = 3;
clutter_removal_enabled = 1;

raw_data_dir = %raw data dir here

ground_truth_dir = %ground truth dir here


test_sequences = csvread('../../test_sequences.csv');
all_error = [];
for sequence_id = 1:1:10
    sequence = test_sequences(sequence_id);
    %sequence = sequence_id
    filename = strcat(raw_data_dir,num2str(sequence),"\adc_data_0.bin");
    gt_filename = strcat(ground_truth_dir, num2str(sequence), '.npy');
    gt = readNPY(gt_filename);
    gt = reshape(gt(1:600,:,:), 600, 16);
    gt_xyz = gt(:, 13:15);
    predict_xyz = zeros(600,3);
    data = load_mm_raw(filename, rx_num, tx_num, chirp_loops, frames, adc_samples);
    sequence_error = 0
    tic
    for frame = 1:1:600
        frame
        %load data
        frame_data = permute(reshape(data(:,:,:,frame,:), rx_num, tx_num, chirp_loops, adc_samples), [1 2 4 3]);
        %1d fft
        one_d_fft = zeros(3, 4, 256, 128);
        for tx = 1:1:3
            for rx = 1:1:4
                for chirp = 1:1:128
                    one_d_fft(tx, rx,:,chirp) = fft(reshape(frame_data(rx,tx,:,chirp),1,adc_samples));
                end
            end
        end
                
        if clutter_removal_enabled == 1
            for tx = 1:1:3
                for rx = 1:1:4
                    average = mean(one_d_fft(tx, rx, :, :), 4);
                    for chirp = 1:1:128
                        one_d_fft(tx, rx, :, chirp) = one_d_fft(tx, rx, :, chirp) - average;
                    end
                end
            end
        else
        end
        range_velocity_fft = zeros(3, 4, 256, 128);
        %calculate 2d fft
        for tx = 1:1:3
            for rx = 1:1:4
                for sample = 1:1:256
                    range_velocity_fft(tx, rx, sample, :) = fftshift(fft(reshape(one_d_fft(tx, rx, sample, :),1, 128)));
                end
            end
        end
        resp = zeros(256, 128);
        for tx = 1:1:3
            for rx = 1:1:4
                resp = resp + abs(reshape(range_velocity_fft(tx, rx, :,:), 256, 128))/12;
            end
        end
        %find target (cfar)
        cfar2D = phased.CFARDetector2D('GuardBandSize',1,'TrainingBandSize',2,...
          'ProbabilityFalseAlarm',1e-1);
        %[resp,rngGrid,dopGrid] = helperRangeDoppler;

        rngGrid = (1:1:256) * d_res;
        dopGrid = (-64:1:63) * v_res;

        CUTIdx = [];
        for rng_idx = 4:1:63
            for dop_idx = 11:1:118
                CUTIdx = [CUTIdx [rng_idx dop_idx]'];
            end
        end
        detections = cfar2D(resp,CUTIdx);
        
        %fov limit
        %% point cloud generation
        target_list = [];
        for i=1:1:size(detections, 1)
            if detections(i) == 1
                t_rng = CUTIdx(1,i);
                t_dop = CUTIdx(2,i);

        %distance estimation
                target_distance = t_rng * d_res;

        %doppler compensation
                compensated_dop = zeros(3,4);
                for tx = 1:1:3
                    for rx = 1:1:4
                        compensated_dop(tx, rx) = range_velocity_fft(tx, rx, t_rng, t_dop);
                    end
                end
                i_dop = t_dop - 64;
                N_fft_dop = 128;
                phi_dop = 2 * pi * i_dop / N_fft_dop;
                delta = phi_dop / 3;
                for tx = 1:1:3
                    for rx = 1:1:4
                        compensated_dop(tx, rx) = compensated_dop(tx, rx) * exp(-1i * tx * delta);
                    end
                end
                
        %azimuth angle estimation
                azimuth_antennas = [compensated_dop(1, :) compensated_dop(2,:)];
                N_angle_fft = 1024;
                azimuth_angle_fft = fftshift(fft(azimuth_antennas, N_angle_fft));
                [argvalue_a, argmax_a] = max(abs(azimuth_angle_fft));
                P1 = azimuth_angle_fft(argmax_a);
                w_x = 2 * pi / N_angle_fft * (argmax_a - N_angle_fft/2);

        %elevation angle estimation
                elevation_antennas = compensated_dop(3,:);
                elevation_angle_fft = fftshift(fft(elevation_antennas, N_angle_fft));
                [argvalue_e, argmax_e] = max(abs(elevation_angle_fft));
                P2 = elevation_angle_fft(argmax_e);
                w_z = angle(P1 * conj(P2) * exp(1i * 2 * w_x));
                
        %fov elimination
        
                phi_max = 60 * pi / 180;
                phi_min = -60 * pi / 180;
                theta_max = 60 * pi / 180;
                theta_min = -60 * pi / 180;
                W_z = w_z / pi;
                W_x = w_x / pi;
                if W_z >= sin(phi_min) && W_z < sin(phi_max)
                        target_x = target_distance * w_x / pi;
                        target_z = target_distance * w_z / pi;
                        if target_distance^2 > target_x^2 + target_z^2
                            target_y = sqrt(target_distance^2 - target_x^2 - target_z^2);
                            target_list = [target_list [target_x target_y target_z]'];
                        end
                    %end
                end
            end
        end
        
        
        %limit fov
        valid_target_list = [];
        for i=1:1:size(target_list, 2)
            if target_list(1,i) < 2.5 && target_list(1,i)>-2.5 && target_list(2,i) < 5 && target_list(3,i) > -1 && target_list(3,i) < 2
                valid_target_list = [valid_target_list target_list(:,i)];
            end
        end
        not_found = 0;
        if size(valid_target_list, 1) == 0
            not_found = 1;
        end
        if not_found == 1
            predict_xyz(frame, 1) = predict_xyz(frame-1, 1);
            predict_xyz(frame, 2) = predict_xyz(frame-1, 2);
            predict_xyz(frame, 3) = predict_xyz(frame-1, 3);
            sequence_error = sequence_error + sqrt((gt_xyz(frame, 1) - frame_x)^2 + (gt_xyz(frame, 2)-frame_y)^2 + (gt_xyz(frame,3) - frame_z)^2);
            continue
        end
            
        %DBScan
        valid_target_list = valid_target_list';
        db_idx = dbscan(valid_target_list, 0.3, 1);

        
%         scatter3(valid_target_list(:,1), valid_target_list(:,2), valid_target_list(:,3), 20, db_idx+2);
%         colormap(jet(5));
%         hold on
%         scatter3(gt_xyz(1,1),gt_xyz(1,2),gt_xyz(1,3));
%         gscatter(valid_target_list(:,1),valid_target_list(:,2),db_idx);
        %hold on
        
        %target location
        
        res_num = zeros(100,1);
        res_x = zeros(100,1);
        res_y = zeros(100,1);
        res_z = zeros(100,1);
        for idx=1:1:size(db_idx)
            if db_idx(idx) >0
                res_num(db_idx(idx)) = res_num(db_idx(idx)) + 1;
                res_x(db_idx(idx)) = res_x(db_idx(idx)) + valid_target_list(idx, 1);
                res_y(db_idx(idx)) = res_y(db_idx(idx)) + valid_target_list(idx, 2);
                res_z(db_idx(idx)) = res_z(db_idx(idx)) + valid_target_list(idx, 3);
            end
        end
        max_pts = 0;
        max_idx = 0;
        for idx=1:1:10
            if res_num(idx) > max_pts
                max_pts = res_num(idx);
                max_idx = idx;
            end
        end
        if max_idx == 0
            predict_xyz(frame, 1) = predict_xyz(frame-1, 1);
            predict_xyz(frame, 2) = predict_xyz(frame-1, 2);
            predict_xyz(frame, 3) = predict_xyz(frame-1, 3);
        else
            frame_x = res_x(max_idx) / max_pts;
            frame_y = res_y(max_idx) / max_pts;
            frame_z = res_z(max_idx) / max_pts;
            predict_xyz(frame, 1) = frame_x;
            predict_xyz(frame, 2) = frame_y;
            predict_xyz(frame, 3) = frame_z;
        end
        sequence_error = sequence_error + sqrt((gt_xyz(frame, 1) - frame_x)^2 + (gt_xyz(frame, 2)-frame_y)^2 + (gt_xyz(frame,3) - frame_z)^2);
        
    end
    toc
    %calculate error
    sequence_error = sequence_error / 600
    all_error = [all_error sequence_error];
    output_dir = strcat('./result/',num2str(sequence),'.mat');
    mkdir('./result/')
    save(output_dir, 'predict_xyz')
end
% mean(all_error)
