%radar parameters
base_frequency = 60.25e9;
wave_length = 1*3e8/base_frequency;
d = wave_length/2;
c = 3e8;
lambda = c/base_frequency;
freq_slope = 72.999e12;
sample_rate = 12000e3;
period = 0.1;
idle_time = 7e-6;
ramp_end_time = 51.14e-6;
bandwidth = freq_slope* ramp_end_time;
%parameters
rx_num = 4;
tx_num = 3;
chirp_loops = 128;
frames = 600;
samples_per_chirp = 256;
minimum_distance = 1;
maximum_distance = 4;
minimum_angle_a = -60;
maximum_angle_a = 60;
minimum_angle_e = -60;
maximum_angle_e = 60;
N_FFTD = 256;
N_FFTT = 180;
min_didx = round(floor((2*freq_slope*minimum_distance*N_FFTD) / (sample_rate*c)));
max_didx = round(ceil((2*freq_slope*maximum_distance*N_FFTD) / (sample_rate*c)));
r_min = 12;
r_max = 168;
%data = load_mm_raw(filename, rx_num, tx_num, chirp_loops, frames, samples_per_chirp);


test_sequences = csvread('../../test_sequences.csv');


raw_data_dir = %raw data dir here
ground_truth_dir = %ground truth dir here

for sequence_id =1:1:10
    sequence
    sequence = test_sequences(sequence_id);
    filename = strcat(raw_data_dir,num2str(sequence),"\adc_data_0.bin");

    gt = readNPY(strcat(ground_truth_dir, num2str(sequence), '.npy'));
    parsed_gt = [];
    gt_polar = [];
    for i=1:1:600
        gt_x = gt(i,1,4);
        gt_y = gt(i,2,4);
        gt_z = gt(i,3,4);
        parsed_gt = [parsed_gt; gt_x gt_y gt_z];
        gt_d = sqrt(gt_x^2 + gt_y^2 + gt_z^2);
        gt_theta = acos(gt_x / sqrt(gt_x^2 + gt_y^2));
        gt_phi = asin(gt_z/gt_d);
        gt_polar = [gt_polar; gt_d gt_theta gt_phi];
        if gt_phi >1
            [gt_x, gt_y, gt_z]
        end
    end
    
    debug = 0;
    if debug == 1
        frames = 1
    end
    t_z = [];
    t_x = [];
    t_y = [];
    data = load_mm_raw(filename, rx_num, tx_num, chirp_loops, frames, samples_per_chirp);
    tic
    for frame_id=1:1:600
        %frame_id
        frame_data = reshape(data(:,:,:,frame_id,:), rx_num, tx_num, chirp_loops, samples_per_chirp);
        average_chirp = mean(frame_data, 3);
        single_chirp_data1 = reshape(frame_data(:,1,1,:) - average_chirp(:,1,1,:), 4, 256);
        single_chirp_data2 = reshape(frame_data(:,2,1,:) - average_chirp(:,2,1,:), 4, 256);
        single_chirp_data = cat(1, single_chirp_data1, single_chirp_data2)';
        fft_res1 = fftshift(fft2(single_chirp_data, 256, 180), 2);
        fft_res1 = fft_res1(min_didx:max_didx, :);
        
        maxv = -1;
        maxd = 0;
        max_theta = 0;
        max_phi = 0;
        for i=1:1:max_didx-min_didx+1
            for j=1:1:180
                if abs(fft_res1(i,j)) > maxv
                    maxd = i;
                    max_theta = j;
                    maxv = abs(fft_res1(i,j));
                end
            end
        end


        %get distance
        target_d = sample_rate*(maxd+min_didx-1)/N_FFTD*c/2/freq_slope;

        %get azimuth angle
        t_theta = acos(2 * (max_theta-90) / N_FFTT);

        %prepare data for elevation fft
        single_chirp_data1 = reshape(frame_data(3,1,1,:) - average_chirp(3,1,1,:), 1, 256);
        single_chirp_data2 = reshape(frame_data(1,3,1,:) - average_chirp(1,3,1,:), 1, 256);
        input_data = zeros(256,2);
        input_data(:,1) = single_chirp_data1;
        input_data(:,2) = single_chirp_data2;
        fft_res2 = fftshift(fft2(input_data, 256, 180), 2);
        fft_res2 = fft_res2(min_didx:max_didx, :);
        maxv = -1;
        max_phi = 0;
        for j=1:1:180
            if abs(fft_res2(maxd,j)) > maxv
                max_phi = j;
                maxv = abs(fft_res2(maxd,j));
            end
        end
        t_phi = -asin(2 * (max_phi-90) / N_FFTT);
        
        %calculate the error
        t_x = [t_x target_d*cos(t_phi)*cos(t_theta)];
        t_y = [t_y target_d*cos(t_phi)*sin(t_theta)];
        t_z = [t_z target_d*sin(t_phi)];
    end
    toc
    predict_xyz = [t_x; t_y; t_z];
    output_dir = strcat('./result/',num2str(sequence),'.mat');
    mkdir('./result/')
    save(output_dir, 'predict_xyz')

end