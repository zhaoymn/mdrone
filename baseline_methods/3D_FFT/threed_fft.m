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

test_sequences = csvread('../../test_sequences.csv');

raw_data_dir = %raw data dir here

for sequence_id = 1:1:1
    sequence_id
    sequence = test_sequences(sequence_id);
    filename = strcat(raw_data_dir,num2str(sequence),"\adc_data_0.bin");
    data = load_mm_raw(filename, rx_num, tx_num, chirp_loops, frames, samples_per_chirp);

    t_z = [];
    t_x = [];
    t_y = [];
    tic
    for frame_id=1:1:2
        %frame_id
        frame_data = reshape(data(:,:,:,frame_id,:), rx_num, tx_num, chirp_loops, samples_per_chirp);
        average_chirp = mean(frame_data, 3);
        single_chirp_data = reshape(frame_data(:,:,1,:) - average_chirp(:,:,1,:), 4,3, 256);
        input_data = zeros(2,4,256);
        input_data(1,1,:) = single_chirp_data(3,1,:);
        input_data(1,2,:) = single_chirp_data(4,1,:);
        input_data(1,3,:) = single_chirp_data(1,2,:);
        input_data(1,4,:) = single_chirp_data(2,2,:);
        input_data(2,1,:) = single_chirp_data(1,3,:);
        input_data(2,2,:) = single_chirp_data(2,3,:);
        input_data(2,3,:) = single_chirp_data(3,3,:);
        input_data(2,4,:) = single_chirp_data(4,3,:);
        input_data = permute(input_data, [3 2 1]);
        fft_res = fftn(input_data, [256 180 180]);
        fft_res = fft_res(min_didx:max_didx, :, :);
        fft_res = cat(2, fft_res(:, 91:180, :), fft_res(:, 1:90, :));
        fft_res = cat(3, fft_res(:, :, 91:180), fft_res(:,:, 1:90));

        maxv = -1;
        maxd = 0;
        max_theta = 0;
        max_phi = 0;
        for i=1:1:max_didx-min_didx+1
            for j=1:1:180
                for k=1:1:180
                    if abs(fft_res(i,j,k)) > maxv
                        maxd = i;
                        max_theta = j;
                        max_phi = k;
                        maxv = abs(fft_res(i,j,k));
                    end
                end
            end
        end

        target_d = sample_rate*(maxd+min_didx-1)/N_FFTD*c/2/freq_slope;
        t_theta = acos(2 * (max_theta-90) / N_FFTT);
        t_phi = -asin(2 * (max_phi-90) / N_FFTT);
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