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
minimum_distance = 0.5;
maximum_distance = 4.5;
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
window_length = 16;

processed_data_dir = %path to processed data dir (where you want to save the pre-processed data)
raw_data_dir = %path to raw data dir

for sequence = 1:1:100
    sequence
    mkdir(strcat(processed_data_dir, num2str(sequence), '\'));
    filename = strcat(raw_data_dir,num2str(sequence),"\adc_data_0.bin");
    data = load_mm_raw(filename, rx_num, tx_num, chirp_loops, frames, samples_per_chirp);
    
    tic
    for frame_id=1:1:600
        %[sequence, frame_id]
        frame_data = reshape(data(:,:,:,frame_id,:), rx_num, tx_num, chirp_loops, samples_per_chirp);
        
        
        one_d_fft = fft(frame_data, 256, 4);
        average = mean(one_d_fft, 3);
        one_d_fft = one_d_fft - average;
        
        %average_chirp = mean(frame_data, 3);
        d_data = permute(one_d_fft, [3, 1, 2, 4]);
        parsed_data = zeros(8,6,max_didx-min_didx+1, r_max - r_min +1);
        for c = 1:1:8
            single_chirp_data1 = reshape(mean(d_data((c-1)*window_length+1:c*window_length,:,1,:), 1), 4, 256);
            single_chirp_data2 = reshape(mean(d_data((c-1)*window_length+1:c*window_length,:,2,:), 1), 4, 256);

            single_chirp_data = cat(1, single_chirp_data1, single_chirp_data2)';
            fft_res_a1 = fftshift(fft(single_chirp_data, 180, 2), 2);
            parsed_data(c, 1, :,:) = fft_res_a1(min_didx:max_didx,r_min:r_max);
            
            single_chirp_data = reshape(mean(d_data((c-1)*window_length+1:c*window_length,:,3,:), 1), 4, 256);
            fft_res_a2 = fftshift(fft(single_chirp_data', 180, 2), 2);
            parsed_data(c, 2, :,:) = fft_res_a2(min_didx:max_didx,r_min:r_max);

            single_chirp_data = [reshape(mean(d_data((c-1)*window_length+1:c*window_length,3,1,:),1) , 1, 256); reshape(mean(d_data((c-1)*window_length+1:c*window_length,1,3,:),1) , 1, 256)];
            fft_res_e1 = fftshift(fft(single_chirp_data', 180, 2), 2);
            parsed_data(c, 3, :,:) = fft_res_e1(min_didx:max_didx,r_min:r_max);

            single_chirp_data = [reshape(mean(d_data((c-1)*window_length+1:c*window_length,4,1,:),1) , 1, 256); reshape(mean(d_data((c-1)*window_length+1:c*window_length,2,3,:),1) , 1, 256)];
            fft_res_e2 = fftshift(fft(single_chirp_data', 180, 2), 2);
            parsed_data(c, 4, :,:) = fft_res_e2(min_didx:max_didx,r_min:r_max);

            single_chirp_data = [reshape(mean(d_data((c-1)*window_length+1:c*window_length,1,2,:),1) , 1, 256); reshape(mean(d_data((c-1)*window_length+1:c*window_length,3,3,:),1) , 1, 256)];
            fft_res_e3 = fftshift(fft(single_chirp_data', 180, 2), 2);
            parsed_data(c, 5, :,:) = fft_res_e3(min_didx:max_didx,r_min:r_max);

            single_chirp_data = [reshape(mean(d_data((c-1)*window_length+1:c*window_length,2,2,:),1) , 1, 256); reshape(mean(d_data((c-1)*window_length+1:c*window_length,4,3,:),1) , 1, 256)];
            fft_res_e4 = fftshift(fft(single_chirp_data', 180, 2), 2);
            parsed_data(c, 6, :,:) = fft_res_e4(min_didx:max_didx,r_min:r_max);
        end
        %parsed_data = sign(parsed_data).*log10(abs(parsed_data)+1);
        %save(strcat('G:\processed_data\', num2str(sequence), '\', num2str(frame_id), '.mat'), 'parsed_data');
    end
    toc
end

