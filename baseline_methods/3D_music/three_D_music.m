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


debug = 0
b = zeros(31,2);
% b_2 = zeros(121,2);
a = zeros(121,4);
s = zeros(31,256);
if debug == 1
    frames = 1
end
n_target = 1
L = 2;
N = 4;
M = 256;
errors = [];
errors_xyz = [];

for phi = 0:1:30
    for phi_i = 1:1:2
        b(phi+1, phi_i) = exp(-1i*(phi_i-1) * 2 * pi * sin((phi+75-90)/180*pi) * d / wave_length);
    end
end

for theta = 0:1:120
    for theta_i=1:1:4
        a(theta + 1,theta_i) = exp(1i*(theta_i-1)*2*pi*cos((theta+30)/180*pi)*d/wave_length);
    end
end
for rl=0:1:30
    for s_i = 1:1:256
        s(rl +1, s_i) = exp(1i*4*pi*(rl+10)*bandwidth/c/ramp_end_time*(s_i-1)/10*1/sample_rate);
    end
end

% for phi = 0:1:120
%     for phi_i = 1:1:2
%         b_2(phi+1, phi_i) = exp(-1i*(phi_i-1) * 2 * pi * sin((phi+30-90)/180*pi) * d / wave_length);
%     end
% end


test_sequences = csvread('../../test_sequences.csv');

delete(gcp('nocreate'))
%to enable parallel computing
%parpool(16)

raw_data_dir = %raw data dir here

for sequence_id = 1:1:10
    sequence = test_sequences(sequence_id)
    filename = strcat(raw_data_dir,num2str(sequence),"\adc_data_0.bin");
    
    data = load_mm_raw(filename, rx_num, tx_num, chirp_loops, frames, samples_per_chirp);

    target_polar = [];
    target_position = [];
    t_z = [];
    t_x = [];
    t_y = [];
    tic
    for frame_id=1:1:frames
        frame_id
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
        %single_chirp_data = reshape(frame_data(:,1,1,:), 4, 256);
        %n_target = 1
        %single_chirp_data = reshape(input_data(1,:), 4, 256);
        Y = zeros(1, L*N*M);
        %Y = reshape(permute(input_data,  [1 3 2]), 1, 8*256);
        %Y = reshape(single_chirp_data', 1, 4*256)';
        for i=1:1:L*N*M
            Y(i) = input_data(floor((i-1)/M/N)+1, floor((mod(i-1,M*N))/M) + 1, mod(i-1, M)+1);
        end
        C = Y'*Y;
        [V, D] = eig(C);
        Q = V(:,1:(L*N*M-n_target));
        test = Q*Q';
        P = zeros(31, 121, 31);
        
        maxv = -1;
        amaxd = 0;
        amaxr = 0;
        amaxb = 0;
        for rl=0:30
            tmp = zeros(121,31);
            for theta = 0:1:120
                for phi = 0:1:30
                    b_i = b(phi+1, :);
                    a_i = a(theta+1, :);
                    s_i = s(rl+1, :);
                    V_R_theta = kron(b_i, kron(a_i,s_i));
                    tmp(theta+1, phi+1) = 1/(V_R_theta * test * V_R_theta');
                end
            end
            P(rl+1, :, :) = tmp;
        end

        for rl=0:30
            for theta = 0:1:120
                for phi = 0:1:30
                    if abs(P(rl+1, theta+1, phi+1)) > maxv
                        maxv = abs(P(rl+1, theta+1, phi+1));
                        amaxd = rl;
                        amaxr = theta;
                        amaxb = phi;
                    end
                end
            end
        end
        
        d_target = (amaxd+10) / 10;
        r_target = amaxr + 30;
        b_target = amaxb + 75-90;
        
        t_z = [t_z d_target * sin(b_target/180*pi)];
        t_y = [t_y d_target * cos(b_target/180*pi) * sin(r_target / 180 *pi)];
        t_x = [t_x d_target * cos(b_target/180*pi) * cos(r_target / 180 * pi)];

    end
    toc
    predict_xyz = [t_x; t_y; t_z]
    output_dir = strcat('./result/',num2str(sequence),'.mat');
    mkdir('./result/')
    save(output_dir, 'predict_xyz')
end