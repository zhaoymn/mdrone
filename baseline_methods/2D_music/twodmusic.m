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


a = zeros(121,8);
s = zeros(31,256);
b = zeros(31, 2);

for theta = 0:1:120
    for a_i=1:1:8
        a(theta+1, a_i) = exp(1i*(a_i-1)*2*pi*cos((theta+30)/180*pi)*d/wave_length);
    end
end

for rl=0:1:30
    for s_i = 1:1:256
        s(rl+1, s_i) = exp(1i*4*pi*(rl+10)*bandwidth/c/ramp_end_time*(s_i-1)/10*1/sample_rate);
    end
end

for theta = 0:1:30
    for a_i=1:1:2
        b(theta+1, a_i) = exp(1i*(a_i-1)*2*pi*sin((theta+75-90)/180*pi)*d/wave_length);
    end
end

test_sequences = csvread('../../test_sequences.csv');

raw_data_dir = %raw data dir here

for sequence_id = 5:1:10
    sequence = test_sequences(sequence_id)
    %sequence = sequence_id
    filename = strcat(raw_data_dir,num2str(sequence),"\adc_data_0.bin");
    data = load_mm_raw(filename, rx_num, tx_num, chirp_loops, frames, samples_per_chirp);

    n_target = 1;
    L = 2;
    N = 8;
    M = 256;
    t_z = [];
    t_x = [];
    t_y = [];
    for frame_id=1:1:frames
        frame_id
        frame_data = reshape(data(:,:,:,frame_id,:), rx_num, tx_num, chirp_loops, samples_per_chirp);
        average_chirp = mean(frame_data, 3);
        single_chirp_data = reshape(frame_data(:,:,1,:) - average_chirp(:,:,1,:), 4,3, 256);
        input_data = zeros(8,256);
        input_data(1,:) = single_chirp_data(1,1,:);
        input_data(2,:) = single_chirp_data(2,1,:);
        input_data(3,:) = single_chirp_data(3,1,:);
        input_data(4,:) = single_chirp_data(4,1,:);
        input_data(5,:) = single_chirp_data(1,2,:);
        input_data(6,:) = single_chirp_data(2,2,:);
        input_data(7,:) = single_chirp_data(3,2,:);
        input_data(8,:) = single_chirp_data(4,2,:);
        %single_chirp_data = reshape(frame_data(:,1,1,:), 4, 256);
        %n_target = 1
        Y = reshape(input_data, 1, 8*256)';
        C = Y*Y';
        e = eig(C);
        [V, D] = eig(C);
        Q = V(:,1:(N*M-n_target));
        test = Q*Q';
        P = zeros(31, 121);
        maxv = -1;
        amaxd = 0;
        amaxr = 0;
        for rl=0:1:30
            for theta = 0:1:120
                aa = a(theta+1, :);
                ss = s(rl+1, :);
                V_R_theta = kron(ss,aa);
                P(rl+1, theta+1) = 1/(V_R_theta * test * V_R_theta');
                if abs(P(rl+1, theta+1)) > maxv
                    maxv = abs(P(rl+1, theta+1));
                    amaxd = rl;
                    amaxr = theta;
                end
            end
        end

        %single_chirp_data = reshape(frame_data(:,1,1,:) - average_chirp(:,1,1,:), 4, 256);
        input_data = zeros(2,256);
        input_data(1,:) = single_chirp_data(1,3,:);
        input_data(2,:) = single_chirp_data(3,1,:);
        %single_chirp_data = reshape(frame_data(:,1,1,:), 4, 256);
        %n_target = 1
        Y = reshape(input_data', 1, 2*256)';
        C = Y*Y';
        e = eig(C);
        [V, D] = eig(C);
        Q = V(:,1:(L*M-n_target));
        test = Q*Q';
        maxv = -1;
        amaxb = 0;
        %for rl=0:1:30
        rl = amaxd;
        P = zeros(31, 31);
        for theta = 0:1:30
            bb = b(theta+1, :);
            ss = s(rl+1, :);
            V_R_theta = kron(bb,ss);
            P(rl+1, theta+1) = 1/(V_R_theta * test * V_R_theta');
            if abs(P(rl+1, theta+1)) > maxv
                maxv = abs(P(rl+1, theta+1));
                amaxb = theta;
            end
        end
        %end

        d_target = (amaxd+10) / 10;
        r_target = amaxr + 30;
        b_target = amaxb + 75 - 90;

        t_z = [t_z d_target * sin(b_target/180*pi)];
        t_y = [t_y d_target * cos(b_target/180*pi) * sin(r_target / 180 *pi)];
        t_x = [t_x d_target * cos(b_target/180*pi) * cos(r_target / 180 * pi)];

    end

    predict_xyz = [t_x; t_y; t_z]
    output_dir = strcat('./result/',num2str(sequence),'.mat');
    mkdir('./result/')
    save(output_dir, 'predict_xyz')
end