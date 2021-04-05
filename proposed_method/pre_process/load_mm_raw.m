function adcData = load_mm_raw(data_path, rx_num, tx_num, chirp_loops, frames, samples_per_chirp)
%LOAD_MM_RAW Summary of this function goes here
%   Detailed explanation goes here
    %data_dir = "/home/zhao/mm_drone/data/";
    %chirps_per_frame = 128;
    %tx_num = 3;
    %rx_num = 4;
    %samples_per_chirp=256;
    %frames = 300;
    numLanes = 2;

    %chirp_loops = 128;


    %fileID = fopen(strcat(data_dir, "adc_data_", num2str(0),'.bin'));
    %fileID = fopen(strcat(data_dir, "adc_data.bin"));
    fileID = fopen(data_path);

    A = fread(fileID, 'int16');

    filesize = size(A,1);

    counter = 1;
    numchirps = frames * chirp_loops * tx_num;% filesize/2/samples_per_chirp/rx_num;
    filesize = 2*samples_per_chirp*rx_num*numchirps;
    LVDS = zeros(1,filesize/2);
    for i=1:4:filesize-1
        LVDS(1,counter)=A(i)+sqrt(-1)*A(i+2);
        LVDS(1,counter+1)=A(i+1)+sqrt(-1)*A(i+3);
        counter = counter + 2;
    end
    LVDS = reshape(LVDS, samples_per_chirp*rx_num, numchirps);
    %figure
    %plot(real(LVDS(1:256,1)))
    %LVDS = LVDS.';


    adcData = zeros(rx_num, tx_num, chirp_loops, frames, samples_per_chirp);
    for rx=1:rx_num
        for tx = 1:tx_num
            for l=1:chirp_loops
                for f= 1:frames
                    adcData(rx, tx, l, f, :) = LVDS((rx-1)*samples_per_chirp + 1:rx*samples_per_chirp, ((f-1)*chirp_loops*tx_num + (l - 1)*tx_num + (tx-1)) + 1);
                end
            end
        end
    end
end

