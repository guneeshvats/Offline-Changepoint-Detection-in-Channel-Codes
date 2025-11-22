clc;
clear all;
close all;

p_vals = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]; % Bit Flip Probability values
N = 1000000; % No. of codewords in file % pair = 1 uses 648 blocklength PCMs, 0 uses 324 blocklength PCMs

    P2 = [ 0 -1 -1 -1 0 0 -1 -1 0 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
        22 0 -1 -1 17 -1 0 0 12 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1
        6 -1 0 -1 10 -1 -1 -1 24 -1 0 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1
        2 -1 -1 0 20 -1 -1 -1 25 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1
        23 -1 -1 -1 3 -1 -1 -1 0 -1 9 11 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1
        24 -1 23 1 17 -1 3 -1 10 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1
        25 -1 -1 -1 8 -1 -1 -1 7 18 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1
        13 24 -1 -1 0 -1 8 -1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1
        7 20 -1 16 22 10 -1 -1 23 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1
        11 -1 -1 -1 19 -1 -1 -1 13 -1 3 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1
        25 -1 8 -1 23 18 -1 14 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0
        3 -1 -1 -1 16 -1 -1 2 25 5 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0];

    P1 = [16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
        25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
        25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
         9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
        24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
        2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0
        ];

blockSize = 27;
% Generating the sparse logic matrix from shift value matrix
pcmatrix1 = ldpcQuasiCyclicMatrix(blockSize, P1);
pcmatrix2 = ldpcQuasiCyclicMatrix(blockSize, P2);

% Initialising Encoder block
cfgLDPCEnc1 = ldpcEncoderConfig(pcmatrix1)
cfgLDPCEnc2 = ldpcEncoderConfig(pcmatrix2)

msgTx1 = randi([0 1],cfgLDPCEnc1.NumInformationBits,N,'double');
msgTx2 = randi([0 1],cfgLDPCEnc2.NumInformationBits,N,'double');

codeword1 = ldpcEncode(msgTx1, cfgLDPCEnc1);
codeword2 = ldpcEncode(msgTx2, cfgLDPCEnc2);

codewords1 = codeword1';
codewords2 = codeword2';

for p = p_vals
    noisy_codewords1 = bsc(codewords1, p);
    noisy_codewords2 = bsc(codewords2, p);

    writematrix(noisy_codewords1, sprintf('bsc_QSLDPC_p%.4f_codewords1.csv', p));
    writematrix(noisy_codewords2, sprintf('bsc_QSLDPC_p%.4f_codewords2.csv', p));
    fprintf('Saved: bsc_QSLDPC_sigma%.4f_codewords1.csv and codewords2.csv\n', p);
end

%--------------------------------------------------------
