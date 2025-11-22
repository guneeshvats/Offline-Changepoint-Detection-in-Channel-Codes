clc; clear; close all;

rng(42);  % reproducible

% ==== config ====
numframes_total = 1e6;          % total rows (codewords) per CSV; set to 5e6 if you really want
chunk_frames    = 2e4;          % rows per write (tune to your RAM/IO)
sigma_vals      = [0.2689, 0.3039, 0.3236, 0.3882, 0.4299];
% corresponding p (hard-decision BER) ≈ Q(1/σ): 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
% for .3236 ---- run again 
% ==== P1 and P2 (unchanged) ====
P1 = [16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
      25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
      25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
       9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
      24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
       2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0];

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

blockSize = 27;
H1 = ldpcQuasiCyclicMatrix(blockSize, P1);
H2 = ldpcQuasiCyclicMatrix(blockSize, P2);

cfg1 = ldpcEncoderConfig(H1);
cfg2 = ldpcEncoderConfig(H2);

% sanity: both should be 648
fprintf('BlockLength code1 = %d, code2 = %d\n', cfg1.BlockLength, cfg2.BlockLength);
Nblk = cfg1.BlockLength;  assert(Nblk == cfg2.BlockLength, 'Blocklength mismatch.');
K1   = cfg1.NumInformationBits;
K2   = cfg2.NumInformationBits;

for sigma = sigma_vals
    f1 = sprintf('awgnBPSK_QCLDPC_sigma%.4f_code1.csv', sigma);
    f2 = sprintf('awgnBPSK_QCLDPC_sigma%.4f_code2.csv', sigma);
    if exist(f1,'file'); delete(f1); end
    if exist(f2,'file'); delete(f2); end

    wrote = 0;
    firstWrite = true;

    while wrote < numframes_total
        take = min(chunk_frames, numframes_total - wrote);

        % ----- Code 1 -----
        m1  = randi([0 1], K1, take, 'logical');  % [K1 x take]
        cw1 = ldpcEncode(m1, cfg1).';             % [take x 648] bits
        x1  = 1 - 2*single(cw1);                  % BPSK float32
        y1  = x1 + single(sigma)*randn(size(x1), 'single');  % AWGN

        % ----- Code 2 -----
        m2  = randi([0 1], K2, take, 'logical');
        cw2 = ldpcEncode(m2, cfg2).';
        x2  = 1 - 2*single(cw2);
        y2  = x2 + single(sigma)*randn(size(x2), 'single');

        % ----- write/append to CSVs -----
        if firstWrite
            writematrix(y1, f1);  % creates file
            writematrix(y2, f2);
            firstWrite = false;
        else
            % If your MATLAB lacks 'WriteMode','append', use dlmwrite fallback below
            writematrix(y1, f1, 'WriteMode','append');
            writematrix(y2, f2, 'WriteMode','append');
            % Older MATLAB fallback (uncomment if needed):
            % dlmwrite(f1, y1, '-append');
            % dlmwrite(f2, y2, '-append');
        end

        wrote = wrote + take;
        fprintf('σ=%.4f: wrote %d/%d codewords to %s and %s (each row = %d cols)\n', ...
            sigma, wrote, numframes_total, f1, f2, Nblk);
    end

    fprintf('DONE σ=%.4f → %s, %s with %d rows × %d cols each\n', ...
        sigma, f1, f2, numframes_total, Nblk);
end
