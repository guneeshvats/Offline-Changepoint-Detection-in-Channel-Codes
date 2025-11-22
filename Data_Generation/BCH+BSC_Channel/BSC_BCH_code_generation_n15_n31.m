%% =======================================================================
%  BCH codeword generator with progress (no h, no inner-product files)
%  - Builds BCH(n,k1) and BCH(n,k2)
%  - For each p in crossover_probs:
%      * encodes N random messages
%      * passes through BSC(p)
%      * writes ONLY noisy codeword CSVs for C1 and C2
%  - Prints per-chunk progress and ETA
% =======================================================================

%% 1) USER INPUTS
n  = 31;
k1 = 6;                 % target; actual k may differ (MATLAB tables)
k2 = 11;
N  = 3000000;          % rows per code per p
crossover_probs = [0.1, 0.2, 0.15, 0.08];
outdir = "n31_c1_c2"; if ~exist(outdir,"dir"), mkdir(outdir); end
chunk = 100000;          % increase if RAM allows (e.g., 5e5 or 1e6)
rng(0,'twister');        % reproducible

%% 2) SANITY CHECKS
m = round(log2(n+1)); assert(2^m - 1 == n, "n must be 2^m - 1.");
assert(0 < k1 && k1 < n && 0 < k2 && k2 < n, "k1,k2 must be in (0,n).");

%% 3) BUILD BCH CODES (G, H)
[gp1,~] = bchgenpoly(n, k1);
[gp2,~] = bchgenpoly(n, k2);
gen1 = double(gp1.x);            % cyclgen expects descending powers
gen2 = double(gp2.x);
[H1b, G1b] = cyclgen(n, gen1);
[H2b, G2b] = cyclgen(n, gen2);
G1 = gf(rref2(G1b),2); G1 = G1(any(G1.x,2),:);
G2 = gf(rref2(G2b),2); G2 = G2(any(G2.x,2),:);
fprintf('Actual: C1 [n=%d, k=%d], C2 [n=%d, k=%d]\n', n, size(G1,1), n, size(G2,1));

%% 4) GENERATE/WRITE PER p  (CODEWORDS ONLY, with progress)
for p = crossover_probs
    fprintf('\n=== p = %.3g ===\n', p);

    fC1 = fullfile(outdir, sprintf('bsc_p%.5g_C1_n%d_k%d.csv', p, n, size(G1,1)));
    fC2 = fullfile(outdir, sprintf('bsc_p%.5g_C2_n%d_k%d.csv', p, n, size(G2,1)));
    if exist(fC1,"file"), delete(fC1); end
    if exist(fC2,"file"), delete(fC2); end

    left = N;
    done = 0;
    t0 = tic;
    while left > 0
        b = min(chunk, left);

        % --- C1 ---
        msg1 = gf(randi([0 1], b, size(G1,1)), 2);
        cw1  = msg1 * G1;                      % no mod(...)
        if p > 0, cw1 = cw1 + gf(rand(b,n) < p, 2); end
        
        % --- C2 ---
        msg2 = gf(randi([0 1], b, size(G2,1)), 2);
        cw2  = msg2 * G2;                      % no mod(...)
        if p > 0, cw2 = cw2 + gf(rand(b,n) < p, 2); end

        % --- Append CSVs ---
        writematrix(cw1.x, fC1, 'WriteMode','append');
        writematrix(cw2.x, fC2, 'WriteMode','append');

        % --- Progress + ETA ---
        left = left - b;
        done = done + b;
        elapsed = toc(t0);
        rate = max(done/elapsed, eps);      % rows/sec
        eta_sec = (N - done) / rate;
        fprintf('  p=%.3g: %d/%d rows (%.1f%%) | %.1f ksps | ETA ~ %s\n', ...
            p, done, N, 100*done/N, rate/1e3, secs2hms(eta_sec));
    end

    fprintf('Saved -> %s\n          %s\n', fC1, fC2);
end
fprintf('\nDone.\n');

%% ========================= LOCAL FUNCTIONS ==============================
function R = rref2(A)
% rref2: Row-reduced echelon form over GF(2).
    A = mod(A, 2);
    [m, n] = size(A);
    i = 1; j = 1;
    while i <= m && j <= n
        [~, p] = max(A(i:m, j)); p = p + i - 1;
        if A(p, j) == 0, j = j + 1; continue; end
        if p ~= i, A([i p], :) = A([p i], :); end
        for r = [1:i-1, i+1:m]
            if A(r, j), A(r, :) = mod(A(r, :) + A(i, :), 2); end
        end
        i = i + 1; j = j + 1;
    end
    R = A;
end

function s = secs2hms(sec)
% secs2hms: pretty ETA string "H:MM:SS" for a number of seconds.
    if ~isfinite(sec) || sec < 0, s = '?:??:??'; return; end
    h = floor(sec/3600); sec = sec - 3600*h;
    m = floor(sec/60);   s2 = round(sec - 60*m);
    s = sprintf('%d:%02d:%02d', h, m, s2);
end
