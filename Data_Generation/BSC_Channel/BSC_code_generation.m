%% =======================================================================
%  BCH data generator with exact min-weight h in C1^⊥ \ C2^⊥
%  Requirements: MATLAB + Communications Toolbox (for bchgenpoly, cyclgen, gf)
%
%  WHAT THIS SCRIPT DOES (high level):
%    1) Takes your inputs (n, k1, k2, N, crossover_probs).
%    2) Builds BCH codes C1 and C2: generator (G) and parity-check (H) matrices.
%    3) Searches the ENTIRE rowspace of H1 to find the TRUE minimum-weight
%       vector h that lies in C1^⊥ but NOT in C2^⊥ (i.e., h*G2' ≠ 0).
%    4) Generates N random codewords from each code, passes them through a BSC
%       for each p in crossover_probs, and writes outputs to CSVs.
%
%  OUTPUT FILES (created under out/):
%    - h_n<31>_k1_<k1>_k2_<k2>.csv : the selected h vector (1 x n)
%    - h_weight.txt                : w(h) (a single integer)
%    For each p in crossover_probs:
%      - bsc_p<p>_C1_n<n>_k<k1>.csv : N x n noisy codewords from C1
%      - bsc_p<p>_C2_n<n>_k<k2>.csv : N x n noisy codewords from C2
%      - h_mult_p<p>_C1.csv         : N x 1 inner products <c,h> for C1 rows
%      - h_mult_p<p>_C2.csv         : N x 1 inner products <c,h> for C2 rows
%
%  Notes:
%   - The min-weight search enumerates all 2^(n-k1)-1 combinations; for n=31,k1=11,
%     this is ~1.05M candidates — feasible in MATLAB when implemented with logicals.
%   - For huge N, files can be very large; adjust 'chunk' to fit your RAM/IO budget.
% ========================================================================

%% ===== 1) USER INPUTS ===================================================
% INPUT: Code length n. For primitive narrow-sense BCH, n must be 2^m - 1.
n  = 31;

% INPUT: Dimensions (message lengths) of codes C1 and C2.
%        - k1 is dim(C1); k2 is dim(C2).
k1 = 11;
k2 = 6;

% INPUT: Number of codewords per code to generate for each p.
N  = 6_000_000;

% INPUT: Vector of BSC crossover probabilities (each will produce its own files).
crossover_probs = [0.001 0.005 0.01 0.05];

% INPUT: Output directory where all CSV files will be written.
outdir = "out";
if ~exist(outdir,"dir"), mkdir(outdir); end

% INPUT: Chunk size for streaming generation to avoid large memory spikes.
%        Each loop generates 'b' rows at a time per code.
chunk = 100000;

%% ===== 2) BASIC SANITY CHECKS ===========================================
% Check n matches 2^m - 1 (primitive BCH length).
m = round(log2(n+1));
assert(2^m - 1 == n, "For primitive BCH, n must be 2^m - 1. Given n=%d is invalid.", n);

% Check k1, k2 are in valid range.
assert(0 < k1 && k1 < n && 0 < k2 && k2 < n, "k1,k2 must be in (0,n).");

%% ===== 3) BUILD BCH CODES (G, H) =======================================
% INPUTS HERE:
%   - n, k1, k2
% USED FUNCTIONS:
%   - bchgenpoly: returns generator polynomial for BCH(n,k)
%   - cyclgen   : converts generator polynomial to generator/parity-check matrices
% OUTPUTS:
%   - G1,H1 for C1; G2,H2 for C2 (binary matrices)
gp1 = bchgenpoly(n, k1);                      % generator polynomial for C1
gp2 = bchgenpoly(n, k2);                      % generator polynomial for C2
[G1b, H1b] = cyclgen(n, gp1);                 % binary G,H for C1 from polynomial
[G2b, H2b] = cyclgen(n, gp2);                 % binary G,H for C2 from polynomial

% Reduce G matrices to full-rank bases (rowspace unchanged, removes dependent rows).
% We keep them as gf objects to do mod-2 algebra conveniently.
G1 = gf(rref2(G1b), 2); G1 = G1(any(G1.x,2), :);   % size(G1) should be [k1, n]
G2 = gf(rref2(G2b), 2); G2 = G2(any(G2.x,2), :);   % size(G2) should be [k2, n]

% Keep H1 as gf for mod-2 operations; we only need H1 to construct C1^⊥.
H1 = gf(H1b, 2);                                   % size(H1) = [n-k1, n]

fprintf('C1: [n=%d, k=%d], C2: [n=%d, k=%d]\n', n, size(G1,1), n, size(G2,1));

%% ===== 4) FIND EXACT MIN-WEIGHT h IN C1^⊥ \ C2^⊥ =======================
% GOAL: Find h with minimum Hamming weight such that:
%   (a) h ∈ C1^⊥  (i.e., h is a binary combination of H1 rows)
%   (b) h ∉ C2^⊥  (equivalently, h * G2' ≠ 0)
%
% INPUTS:
%   - H1 : parity-check matrix of C1 (rowspace(H1) = C1^⊥)
%   - G2 : generator matrix of C2 (used to test if a vector lies in C2^⊥)
% OUTPUTS:
%   - h  : 1 x n binary vector (gf(…,2)) meeting (a) and (b)
%   - wh : scalar, weight(h)
[h, wh] = minweight_h(H1, G2);

fprintf('Chosen h has weight %d\n', wh);

% Save h and its weight for reproducibility / later runs.
writematrix(h.x, fullfile(outdir, sprintf('h_n%d_k1_%d_k2_%d.csv', n, size(G1,1), size(G2,1))));
fid = fopen(fullfile(outdir, 'h_weight.txt'), 'w'); fprintf(fid, '%d\n', wh); fclose(fid);

%% ===== 5) GENERATE DATA FOR EACH p (STREAMED TO DISK) ===================
% For each BSC probability p, we:
%   - generate N random messages for C1 and C2 (in chunks),
%   - encode to codewords via msg*G (GF(2) multiplication),
%   - pass through BSC(p) by adding Bernoulli(p) noise mod 2,
%   - compute inner products <c,h> (scalar 0/1 per row),
%   - append results to CSV files.
for p = crossover_probs
    fprintf('\n=== p = %.3g ===\n', p);

    % OUTPUT FILE PATHS for this p:
    fC1 = fullfile(outdir, sprintf('bsc_p%.3g_C1_n%d_k%d.csv', p, n, size(G1,1)));
    fC2 = fullfile(outdir, sprintf('bsc_p%.3g_C2_n%d_k%d.csv', p, n, size(G2,1)));
    fH1 = fullfile(outdir, sprintf('h_mult_p%.3g_C1.csv', p));
    fH2 = fullfile(outdir, sprintf('h_mult_p%.3g_C2.csv', p));

    % Fresh files per p.
    if exist(fC1,"file"), delete(fC1); end
    if exist(fC2,"file"), delete(fC2); end
    if exist(fH1,"file"), delete(fH1); end
    if exist(fH2,"file"), delete(fH2); end

    left = N;
    while left > 0
        % b = number of rows to generate in this iteration (<= chunk)
        b = min(chunk, left);

        % --------- C1 PIPELINE (INPUT -> OUTPUTS) -------------------------
        % INPUTS for this block:
        %   - b: number of messages to generate in this chunk
        %   - G1: k1 x n generator of C1
        %   - p: BSC probability
        % STEPS:
        %   1) msg1: b x k1 i.i.d. Bernoulli(0.5) messages (GF(2))
        %   2) cw1 = msg1 * G1 mod 2  (b x n codewords)
        %   3) BSC noise: flip bits with prob p → cw1 = cw1 ⊕ noise
        %   4) ip1 = <cw1(i,:), h> mod 2 for each row i  (b x 1)
        % OUTPUTS appended to disk:
        %   - cw1 rows to bsc_p…_C1…csv
        %   - ip1 rows to h_mult_p…_C1.csv
        msg1 = gf(randi([0 1], b, size(G1,1)), 2);             % b x k1
        cw1  = mod(msg1 * G1, 2);                               % b x n
        if p > 0
            noise1 = gf(rand(b, n) < p, 2);                     % b x n
            cw1    = mod(cw1 + noise1, 2);                      % b x n
        end
        ip1 = mod(double(cw1.x) * double(h.x'), 2);             % b x 1

        % --------- C2 PIPELINE (INPUT -> OUTPUTS) -------------------------
        % Same steps with G2 (k2 x n). Outputs go to C2-specific files.
        msg2 = gf(randi([0 1], b, size(G2,1)), 2);             % b x k2
        cw2  = mod(msg2 * G2, 2);                               % b x n
        if p > 0
            noise2 = gf(rand(b, n) < p, 2);                     % b x n
            cw2    = mod(cw2 + noise2, 2);                      % b x n
        end
        ip2 = mod(double(cw2.x) * double(h.x'), 2);             % b x 1

        % --------- APPEND TO CSV (side-effect: writes to disk) ------------
        % NOTE: We write numeric arrays (double 0/1 from gf.x) for portability.
        writematrix(cw1.x, fC1, 'WriteMode', 'append');         % N-by-n across loop
        writematrix(cw2.x, fC2, 'WriteMode', 'append');
        writematrix(ip1,   fH1, 'WriteMode', 'append');         % N-by-1 across loop
        writematrix(ip2,   fH2, 'WriteMode', 'append');

        left = left - b;                                        % progress
    end

    fprintf('Saved -> %s\n          %s\n          %s\n          %s\n', fC1, fC2, fH1, fH2);
end

fprintf('\nDone.\n');

%% ========================= LOCAL FUNCTIONS ==============================
function R = rref2(A)
% rref2: Row-reduced echelon form over GF(2).
% INPUT:
%   - A : m x n binary matrix (double 0/1)
% OUTPUT:
%   - R : m x n binary matrix with same rowspace as A, reduced mod 2
    A = mod(A, 2);
    [m, n] = size(A);
    i = 1; j = 1;
    while i <= m && j <= n
        % Find pivot row at/after i in column j
        [~, p] = max(A(i:m, j)); p = p + i - 1;
        if A(p, j) == 0
            j = j + 1; continue;            % no pivot in this column
        end
        % Swap pivot row to position i
        if p ~= i, A([i p], :) = A([p i], :); end
        % Zero-out all other entries in column j (mod 2)
        for r = [1:i-1, i+1:m]
            if A(r, j), A(r, :) = mod(A(r, :) + A(i, :), 2); end
        end
        i = i + 1; j = j + 1;
    end
    R = A;
end

function [h, wh] = minweight_h(H1, G2)
% minweight_h: EXACT minimum-weight h ∈ C1^⊥ \ C2^⊥ by enumerating rowspace(H1).
% WHY THIS WORKS:
%   - Any h ∈ C1^⊥ is a binary linear combination of rows of H1.
%   - h ∈ C2^⊥ iff h*G2' == 0 (orthogonal to every generator row of C2).
%   - So we check all nonzero combinations, reject those with h*G2'==0,
%     and keep the smallest Hamming weight among the rest.
%
% INPUT:
%   - H1 : (n-k1) x n gf(…,2) parity-check of C1; rowspace(H1) = C1^⊥
%   - G2 : (k2) x n gf(…,2) generator of C2 (for orthogonality test)
% OUTPUT:
%   - h  : 1 x n gf(…,2) vector (min weight in C1^⊥ \ C2^⊥)
%   - wh : scalar = sum(h) = weight(h)

    % Convert to logical for fast bit math
    H   = logical(H1.x);                  % (n-k1) x n
    GT2 = logical(G2.x.');                % n x k2
    rH  = size(H, 1);                     % n-k1
    n   = size(H, 2);

    best_w = n + 1; best_h = [];

    % Enumerate all nonzero combinations of H's rows: 1..2^(n-k1)-1
    % Each 'mask' chooses which rows are summed mod 2.
    total = uint32(2^rH);
    for mask = uint32(1):total-1
        % Decode mask into a logical index over rows 1:rH
        rows = bitget(mask, 1:rH);        % 1 x rH logical
        % Sum selected rows mod 2 -> candidate h (1 x n)
        v = mod(sum(H(rows, :), 1), 2);

        % Check NOT in C2^⊥  <=>  v * G2' ≠ 0 (at least one 1 in result)
        if any(mod(v * GT2, 2))
            wv = sum(v);                  % Hamming weight
            if wv < best_w
                best_w = wv; best_h = v;
                if best_w == 1            % cannot beat weight 1
                    break;
                end
            end
        end
    end

    % Convert back to gf(…,2) row vector for downstream GF(2) ops
    h  = gf(best_h, 2);
    wh = best_w;

    % If best_h is empty, codes are nested too tightly (rare for these params)
    if isempty(best_h)
        error('No h found in C1^⊥ \\ C2^⊥. Check (n,k1,k2) or code construction.');
    end
end
