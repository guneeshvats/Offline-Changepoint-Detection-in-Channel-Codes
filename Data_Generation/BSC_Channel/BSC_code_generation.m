clc; clear;

n = 15;
k1 = 5;
k2 = 7;
N = 200;
crossover_probs = [0.1, 0.2, 0.5];

% Generate messages
msgTx1 = gf(randi([0 1], N, k1));
msgTx2 = gf(randi([0 1], N, k2));

% Encode using BCH
codewords1 = bchenc(msgTx1, n, k1);  % GF object
codewords2 = bchenc(msgTx2, n, k2);  % GF object

% Get binary representations
cw1_bin = double(codewords1.x);
cw2_bin = double(codewords2.x);

%%%%%%%%%%%%%%%%%%%
% Select valid h_prime vector from difference space
dualC1 = dualCode(cw1_bin);  % full dual code of C1
dualC2 = dualCode(cw2_bin);  % full dual code of C2
fprintf('dual of C1 = ');
disp(dualC1);
disp(size(unique(dualC1, 'rows')))
fprintf('dual of C2 = ');
disp(dualC2);
valid_h_candidates = setdiff(dualC1, dualC2, 'rows');
valid_h_candidates(~any(valid_h_candidates, 2), :) = [];

% Find minimum Hamming weight vector
hamming_weights = sum(valid_h_candidates, 2);
[~, min_idx] = min(hamming_weights);
h_prime = valid_h_candidates(min_idx, :);
fprintf('h_prime = ');
disp(h_prime);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for p = crossover_probs
    % Add BSC noise to both sets of codewords
    noisy_codewords1 = bsc(cw1_bin, p);  
    noisy_codewords2 = bsc(cw2_bin, p);

    % Save noisy codewords
    writematrix(noisy_codewords1, sprintf('bsc_p%.2f_codewords1.csv', p));
    writematrix(noisy_codewords2, sprintf('bsc_p%.2f_codewords2.csv', p));
    fprintf('Saved: bsc_p%.2f_codewords1.csv and codewords2.csv\n', p);
    
    % Multiply each noisy codeword with h_prime
    multiplied_h1 = mod(noisy_codewords1 * h_prime', 2);
    multiplied_h2 = mod(noisy_codewords2 * h_prime', 2);

    writematrix(multiplied_h1, sprintf('h_mult_p%.2f_codewords1.csv', p));
    writematrix(multiplied_h2, sprintf('h_mult_p%.2f_codewords2.csv', p));
    fprintf('Saved: h_mult_p%.2f_codewords1.csv and codewords2.csv\n', p);

    % Compute distances for CPD analysis
    [distances_H1, distances_H2] = compute_distances(noisy_codewords1, noisy_codewords2, n, k1, k2);

    writematrix(distances_H1, sprintf('distances_p%.2f_H1.csv', p));
    writematrix(distances_H2, sprintf('distances_p%.2f_H2.csv', p));
    fprintf('Saved: distances_p%.2f_H1.csv and H2.csv\n', p);

end

 function [dist_H1, dist_H2] = compute_distances(noisy1, noisy2, n, k1, k2)
    % Decode using (n, k1) and (n, k2) BCH codes
    decoded_H1 = bchdec(gf(noisy1), n, k1);
    decoded_H2 = bchdec(gf(noisy2), n, k2);

    % Re-encode using (n, k1) and (n, k2) BCH codes
    reencoded_H1 = bchenc(decoded_H1, n, k1);
    reencoded_H2 = bchenc(decoded_H2, n, k2);

    % Compute squared Hamming distance for noisy1
    hamming_dist_noisy1_reencodedH1 = sum(xor(noisy1, double(reencoded_H1.x)));
    squared_dist_noisy1_reencodedH1 = hamming_dist_noisy1_reencodedH1.^2;

    % Noisy1 vs Reencoded_H2
    hamming_dist_noisy1_reencodedH2 = sum(xor(noisy1, double(reencoded_H2.x)));
    squared_dist_noisy1_reencodedH2 = hamming_dist_noisy1_reencodedH2.^2;

    % Subtract squared Hamming distance between noisy1 & reencoded_H2 from noisy1 & reencoded_H1
    dist_H1 = squared_dist_noisy1_reencodedH1 - squared_dist_noisy1_reencodedH2;

    % Noisy2 vs Reencoded_H1
    hamming_dist_noisy2_reencodedH1 = sum(xor(noisy2, double(reencoded_H1.x)));
    squared_dist_noisy2_reencodedH1 = hamming_dist_noisy2_reencodedH1.^2;

    % Noisy2 vs Reencoded_H2
    hamming_dist_noisy2_reencodedH2 = sum(xor(noisy2, double(reencoded_H2.x)));
    squared_dist_noisy2_reencodedH2 = hamming_dist_noisy2_reencodedH2.^2;

    % Subtract squared Hamming distance between noisy2 & reencoded_H2 from noisy2 & reencoded_H1
    dist_H2 = squared_dist_noisy2_reencodedH1 - squared_dist_noisy2_reencodedH2;
 end

function dual = dualCode(G)
    % dualCode - Computes the dual of a binary code over GF(2)
    % Input:
    %   G - matrix of binary codewords (each row is a codeword)
    % Output:
    %   dual - matrix where each row is a codeword in the dual space

    % Ensure binary input
    G = mod(G, 2);

    % Transpose to treat each row as an equation
    G = mod(rref_gf2(G), 2);
    [m, n] = size(G);

    % Identify pivot columns
    pivots = zeros(1, n);
    row_idx = 1;
    for col = 1:n
        if row_idx <= m && G(row_idx, col) == 1
            pivots(col) = 1;
            row_idx = row_idx + 1;
        end
    end

    % Null space dimension = n - rank
    free_vars = find(pivots == 0);
    r = length(free_vars);
    dual = zeros(r, n);

    % Construct null space vectors
    for i = 1:r
        v = zeros(1, n);
        v(free_vars(i)) = 1;

        % Back-substitute to solve for pivot vars
        row = 1;
        for col = 1:n
            if pivots(col) == 1
                v(col) = mod(-G(row, free_vars(i)), 2);
                row = row + 1;
            end
        end
        dual(i, :) = mod(v, 2);
    end
end
function R = rref_gf2(A)
    % Perform RREF over GF(2)
    [m, n] = size(A);
    A = mod(A, 2);
    lead = 1;
    for r = 1:m
        if lead > n
            break;
        end
        i = r;
        while A(i, lead) == 0
            i = i + 1;
            if i > m
                i = r;
                lead = lead + 1;
                if lead > n
                    break;
                end
            end
        end
        if lead > n
            break;
        end
        A([i, r], :) = A([r, i], :);
        for i = 1:m
            if i ~= r && A(i, lead) == 1
                A(i, :) = mod(A(i, :) + A(r, :), 2);
            end
        end
        lead = lead + 1;
    end
    R = A;
end
