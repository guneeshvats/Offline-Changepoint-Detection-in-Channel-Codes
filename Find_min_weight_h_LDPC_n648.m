clc;
close all;
clear all;

    P1 = [16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
        25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
        25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
         9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
        24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
        2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0
        ]; % Do not change last n-k columns

    P2 = [-1 17 -1 -1 -1 -1  -1  -1   -1   -1   -1  -1  -1  -1   2  -1  -1  1   -1   -1  -1  -1  -1  7
        16  17  0  24  10   3  14  -1   4   2   7  -1  26  -1   2  -1  21   -1   1   -1  -1  -1  -1  -1
        25  12  12   -1   3  26   6  21  -1  15  22  -1  15  -1   4  -1  -1     -1   -1   1  -1  -1  -1 -1
        25  18  26  16  22  23   9  -1   0  -1   4  -1   4  -1   8  23  11  -1  -1  -1   1   -1   -1  -1
        9   7   0   1  17  -1  -1   7   3  -1   3  23  -1  16  -1  -1  21  -1   -1  -1  -1   1   -1  -1
        24   5  26   6   1  -1  -1  15  24  15  -1   8  -1  13  -1  14  -1   -1  -1  -1  -1  -1   1   -1
        2   2  19  14  24   1  15  19  -1  21  -1   2  -1  24  -1   2  -1   -1   -1  -1  -1  -1  -1   1];
        % Do not change last n-k columns

blockSize = 27;

pcmatrix1 = ldpcQuasiCyclicMatrix(blockSize,P1);
H1 = full(pcmatrix1); 
cfgLDPCEnc1 = ldpcEncoderConfig(pcmatrix1)
[reshapedH1, G1] = GenerateHandGfromLDPC(cfgLDPCEnc1);

pcmatrix2 = ldpcQuasiCyclicMatrix(blockSize,P2);
H2 = full(pcmatrix2); 
cfgLDPCEnc2 = ldpcEncoderConfig(pcmatrix2)
[reshapedH2, G2] = GenerateHandGfromLDPC(cfgLDPCEnc2);

[row_H1, col_H1] = size(H1);
[row_H2, col_H2] = size(H2);

%==========================================================================
h1 = ones(1,col_H1);

for i = 1:row_H1

    h_vec = H1(i,:);

    s_vec = mod(h_vec*G2',2);    

    if (sum(s_vec) ~= 0) && sum(h_vec) <= sum(h1)
        h1 = h_vec;
    end
end

sum(h1)

%==========================================================================
h2 = ones(1,col_H2);

for i = 1:row_H2

    h_vec = H2(i,:);

    s_vec = mod(h_vec*G1',2);    

    if (sum(s_vec) ~= 0) && sum(h_vec) <= sum(h2)
        h2 = h_vec;
    end

  

end

sum(h2)

% -------------------------------------------------------------------------

h_min = h1;

if sum(h2) <= sum(h1)
    h_min = h2;
end

h_min;
weight_h_min = sum(h_min)

%==========================================================================

function [reshapedH, G] = GenerateHandGfromLDPC(cfgLDPCEnc)

    pcmatrix = cfgLDPCEnc.ParityCheckMatrix; %Get the parity check matrix from the Encoder object
    
    parity = full(pcmatrix);   %Expand the matrix to full size
    H = g2rref(parity);    %Do a rref on the binary matrix.  Code for this function here: https://github.com/nnininnine/MATLAB/blob/master/g2rref.m
    meat = H(:,[(cfgLDPCEnc.NumParityCheckBits+1):end]);  %Get the non-identity matrix "meat" of the parity check matrix
       
    reshapedH = [meat eye(cfgLDPCEnc.NumParityCheckBits)]; %Reformat the parity check matrix to be in a standard form of [meat identityMatrix].
    G = [eye(cfgLDPCEnc.NumInformationBits) transpose(meat)]; %The generator is now [indentityMatrix meat']
    test = mod(reshapedH*transpose(G),2); %Test to see if we formed it correctly; test should be an all-0 matrix in GF(2) (i.e. binary)
    sum(test(:))  %Quick test of our test - this should be zero because all the elements of the test matrix should be zero

end

%==========================================================================

function [A] = g2rref(A)
    %G2RREF   Reduced row echelon form in gf(2).
    %   R = RREF(A) produces the reduced row echelon form of A in gf(2).
    %
    %   Class support for input A:
    %      float: with values 0 or 1
    %   Copyright 1984-2005 The MathWorks, Inc. 
    %   $Revision: 5.9.4.3 $  $Date: 2006/01/18 21:58:54 $

    [m,n] = size(A);

    % Loop over the entire matrix.
    i = 1;
    j = 1;

    while (i <= m) && (j <= n)
        % Find value and index of largest element in the remainder of column j.
        k = find(A(i:m,j),1) + i - 1;
        
        % Swap i-th and k-th rows.
        A([i k],j:n) = A([k i],j:n);
        
        % Save the right hand side of the pivot row
        aijn = A(i,j:n);
        
        % Column we're looking at
        col = A(1:m,j);
        
        % Never Xor the pivot row against itself
        col(i) = 0;
        
        % This builds an matrix of bits to flip
        flip = col*aijn;
        
        % Xor the right hand side of the pivot row with all the other rows
        A(1:m,j:n) = xor( A(1:m,j:n), flip );
        
        i = i + 1;
        j = j + 1;
    end
end