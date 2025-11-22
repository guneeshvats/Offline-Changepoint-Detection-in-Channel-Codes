clc;
close all;
clear all;

%=========================================================================

% bchnumerr(15)
% genpoly1 = bchgenpoly(15,11)
% genpoly2 = bchgenpoly(15,7)
% genpoly3 = bchgenpoly(15,5)

%=========================================================================
% Code-1: BCH(n=15,k1=11,t1=1)
% Code-2: BCH(n=15,k1=7,t2=2)

% n = 15;
% % 
% k1 = 11;
% t1 = 1;
% g1 = [1, 0, 0, 1, 1];
% % 
% k2 = 7;
% t2 = 2;
% g2 = [1, 1, 1, 0, 1, 0, 0, 0, 1];

%k2 = 5;
%t2 = 3;
%g2 = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1];


%=========================================================================
% Code-1: BCH(n=31,k1=11,t1=1)
% Code-2: BCH(n=31,k1=6,t2=2)

n = 31;

k1 = 11;
t1 = 1;
g1 = [1,0,1,1,0,0,0,1,0,0,1,1,0,1,1,0,1,0,1,0,1];

k2 = 6;
t2 = 2;
g2 = [1,1,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,1,1,1];

%k2 = 5;
%t2 = 3;
%g2 = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1];


%==========================================================================
[H1 G1] = cyclgen(n,g1);
[H2 G2] = cyclgen(n,g2);

%==========================================================================

% ----- Find h1 \in C1_perp but h1 not in C2_perp --------------------------

h1 = ones(1,n);

for m = 1:2^(n-k1)-1
    m_vec = func_convert_dec_to_binary(m,n-k1);
    v_vec = encode(m_vec, n, n-k1, 'linear', H1);
    s_vec = mod(v_vec*G2',2);    

    if (sum(s_vec) ~= 0) && sum(v_vec) <= sum(h1)
        h1 = v_vec;
    end

end

h1

% ----- Find h2 \in C2_perp but h2 not in C1_perp --------------------------

h2 = ones(1,n);

for m = 1:2^(n-k2)-1
    m_vec = func_convert_dec_to_binary(m,n-k2);
    v_vec = encode(m_vec, n, n-k2, 'linear', H2);
    s_vec = mod(v_vec*G1',2);    

    if (sum(s_vec) ~= 0) && sum(v_vec) <= sum(h2)
        h2 = v_vec;
    end
end
% -------------------------------------------------------------------------

% h_min = h1;
% 
% if sum(h2) <= sum(h1)
%     h_min = h2;
% end
% 
% h_min
% weight_h_min = sum(h_min)

% -------------------------------------------------------------------------
% Decide which one won, and report the source
w1 = sum(h1);
w2 = sum(h2);

if w1 < w2
    h_min = h1; 
    weight_h_min = w1; 
    min_source = 'h1';
elseif w2 < w1
    h_min = h2; 
    weight_h_min = w2; 
    min_source = 'h2';
else
    % tie
    h_min = h1;          % either is fine
    weight_h_min = w1; 
    min_source = 'both (tie)';
end

h_min
weight_h_min
disp(['min came from: ' min_source])

%==========================================================================

function bin_vec = func_convert_dec_to_binary(num,n)

    u = zeros(1,n);

    for j = 0:n-1
		u(1,n-j) = rem(num,2);
		num = floor(num/2);
    end

	bin_vec = u;

end
