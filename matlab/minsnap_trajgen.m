%% Single Segment

N = 7; % degree of polynomial to fit
snp = 4; % 4th derivative (snap) 
T = 2; % segment time length

% Minimum snap Hessian matrix (not used for fully constrained single segment)
Q = zeros(N+1); 
for i = snp:N+1
    for j = snp:N+1
        k = [0,1,2,3];
        %Q(i,j) = 2 * prod((i-k).*(j-k)) * T^(i+j-7) / (i+j-7);
        Q(i,j) = prod((i-k).*(j-k)) * T^(i+j-7) / (i+j-7);
    end
end

% Derivative constraint matrix
A0 = zeros(snp,N+1); % works because snp = (N+1)/2
AT = zeros(snp,N+1);
for i = 0:snp-1 % order of derivative
    A0(i+1,i+1) = factorial(i);
    for j = 0:N % order of polynomial term 
        if j >= i
            AT(i+1,j+1) = (factorial(j) / factorial(j-i)) * T^(j-i);
        end
    end
end
A = [A0; AT];

% go from 0 at t=0 to 10 at t=T
% with vel, acc, jerk zero at endpoints
d = zeros(2*snp,1);
d(snp+1) = 10;

p = A\d;

%% Multiple Segment

N = 7; % degree of polynomial to fit
n = 4; % 4th derivative (snap) 

M = 3; % number of segments
T = [1,2,1]; % segment time lengths 
W = [0,5,10,-5]; % waypoints
Tstamp = cumsum(T); % timestamps of waypoints
L = n*(M+1); % number of total derivative constraints (free and fixed)

% For each segment, compute Q and A
for m = 1:M

    % Minimum snap Hessian matrix Q
    Q{m} = zeros(N+1);
    for i = n:N+1
        for j = n:N+1
            k = [0,1,2,3];
            Q{m}(i,j) = prod((i-k).*(j-k)) * T(m)^(i+j-7) / (i+j-7);
        end
    end

    % Derivative constraint matrix A
    A{m} = zeros(N+1);
    for i = 0:n-1 % order of derivative
        A{m}(i+1,i+1) = factorial(i);
        for j = 0:N % order of polynomial term 
            if j >= i
                A{m}(i+n+1,j+1) = (factorial(j) / factorial(j-i)) * T(m)^(j-i);
            end
        end
    end
end

% Assemble block diagonal matrices Q1...M, A1...M 
QM = blkdiag(Q{:});
AM = blkdiag(A{:});

% Unconstrained minimization
H = inv(AM)'*QM*inv(AM);
cvx_begin
    variable d(L) % polynomial derivatives at keyframes 
    dxp = expand(d,n);
    minimize(quad_form(dxp,H))
    subject to
        for i = 0:M
            d(i*n+1) == W(i+1);  % M-th position waypoint
            if i==0 || i==M
                d(i*n+2:(i+1)*n) == 0; % constrain endpoint derivatives to 0
            end
        end
cvx_end

p = AM\dxp;     % find all coefficients
P = zeros(N+1,M); % separate into polynomials for each segment
for i = 1:M
    P(:,i) = flip(p((i-1)*(N+1)+1:i*(N+1)));
end

dt = 0.01;
T1 = 0:dt:T(1); T2 = 0:dt:T(2); T3 = 0:dt:T(3);
hold on
plot(T1,polyval(P(:,1),T1), T(1)+T2,polyval(P(:,2),T2), T(1)+T(2)+T3,polyval(P(:,3),T3));
scatter(Tstamp, W(2:end));

%% 3-D Trajectory

N = 7;
M = 2;
n = 4;
% T =   [1, 3, 2, 5, 2];
% Wx = [0, 3, 8, 4,-1, 2];
% Wy = [0,-2, 6, 4, 2,-4];
% Wz = [0, 7, 9, 6,-5, 2];
T =   [1, 3];
Wx = [0, 3, 8];
Wy = [0,-2, 6];
Wz = [0, 0, 0];

X = MinSnapTG(N,M,n,T,Wx);
Y = MinSnapTG(N,M,n,T,Wy);
Z = MinSnapTG(N,M,n,T,Wz);

dt = 0.01;
hold on
for i = 1:M
    Ti = 0:dt:T(i);
    plot3(polyval(X(:,i),Ti), polyval(Y(:,i),Ti), polyval(Z(:,i),Ti));
end
scatter3(Wx(2:end), Wy(2:end), Wz(2:end));

%% Utilities

d = [1:12]';
dxp = expand(d,4);

% Duplicates center blocks of array
function dxp = expand(d,n)
    L = length(d);
    drep = d(n+1:L-n);
    drep = reshape(drep,n,length(drep)/n);
    drep = repmat(drep,[2,1]);
    drep = reshape(drep,numel(drep),1);
    dxp = [d(1:n); drep; d(L-n+1:L)];
end


%% Function

% N - degree of polynomial to fit
% M - number of segments
% n - order derivative to minimize 
% T - segment time lengths 
% W - waypoints
function P = MinSnapTG(N,M,n,T,W)

    L = n*(M+1); % number of total derivative constraints (free and fixed)
    
    % For each segment, compute Q and A
    for m = 1:M

        % Minimizing derivative Hessian matrix Q
        Q{m} = zeros(N+1);
        for i = n:N+1
            for j = n:N+1
                k = 0:n-1;
                Q{m}(i,j) = prod((i-k).*(j-k)) * T(m)^(i+j+1-2*n) / (i+j+1-2*n);
            end
        end

        % Derivative constraint matrix A
        A{m} = zeros(N+1);
        for i = 0:n-1 % order of derivative
            A{m}(i+1,i+1) = factorial(i);
            for j = 0:N % order of polynomial term 
                if j >= i
                    A{m}(i+n+1,j+1) = (factorial(j) / factorial(j-i)) * T(m)^(j-i);
                end
            end
        end
    end

    % Assemble block diagonal matrices Q1...M, A1...M 
    QM = blkdiag(Q{:});
    AM = blkdiag(A{:});

    % Unconstrained minimization
    H = inv(AM)'*QM*inv(AM);
    cvx_begin
        variable d(L) % polynomial derivatives at keyframes 
        dxp = expand(d,n);
        minimize(quad_form(dxp,H))
        subject to
            for i = 0:M
                d(i*n+1) == W(i+1);  % M-th position waypoint
                if i==0 || i==M
                    d(i*n+2:(i+1)*n) == 0; % constrain endpoint derivatives to 0
                end
            end
    cvx_end

    p = AM\dxp;     % find all coefficients
    P = zeros(N+1,M); % separate into polynomials for each segment
    for i = 1:M
        P(:,i) = flip(p((i-1)*(N+1)+1:i*(N+1)));
    end
    % return P
end

