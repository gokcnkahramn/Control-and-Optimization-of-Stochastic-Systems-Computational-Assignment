%% Warning! To work those codes, plese run by section.
%% Q1
%% a.i
% Transition probabilities
Prob1good = 0.1; Prob1bad = 0.9; Prob0good = 0.9; Prob0bad = 0.1;
% Cost parameters
zeta = 0.75;
etas = [0.8, 0.6, 0.01]; % Our diffrent eta values
beta = 0.9; % This is our discount factor
max_iterations = 10; % Maximum number of iterations for value iteration

% Our states and action
X = {'B', 'G'};
U = {0, 1};

% Main code (Value Iteration)
for i = 1:length(etas)
    eta = etas(i);
    
    % Initialize value function and policy
    V = zeros(length(X), 1); % Initialize V to zeros
    policy = cell(length(X), 1);
    
    for iter = 1:max_iterations
        V_prev = V;

        for j = 1:length(X)
            x = X{j};
            Q = zeros(length(U), 1);

            for k = 1:length(U)
                u = U{k};
                Q(k) = -zeta*(strcmp(x, 'G') && u==1) + eta*u;
                for m = 1:length(X)
                    x_next = X{m};
                    Q(k) = Q(k) + beta*(Prob1good*V_prev(strcmp(X, x_next)) + Prob1bad*V_prev(strcmp(X, 'B')));
                end
            end

            [V(j), idx] = max(Q);
            policy{j} = U{idx};
        end

        if max(abs(V - V_prev)) < 1e-6
            break;
        end
    end
    
    % This is our results
    disp(['For eta = ', num2str(eta)]);
    disp('Optimal Policy:');
    disp(policy);
    disp('Optimal Value Function:');
    disp(V);
end



%% a.ii
% This part is as same as a.i
Prob1good = 0.1; Prob1bad = 0.9; Prob0good = 0.9; Prob0bad = 0.1;
zeta = 0.75;
etas = [0.8, 0.6, 0.01];
beta = 0.9; 
max_iterations = 10;

X = {'B', 'G'};
U = {0, 1};

% I initialized random policy
policy = cell(length(X), 1);
for i = 1:length(X)
    policy{i} = U{randi(2)};
end

% Main code (policy iteration)
for i = 1:length(etas)
    eta = etas(i);
    
    for iter = 1:max_iterations
        % It makes policy evaluation
        V = zeros(length(X), 1);
        for j = 1:max_iterations % I also use max_iteration for making policy evaluation
            V_prev = V;
            
            for j = 1:length(X)
                x = X{j};
                u = policy{j};
                Q = -zeta*(strcmp(x, 'G') && u==1) + eta*u;
                for k = 1:length(X)
                    x_next = X{k};
                    Q = Q + beta*(Prob1good*V_prev(strcmp(X, x_next)) + Prob1bad*V_prev(strcmp(X, 'B')));
                end
                
                V(j) = Q;
            end
            
            diff = max(abs(V - V_prev));
            if diff < 1e-6
                break;
            end
        end
        
        % Improvement of policy
        policy_stable = true;
        for j = 1:length(X)
            x = X{j};
            old_policy = policy{j};
            Q = zeros(length(U), 1);
            
            for k = 1:length(U)
                u = U{k};
                Q(k) = -zeta*(strcmp(x, 'G') && u==1) + eta*u;
                for m = 1:length(X)
                    x_next = X{m};
                    Q(k) = Q(k) + beta*(Prob1good*V(strcmp(X, x_next)) + Prob1bad*V(strcmp(X, 'B')));
                end
            end
            
            [V_max, idx] = max(Q);
            policy{j} = U{idx};
            
            if ~isequal(old_policy, policy{j})
                policy_stable = false;
            end
        end
        
        if policy_stable
            break;
        end
    end
    
    disp(['For eta = ', num2str(eta)]);
    disp('Optimal Policy:');
    disp(policy);
    disp('Optimal Value Function:');
    disp(V);
end

%% a.iii
% This initialization part is also as same as a.i and a.ii
beta = 0.9;
zeta = 0.75;
eta = 0.6;

X = {'B', 'G'};
U = [0, 1];

% This is our transition probabilities given
P = zeros(numel(X), numel(X), numel(U));
P(:,:,1) = [
    0.9, 0.1;
    0.9, 0.1;
];
P(:,:,2) = [
    0.1, 0.9;
    0.1, 0.9;
];

% Q-learning functions
num_episodes = 100;
Q = zeros(numel(X), numel(U)); % Q-function initialization

% Main function for Q-learning
for eps = 1:num_episodes
    % Initialize the state
    x_idx = randi(numel(X));
    x = X{x_idx};

    % Action based step chosen
    epsilon = 0.1;
    if rand < epsilon
        u_idx = randi(numel(U));
    else
        [~, u_idx] = max(Q(x_idx, :));
    end
    u = U(u_idx);

    % Updating
    cost = -zeta * (strcmp(x, 'G') && u == 1) + eta * u;

    x_next_idx = randsample(numel(X), 1, true, P(:, x_idx, u_idx));
    x_next = X{x_next_idx};

    Q(x_idx, u_idx) = Q(x_idx, u_idx) + 0.01 * (cost + beta * max(Q(x_next_idx, :)) - Q(x_idx, u_idx));
end

% Printing values
[~, policy] = max(Q, [], 2);
fprintf('Q-Learning - eta = %.2f:\n', eta);
disp(policy);


%% b

% Channel states and actions defined
X = {'B', 'G'}; 
U = {0, 1};     

% Given transition probabilities
P = [
    0.9 0.1 0.9 0.1;
    0.5 0.5 0.9 0.1
];

% Cost parameters given
zeta = 0.75;
eta = 0.6;

% This is our cost function
c = [zeta; -eta; 0]; % [ζ; -η; 0]

% Transition probabilities in matrix form
P_matrix = [
    P(1, 2) P(1, 4);
    P(2, 2) P(2, 4);
];

% Define the constraint matrix Aeq and vector beq for equalities
Aeq = [
    P_matrix(1, 1) - P_matrix(1, 2),   1, -1;
    P_matrix(2, 1) - P_matrix(2, 2),   0, -1;
    1 - P_matrix(1, 1),                0, 0;
    1 - P_matrix(2, 1),                0, 0;
];
beq = [0; 0; P(1, 1); P(2, 1)];

% Lower bound and upper bound values
LowerBound = [0; 0; 0];
upperBound = [1; 1; 1];

% Solve the linear programming problem
opts = optimoptions('linprog', 'Display', 'off');
[optU, valueFinal, chcktrue] = linprog(c, [], [], Aeq, beq, LowerBound, upperBound, opts);

%Optimization succession control
if chcktrue ~= 1 
    error('Optimization failed because it is about infeasible or unbounded.');
end

% Optimal policies
optimalU = round(optU(2));
optimalX = round(optU(1)); 

% Result
fprintf('Optimal Policy:\n');
fprintf('   When the channel is in state %s, the encoder should use the channel (u = 1).\n', X{optimalX+1});
fprintf('   When the channel is in state %s, the encoder should not use the channel (u = 0).\n', X{2 - optimalX});
fprintf('Objective Value (Minimum Cost): %.2f\n', valueFinal);


%% Q2

%% b

% This is our parameters in Kalman Filter
A = [1/2, 1, 0; 0, 2, 1; 0, 0, 2];
C = [4, 0, 0];
Q = eye(3); 
R = 1;     

T = 1000; % Total number of steps given in question

% Estimation initialization
postXT = zeros(3, 1);
postPT = eye(3);

% For getting results, we need to make storage
estXT = zeros(3, T);
diffPriXT = zeros(3, T);
mTildeT = zeros(3, T);

% Main function of Kalman filter
for t = 1:T
    % Prediction function
    priXT = A * postXT;
    priPT = A * postPT * A' + Q;

    % Measurement generation
    yt = C * priXT + sqrt(R) * randn();

    % Step of update
    Kt = priPT * C' / (C * priPT * C' + R);
    postXT = priXT + Kt * (yt - C * priXT);
    postPT = (eye(size(priPT)) - Kt * C) * priPT;

    % Storage
    estXT(:, t) = postXT;
    mTildeT(:, t) = yt - C * priXT;
    diffPriXT(:, t) = priXT - postXT;
end

% Plotting
timet = 1:T;
figure;
plot(timet, estXT(1, :), 'r', 'LineWidth', 2);
hold on;
plot(timet, estXT(2, :), 'g', 'LineWidth', 2);
plot(timet, estXT(3, :), 'b', 'LineWidth', 2);
legend('x1_t', 'x2_t', 'x3_t');
xlabel('Plot of Time Step');
ylabel('Plot of State Estimate');
title('Estimation of Kalman Filter');

figure;
plot(timet, mTildeT(1, :), 'r', 'LineWidth', 2);
hold on;
plot(timet, mTildeT(2, :), 'g', 'LineWidth', 2);
plot(timet, mTildeT(3, :), 'b', 'LineWidth', 2);
legend('tilde1', 'tilde2', 'tilde3');
xlabel('Plot of Time step');
ylabel('Plot of Measurement Residual');
title('Measurement Residuals');

figure;
plot(timet, diffPriXT(1, :), 'r', 'LineWidth', 2);
hold on;
plot(timet, diffPriXT(2, :), 'g', 'LineWidth', 2);
plot(timet, diffPriXT(3, :), 'b', 'LineWidth', 2);
legend('Priori Diff for 1', 'Priori Diff for 2', 'Priori Diff for 3');
xlabel('Plot of Time step');
ylabel('Plot of Difference');
title('Difference of State Estimates');

%% c
% Different initial for our covariance matrix
initPtPostDiff = eye(3) * 10;

% Different init for getting Kalman Filter
initXtPostDiff = zeros(3, T);
for t = 1:T
    % Prediction
    priXT = A * initXtPostDiff(:, t);
    priPT = A * initPtPostDiff * A' + Q;

    % Measurement generate
    yt = C * priXT + sqrt(R) * randn();

    % Update Step
    Kt = priPT * C' / (C * priPT * C' + R);
    initXtPostDiff(:, t) = priXT + Kt * (yt - C * priXT);
    initPtPostDiff = (eye(size(priPT)) - Kt * C) * priPT;
end

if isequal(postPT, initPtPostDiff)
    disp('Riccati recursions is unique. It means it converged to the same limit.');
else
    disp('Riccati recursions is not unique. It means it did not converge to the same limit.');
end