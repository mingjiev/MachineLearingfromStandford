function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    m_th = length(theta);
    tmp = zeros(m_th,1);
    
    for i = 1:m
        atmp = 0;
        for j = 1:length(theta)
            atmp = atmp + theta(j) * X(i,j) ;
        end
        
        tmp(1) = tmp(1) + ( atmp - y(i) );
        for j = 2:m_th
            tmp(j) = tmp(j) + X(i,j) * (atmp - y(i));
        end
    end
    
    tmp = tmp./m;
    theta = theta-alpha.*tmp;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
