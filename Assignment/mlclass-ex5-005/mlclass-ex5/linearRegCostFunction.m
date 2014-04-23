function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
fir = 0;
las = 0;
dao = zeros(size(theta));
se = zeros(m,1);
for i = 1:m
    tmp = 0;
    for j = 1:length(theta)
        tmp = tmp + theta(j) * X(i,j) ;
    end
    fir = fir + (tmp - y(i))* (tmp - y(i)) ; 
    for p = 1:length(theta)
        dao(p) = dao(p) +(tmp - y(i)) * X(i,p);
    end
end
for  k = 1:length(theta)
    if k>1
        las = las + theta(k)*theta(k);
    end
    if k==1
        grad(k)=dao(k)/m;
    else
        grad(k)=(dao(k)+lambda*theta(k))/m;
    end
end

J = (fir + las * lambda)/(2 * m);



% =========================================================================

grad = grad(:);

end
