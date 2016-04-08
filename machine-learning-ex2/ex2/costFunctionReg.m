function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%                m                                                                           n
% J(𝜃) = (1 / m) 𝛴 [ -y^(i)log(h_𝜃( x^(i) )) - (1 - y^(i))log(1 - h_𝜃( x^(i) )) ] + (𝜆 / 2m) 𝛴 𝜃_j^2
%               i=1                                                                         j=1


hypothesis = sigmoid(X * theta);
pos = -y .* log(hypothesis);
neg = (y - 1) .* log(1 - hypothesis);

J = sum(pos + neg) / m + (lambda / 2 / m) * sum(theta(2:end) .^ 2);

%             m
% 𝛿 = (1 / m) 𝛴 (h_𝜃( x^(i) ) - y^(i))x_j^(i) + (𝜆 / m)𝜃_j
%            i=1


grad(1) = X(:,1)' * (hypothesis - y) / m;
grad(2:end)= X(:,2:end)' * (hypothesis - y) / m + (lambda * theta(2:end) / m);



% =============================================================

end
