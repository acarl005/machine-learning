function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% forward propagation
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = h = sigmoid(z3);

% convert y vector of indices to matrix of class labels
y = eye(num_labels)(y, :);

% calculate the cost
pos = -y .* log(h); % get the cost from the "positives" (examples that DO belong to class k)
neg = (y - 1) .* log(1 - h); % get the cost from the others "the negatives"
J += sum(sum(pos + neg)) / m;

% regularize
% remove first column because we don't penalize the bias units
weights_1 = Theta1(:, 2:end);
weights_2 = Theta2(:, 2:end);
% square them and sum up
sum_weights = sum(sum(weights_1 .^ 2)) + sum(sum(weights_2 .^ 2));

J += lambda / (2 * m) * sum_weights;


% calculate the gradient
% delta2 = zeros(size(Theta1, 1), 1);
% delta3 = zeros(size(Theta2, 1), 1);
%
% for t = 1:m
%   a3_t = a3(t, :);
%   delta3 = a3_t' - (1:num_labels == y(t))';
%   delta2 = Theta2' * delta3 .* sigmoidGradient([1 z2(t, :)])';
%   delta2 = delta2(2:end);
%   Theta1_grad = Theta1_grad + (delta2 * a1(t, :));
%   Theta2_grad = Theta2_grad + (delta3 * a2(t, :));
% end
%
%
%
% Theta1_grad /= m;
% Theta2_grad /= m;

% backpropagation
delta_3 = a3 - y;
z2 = [ones(m,1) z2]; % add a column of ones. we need to in order to make the dimensions work
delta_2 = delta_3 * Theta2 .* sigmoidGradient(z2); % delta_3 * Theta2 has bias units, hence the need for extra column of ones
delta_2 = delta_2(:, 2:end); % now get rid of those bias units
Theta1_grad = (delta_2' * a1) /m;
Theta2_grad = (delta_3' * a2) /m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
