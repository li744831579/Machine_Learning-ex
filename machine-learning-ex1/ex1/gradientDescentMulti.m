function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
num_features = size(X,2);
temp = zeros(num_features,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %temp(1) = theta(1) - alpha * (1/m) * sum((X*theta - y));
    %if (num_features >= 2)
      %for feature_iter = 2:num_features
       % temp(feature_iter) = theta(feature_iter) - alpha * (1/m) * ((X*theta - y)' * X(:,feature_iter));
      %endfor      
    %else
     % temp(2) = theta(2) - alpha * (1/m) * ((X*theta - y)' * X(:,2));
    %endif
    %theta = temp;
    %fprintf("theta = %f \n",theta);
    
    theta = theta - alpha/m * X'*(X*theta - y);
    fprintf("theta = %f \n",theta);
    










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end


end
