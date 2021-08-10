%% %% CODE TO IMPLEMENT AND VISUALIZE RMSProp with Momentum

dbstop at 77

% Declaring x, and X for storing all updated x's for plotting ...
x = [3,4];
X = zeros(2,50);
% Recording the starting point ...
X(:,1) = x;
% Initializing the hyperparameters ...
mu = 0.3; step_size = 0.05; tolerance = 0.001;
decay_rate = 0.90; delta = 1e-7;
% For the first condition check in the while loop
grad = calc_grad(x);
% For counting number of iterations
count = 1;

v = [0,0];  r = [0,0];
var_step = [0,0];

% Optimization loop for RMSProp with Momentum
while norm(grad,2)>tolerance
    % Updating r using the decay rate and the value of gradient at (x,f(x)) ...
    r = decay_rate * r + (1-decay_rate)*grad.*grad;
    % Calculating the variable step sizes using r ...
    for i = 1:length(r)
        var_step(i) = 1/(delta+sqrt(r(i)));
    end
    % Updating the momentum with the value of gradient at (x,f(x)) ...
    v = mu * v - step_size * var_step.*grad;
    % Updating x with the RMSProp momentum 
    x = x + v;
    % Recalculating the gradient for the next iteration ...
    grad = calc_grad(x);
    % Updating count ...
    count = count+1;
    % Storing x ...
    X(:,count) = x;
end

fprintf("\nRMSPROP WITH MOMENTUM\n")
fprintf("Hyperparameters: Mu = %d, Step_Size = %d, Tolerance = %d,\n\t\t\t\tDecay Rate = %d, Delta = %d\n",mu,step_size,tolerance,decay_rate,delta)
fprintf("\nNumber of steps taken to converge with RMSProp : %d\n",count)
fprintf("Convergence point x* = (%d,%d)\n",X(1,count),X(2,count))
fprintf("Optimized Objective at x* = %d\n\n",calc_func(X(:,count)))

% Plotting the function wrt x to visualize RMSProp ...
figure(2) 
x_1 = linspace(-4,4);
x_2 = linspace(-4,5);
[xx,yy]= meshgrid(x_1,x_2);
zz = xx.^4 + yy.^2;
colormap gray
surfl(xx,yy,zz);
shading interp
title('Visualization of RMSProp with Momentum')
hold on 

points = zeros(count,3);
% It can be observed can see that the last filled point is where 
% the algorithm converges, or the optimal point.
for i = 1:count
    if i < count 
        scatter3(X(1,i),X(2,i),calc_func(X(:,i)),'.','black');
        points(i,1) = X(1,i);  points(i,2) = X(2,i); points(i,3) = calc_func(X(:,i));
    else 
        scatter3(X(1,i),X(2,i),calc_func(X(:,i)),'^','filled','black');
    end
    hold on
end

plot3(points(:,1),points(:,2),points(:,3),'red')
xlabel('x\rightarrow') 
ylabel('y\rightarrow') 
zlabel('x^4+y^2\rightarrow')

clear
%% END OF CODE

% WRITTEN BY Adithya Gowtham R || EE17B146