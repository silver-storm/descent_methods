%% CODE TO IMPLEMENT AND VISUALIZE Adam AND AdamW

dbstop at 76
dbstop at 148

%% Adam

% Declaring x, and X for storing all updated x's for plotting ...
x = [3,4];
X = zeros(2,50);
% Recording the starting point ...
X(:,1) = x;
% Initializing the hyperparameters ...
step_size = 0.1; tolerance = 0.001;
beta1 = 0.5; beta2 = 0.8; delta = 1e-7;
% For the first condition check in the while loop
grad = calc_grad(x);
% For counting number of iterations
count = 1;

% Initializing the algorithm parameters ...
v = [0,0];  r = [0,0];

% Optimization loop for Adam
while norm(grad,2)>tolerance
    % Updating v using beta1 and the value of gradient at (x,f(x)) ...
    v = (beta1 * v + (1-beta1) * grad )/(1-beta1^count);
    % Updating r using beta2 and the value of gradient at (x,f(x)) ...
    r = (beta2*r + (1-beta2) * grad.*grad)/(1-beta2^count);
    % Updating x with v, r and delta ...
    for i = 1:length(r)
        x(i) = x(i) - step_size * v(i)/(delta+sqrt(r(i)));
    end
    % Recalculating the gradient for the next iteration ...
    grad = calc_grad(x);
    % Updating count ...
    count = count+1;
    % Storing x ...
    X(:,count) = x;
end

fprintf("\nADAM\n")
fprintf("Hyperparameters: Step_Size = %d, Tolerance = %d, Beta1 = %d,\n\t\t\t\t Beta2 = %d, Decay Rate = %d, Delta = %d\n",step_size,tolerance,beta1,beta2,delta)
fprintf("\nNumber of steps taken to converge with Adam : %d\n",count)
fprintf("Convergence point x* = (%d,%d)\n",X(1,count),X(2,count))
fprintf("Optimized Objective at x* = %d\n\n",calc_func(X(:,count)))

% Plotting the function wrt x to visualize Adam
figure(2) 
x_1 = linspace(-4,4);
x_2 = linspace(-4,5);
[xx,yy]= meshgrid(x_1,x_2);
zz = xx.^4 + yy.^2;
colormap gray
surfl(xx,yy,zz);
shading interp
title('Visualization of the Adam optimization algorithm')
hold on 

points = zeros(count,3);
% It can be observed can see that the last filled point is where 
% the gradient descent algorithm converges, or the optimal point.
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

%% AdamW

% Declaring x, and X for storing all updated x's for plotting ...
x = [3,4];
X = zeros(2,50);
% Recording the starting point ...
X(:,1) = x;
% Initializing the hyperparameters ...
step_size = 0.1; tolerance = 0.001;
beta1 = 0.5; beta2 = 0.8; delta = 1e-7;
L2_reg = 0.01; decay = 0.01; 
% For the first condition check in the while loop
grad = calc_grad(x);
% For counting number of iterations
count = 1;

v = [0,0];  r = [0,0];

% Optimization loop for the gradient descent method
while norm(grad,2)>tolerance
    % Updating v using beta1 and the value of gradient at (x,f(x)) ...
    v = (beta1 * v + (1-beta1) * (grad + L2_reg*x))/(1-beta1^count);
    % Updating r using beta2 and the value of gradient at (x,f(x)) ...
    r = (beta2*r + (1-beta2) * (grad + L2_reg*x).*(grad + L2_reg*x))/(1-beta2^count);
    % Updating x with v, r, delta and the decay parameter wrt x ...
    for i = 1:length(r)
        x(i) = x(i) - step_size * v(i)/(delta+sqrt(r(i))) - step_size * decay * x(i);
    end
    % Recalculating the gradient for the next iteration ...
    grad = calc_grad(x);
    % Updating count ...
    count = count+1;
    % Storing x ...
    X(:,count) = x;
end

fprintf("\nADAM_W\n")
fprintf("Hyperparameters: Step_Size = %d, Tolerance = %d, Beta1 = %d,\n\t\t\t\t Beta2 = %d, Delta = %d, L2_Regularization = %d,\n\t\t\t\t Decay Rate = %d \n",step_size,tolerance,beta1,beta2,delta,decay,L2_reg)
fprintf("\nNumber of steps taken to converge with AdamW : %d\n",count)
fprintf("Convergence point x* = (%d,%d)\n",X(1,count),X(2,count))
fprintf("Optimized Objective at x* = %d\n\n",calc_func(X(:,count)))

% Plotting the function wrt x to visualize AdamW
figure(2) 
x_1 = linspace(-4,4);
x_2 = linspace(-4,5);
[xx,yy]= meshgrid(x_1,x_2);
zz = xx.^4 + yy.^2;
colormap gray
surfl(xx,yy,zz);
shading interp
title('Visualization of the AdamW optimization algorithm')
hold on 

points = zeros(count,3);
% It can be observed can see that the last filled point is where 
% the gradient descent algorithm converges, or the optimal point.
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