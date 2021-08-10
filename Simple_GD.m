%% CODE TO IMPLEMENT AND VISUALIZE GD with fixed step size and backtracking line search

dbstop at 66
dbstop at 133

%% Gradient Descent with fixed step-size

% Declaring x, and X for storing all updated x's for plotting ...
x = [3,4];
X = zeros(2,50);
% Recording the starting point ...
X(:,1) = x;
% Initializing the hyperparameters ...
step_size = 0.05; tolerance = 0.001;
% For the first condition check in the while loop
grad = calc_grad(x);
% For counting number of iterations
count = 1;

% Optimization loop for gradient descent with fixed step size
while norm(grad,2)>tolerance
    % Updating x using the value of gradient at (x,f(x)) ...
    x(1) = x(1) - grad(1) * step_size;
    x(2) = x(2) - grad(2) * step_size;
    % Recalculating the gradient for the next iteration ...
    grad = calc_grad(x);
    % Updating count ...
    count = count+1;
    % Storing x ...
    X(:,count) = x;
end

fprintf("\nGRADIENT DESCENT WITH FIXED STEP SIZE\n")
fprintf("Hyperparameters: Step_Size = %d, Tolerance = %d\n",step_size,tolerance)
fprintf("\nNumber of steps taken to converge with fixed-step GD : %d\n",count)
fprintf("Convergence point x* = (%d,%d)\n",X(1,count),X(2,count))
fprintf("Optimized Objective at x* = %d\n\n",calc_func(X(:,count)))

% Plotting the function wrt x to visualize gradient descent
figure(2) 
x_1 = linspace(-5,5);
x_2 = linspace(-5,5);
[xx,yy]=meshgrid(x_1,x_2);
zz = xx.^4 + yy.^2;
colormap gray
surfl(xx,yy,zz)
shading interp
title('Visualization of the gradient descent process')
hold on 

points = zeros(count,3);
% It can be observed can see that the last triangular point is where 
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

clear

%% Gradient Descent with backtracking-line search

% Declaring x, and X for storing all updated x's for plotting ...
x = [3,4];
X = zeros(2,20);
% Recording the starting point ...
X(:,1) = x;
% Initializing the hyperparameters ...
alpha = 0.1; beta = 0.5;
tolerance = 0.001;    
% For the first condition check in the while loop
del_x = -calc_grad(x);
% For counting number of iterations
count = 1;

% Optimization loop for gradient descent with backtracking line search  
while norm(del_x) > tolerance
   % Bactracking line search for getting optimal step-size
   t = 1;
   while calc_func(x+t*del_x) > calc_func(x) - alpha*t*(del_x(1)^2+del_x(2)^2)
        t = beta * t;
   end
   % Updating x with the calculated step-size ...
   x = x + t*del_x;
   % Calculating gradient for the next iteration ...
   del_x = -calc_grad(x);
   % Updating count ...
   count = count+1;
   % Storing x ...
   X(:,count) = x;
end

fprintf("\nGRADIENT DESCENT WITH BACKTRACKING LINE_SEARCH\n")
fprintf("Hyperparameters: alpha = %d, beta = %d, Tolerance = %d\n",alpha,beta,tolerance)
fprintf("\nNumber of steps taken to converge with backtracking line search : %d\n",count)
fprintf("Convergence point x* = (%d,%d)\n",X(1,count),X(2,count))
fprintf("Optimized Objective at x* = %d\n\n",calc_func(X(:,count)))

% Plotting the function wrt x to visualize gradient descent
figure(2) 
x_1 = linspace(-5,5);
x_2 = linspace(-5,5);
[xx,yy]=meshgrid(x_1,x_2);
zz = xx.^4 + yy.^2 ;
colormap gray
surfl(xx,yy,zz)
shading interp
title('Visualization of gradient descent with backtracking line search')
hold on

points = zeros(count,3);
% It can be observed can see that the last triangular point is where 
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

clear
%% END OF CODE

% WRITTEN BY Adithya Gowtham R || EE17B146