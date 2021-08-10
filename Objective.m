x_1 = linspace(-4,4);
x_2 = linspace(-4,5);
[xx,yy]= meshgrid(x_1,x_2);
zz = xx.^4 + yy.^2;
colormap default
surfl(xx,yy,zz);
shading interp
title("The Objective Function")
xlabel('x\rightarrow') 
ylabel('y\rightarrow') 
zlabel('x^4+y^2\rightarrow')