function f = gmmobj(theta2)
% This function computes the GMM objective function
% Based on code written by Aviv Nevo, May 1998
% Adapted by Matthijs Wildenbeest, April 2010

global theta1 X iv W

delta = meanval(theta2);

temp1 = X'*iv;
temp2 = iv'*delta;
theta1 = (temp1/W*temp1')\temp1/W*temp2;
clear temp1 temp2 
gmmresid = delta - X*theta1;
temp1 = gmmresid'*iv;
f = temp1/W*temp1';
clear temp1