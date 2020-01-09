function f = gmmobj2(theta2)
% This function computes the GMM objective function
% Based on code written by Aviv Nevo, May 1998
% Adapted by Matthijs Wildenbeest, April 2010

global theta1 X iv W pj X2

delta = meanval(theta2);

%%% SUPPLY SIDE
mu=mufunc(X,theta2); % non-linear part utility function
sij=ind_sh(delta,mu); % individual buying probabilities
alpha=theta2(1); % price coefficient
b=markup(alpha,sij); % markup
mc=log(pj-b); % marginal cost specification
delta = [delta; mc]; % delta now includes mc

X1 = blkdiag(X,X2);
iv2= blkdiag(iv,iv); 
W=iv2'*iv2;
%%%

temp1 = X1'*iv2;
temp2 = iv2'*delta;
theta1 = (temp1/W*temp1')\temp1/W*temp2;
clear temp1 temp2 
gmmresid = delta - X1*theta1;
temp1 = gmmresid'*iv2;
f = temp1/W*temp1';
clear temp1