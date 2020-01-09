function f = mufunc(X,sigma)
% This function computes the non-linear part of the utility
% Based on code written by Aviv Nevo, May 1998
% Adapted by Matthijs Wildenbeest, April 2010

global nsm vfull incfull pj
[n k] = size(X);

for i = 1:nsm
        v_i = vfull(:,i:nsm:k*nsm);
        inc_i = incfull(:,i);
        mu(:,i) = (-sigma(1).*pj./inc_i+(X.*v_i*(sigma(2:end))'));   
end
f = mu;