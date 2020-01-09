function f = meanval(theta2)
% This function computes the mean utility level
% Based on code written by Aviv Nevo, May 1998
% Adapted by Matthijs Wildenbeest, April 2010

global sj sj0 nsm X

tol = 1e-6;
    
norm = 10;
avgnorm = 10;

mvalold=(log(sj)-log(sj0));

i = 0;
while norm > tol && avgnorm > tol
      
    mu=mufunc(X,theta2);
          
    mval = mvalold + log(sj) - log(sum(ind_sh(mvalold,mu),2)/nsm);
       
    t = abs(mval-mvalold);
    norm = max(t);
    avgnorm = mean(t);
  	mvalold = mval;
    i = i + 1;
end
%disp(['# of iterations for delta convergence:  ' num2str(i)]);

f = mval;