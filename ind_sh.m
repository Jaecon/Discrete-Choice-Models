function f = ind_sh(delta,mu)
% This function computes the "individual" probabilities of choosing each brand
% Based on code written by Aviv Nevo, May 1998.
% Adapted by Matthijs Wildenbeest, April 2010

global nsm cdindex cdid
eg = exp(mu).*kron(ones(1,nsm),exp(delta));
temp = cumsum(eg); 
sum1 = temp(cdindex,:);
sum1(2:size(sum1,1),:) = diff(sum1);

denom1 = 1./(1+sum1);
denom = denom1(cdid,:);

sij=eg.*denom;

f = sij;