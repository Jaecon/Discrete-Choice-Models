function b = markup(alpha,sij)
% This function computes the markups

% Written by Matthijs Wildenbeest, April 2010.

global D1 D nsm incfull cdindex owner

n = 1;
D=zeros(size(sij,1),size(sij,1));
s=sum(sij,2)/nsm;
ss=sum((alpha./incfull).*sij,2)/nsm;
for i = 1:size(cdindex,1)
    D1 = (((alpha./incfull(n:cdindex(i),:)).*sij(n:cdindex(i),:))*sij(n:cdindex(i),:)')/nsm;
    D(n:cdindex(i),n:cdindex(i)) = (diag(ss(n:cdindex(i))) - D1);
    n = cdindex(i) + 1;
end

b=(D.*owner)\s;