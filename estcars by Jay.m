% Main file
% Written by Matthijs Wildenbeest, April 2010.
% cd '\\Client\C$\Users\Jay Shin\Box Sync\Box_Sync-01112018-backup_gmail\1_CU\2018_6_Spring\924_De los Santos\Assignment1'

clear all;
warning off all
format long

global theta1 X iv W nsm vfull incfull pj sj sj0 cdindex cdid owner X2

seed = rng;

load cars.mat;      % sales 50 or less excluded, no exotic makes

tbl = 2;              % number of markets
k = 7;              % number of characteristics used (constant, pj, 10*hp./weight, american+asian, airco, kpe/10, sizecar)
nsm = 200;          % number of simulated "indviduals" per market 

% net ("besteedbaar") income distribution netherlands 
% fitted parameters: location: 3.30; scale: 0.60 (gross income: 3.73 and 0.79)
inc = lognrnd(3.30,0.6,tbl,nsm);

v = normrnd(0,1,tbl,k*nsm); % multivariate normal draws, one for each characteristic

F=length(cdid2);    % total number of brands
J=length(sales);    % total number of models

% define horsepower
hp=hpkw/.735;

% correct price for inflation
pj=price./(cpi(cdid)/100)/1000;

% define size
sizecar=(lengthcar/1000).*(width/1000);

% define kilometers per liter and per euro
kpl=100./gasavg;
kpe=kpl./(gasprice(cdid,:)/100);

% matrix of income
incfull=inc(cdid,:); 

% create marketshares using sales data
sj=sales./(1000*numhh(cdid,:));

% compute the outside good market share by market
temp = cumsum(sj);
sum1 = temp(cdindex,:);
sum1(2:size(sum1,1),:) = diff(sum1);
sj0 = 1.0 - sum1(cdid,:);

y=log(sj)-log(sj0);
A=[ones(J,1) 10*hp./weight american+asian airco kpe/10 sizecar];
X=[A pj];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Q1. OLS logit estimation    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e=y-X*inv(X'*X)*X'*y;
n=size(X,1);
k=size(X,2);
N=eye(n)-ones(n,1)*ones(n,1)'/n;

betaols = (X'*X)\X'*y;
covbetaols=(e'*e/(n-k))*inv(X'*X);
sebetaols=sqrt(diag(covbetaols));

alphaols=betaols(size(A,2)+1);
eols=alphaols*pj.*(1-sj);

R2ols=(betaols'*X'*N*X*betaols)/(y'*N*y);
tstatols=betaols./sebetaols;
pvalueols=2*(1-tcdf(abs(betaols./sebetaols),n-k));

% print estimation results on screen
varnames=['Constant      '; 'HP/Weight(x10)'; 'Foreign       '; 'AC            ';  'KPE(/10)      '; 'Size          '; 'Price         '];
fprintf(1,'OLS LOGIT DEMAND ESTIMATION\n')
fprintf(1,'-------------------------------\n')
fprintf(1,'variable       beta    se     p\n')
fprintf(1,'%s', varnames(1,:))
fprintf(1,'   % 1.3f %1.3f %1.3f\n',[betaols(1)';sebetaols(1)'; pvalueols(1)'])
for i=2:k
    fprintf(1,'%s', varnames(i,:))
    fprintf(1,'    % 1.3f %1.3f %1.3f\n',[betaols(i)';sebetaols(i)'; pvalueols(i)'])
end
fprintf(1,'R-squared     %1.3f\n',R2ols)
fprintf(1,'# inel.dem. %1.0f\n',length(find(abs(eols)<1)))
fprintf(1,'Avg. elas.   % 1.3f\n',mean(eols))
fprintf(1,'# products %1.0f\n',length(X))
fprintf(1,'-------------------------------\n')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Q2. Group fixed effects    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tbl = table(y, groupcode, 10*hp./weight, american+asian, airco, kpe/10, sizecar, pj);
tbl.Properties.VariableNames = {'share' 'group' 'hp' 'foreign' 'ac' 'kpe' 'size' 'price'};
tbl.group = nominal(tbl.group);
lme = fitlme(tbl, 'share ~ hp + foreign + ac + kpe + size + price + (1|group)');
[~,~,stats] = fixedEffects(lme);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Q3. Two-stage least squares    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A=[ones(J,1) 10*hp./weight american+asian airco kpe/10 sizecar];
X=[A pj];
C=[10*hp./weight american+asian airco kpe/10 sizecar];

% create a matrix containing the instruments; similar to BLP
makeindex=[find(diff(makecode)~=0); J];
groupindex=[find(diff(groupcode)~=0); size(X,1)];
nmdl=[makeindex(1); diff(makeindex)];% number of models per brand

% create ownership matrix
owner=[];
ngroup=[groupindex(1); diff(groupindex)];
n=1;
groupcode2=zeros(length(groupcode),1);
for i=1:length(ngroup)
    owner=blkdiag(owner,ones(ngroup(i)));
    groupcode2(n:groupindex(i),:) = i*ones(ngroup(i),1);
    n=groupindex(i)+1;
end

temp=cumsum(C);
sum1 = temp(groupindex,:);
sum1(2:size(sum1,1),:) = diff(sum1);
z1=C; % product characteristics - 5 variables
z2=sum1(groupcode2,:)-C; % the sum of the product characteristics across own firm products - 5 variables
sum2=temp(cdindex,:);
sum2(2:size(sum2,1),:) = diff(sum2);
z3=sum2(cdid,:)-sum1(groupcode2,:); % the sum of the product characteristics across rival firm products - 5 variables
% z2nd=[z2(:,2) z2(:,5) z2(:,6)]; % non-dummy attributes
% z3nd=[z3(:,2) z3(:,5) z3(:,6)]; % non-dummy attributes
iv=[z1, z2, z3]; % instruments - total 15 variables excluding a constant term

% iv=[z1 z2(:,1) sum(z2nd,2) z3(:,1) sum(z3nd,2) sum(z1,2).*z2(:,1) sum(z1,2).*sum(z2nd,2) sum(z1,2).*z3(:,1) sum(z1,2).*sum(z3nd,2)]; % instruments

xhat = iv/(iv'*iv)*iv'*X; % z(z'z)^(-1)z' - project matrix on z-space
PX2sls = (xhat'*xhat)\xhat'; % project matrix on weighted x-space for 2SLS
beta2sls = PX2sls*y;
covbeta2sls=(1/(n-k))*((y-X*beta2sls)'*(y-X*beta2sls))*inv(xhat'*xhat);
sebeta2sls=sqrt(diag(covbeta2sls));

alpha2sls=beta2sls(size(A,2)+1);
e2sls=alpha2sls*pj.*(1-sj); %own-elasticity

ecp2sls=zeros(J,J);
for jj=1:J
    for kk=1:J
        ecp2sls(jj,kk)=-alpha2sls*pj(kk).*sj(kk);
    end 
end

tstat2sls=beta2sls./sebeta2sls;
pvalue2sls=2*(1-tcdf(abs(beta2sls./sebeta2sls),n-k));

% print estimation results on screen
varnames=['Constant      '; 'HP/Weight(x10)'; 'Foreign       '; 'AC            ';  'KPE(/10)      '; 'Size          '; 'Price         '];
fprintf(1,'IV LOGIT DEMAND ESTIMATION WITHOUT GROUP FIXED EFFECTS\n')
fprintf(1,'-------------------------------\n')
fprintf(1,'variable       beta    se     p\n')
fprintf(1,'%s', varnames(1,:))
fprintf(1,'   % 1.3f %1.3f %1.3f\n',[beta2sls(1)';sebeta2sls(1)'; pvalue2sls(1)'])
for i=2:k
    fprintf(1,'%s', varnames(i,:))
    fprintf(1,'    % 1.3f %1.3f %1.3f\n',[beta2sls(i)';sebeta2sls(i)'; pvalue2sls(i)'])
end
fprintf(1,'R-squared      n.a.\n')
fprintf(1,'# inel.dem.  %1.0f\n',length(find(abs(e2sls)<1)))
fprintf(1,'Avg. elas.   % 1.3f\n',mean(e2sls))
fprintf(1,'# products %1.0f\n',length(X))
fprintf(1,'-------------------------------\n')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Q4. Own/Cross-Price Elasticities   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
q4cross=ecp2sls([260 371 436 323 419 238 217 301],[260 371 436 323 419 238 217 301]);
r = size(q4cross,1) + 1;
q4cross(1:r:end) = 0;
q4own = diag(e2sls([260 371 436 323 419 238 217 301]));
q4ans = q4cross + q4own;
xlswrite('q4.xlsx', q4ans)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Q6. Random-Coefficients Logit with Demand Side   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W=iv'*iv; % weight matrix
X=A;
vfull=v(cdid,:); 

options = optimset('Display','iter','MaxIter',1000,'MaxFunEvals',10000,'TolFun',1e-4,'TolX',1e-4,'GradObj','off','LargeScale','on');

tic
[theta2hat3,fval3] = fminsearch('gmmobj',[5 1 1 1 1 1 1],options);
t=toc;

fprintf(1,'It took %1.0f minutes and %1.0f seconds to estimate the demand side of BLP.\n',floor(t/60),round(t-floor(t/60)*60))

theta1hat3 = theta1;

delta=meanval(theta2hat3);
sigma=theta2hat3(1:end);
mu=mufunc(X,sigma); 

% print estimation results on screen
fprintf(1,'GMM DEMAND ESTIMATION\n')
fprintf(1,'--------------------------\n')
fprintf(1,'variable       beta  stdev\n')
fprintf(1,'%s', varnames(1,:))
fprintf(1,'   % 1.3f % 1.3f\n',[theta1hat3(1)';theta2hat3(2)'])
for i=2:size(X,2)
    fprintf(1,'%s', varnames(i,:))
    fprintf(1,'    % 1.3f % 1.3f\n',[theta1hat3(i)';theta2hat3(i+1)'])
end
fprintf(1,'%s', varnames(size(X,2)+1,:))
fprintf(1,'    % 1.3f   ---\n',theta2hat3(1)')
fprintf(1,'Obj. func.    %1.3f\n',fval3)
fprintf(1,'# products  %1.0f\n',length(X))
fprintf(1,'--------------------------\n')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Q7. Random-Coefficients Logit with Demand and Supply Side   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A=[ones(J,1) 10*hp./weight american+asian airco kpe/10 sizecar];
X2=[A(:,1) log(A(:,2)) A(:,3) A(:,4) log(kpl/10) log(A(:,6))]; % contains the cost side explanatory variables

tic
[theta2hat4,fval4] = fminsearch('gmmobj2',[5 1 1 1 1 1 1],options);
t=toc;

fprintf(1,'It took %1.0f minutes and %1.0f seconds to estimate the demand and supply side of BLP.\n',floor(t/60),round(t-floor(t/60)*60))

theta1hat4 = theta1;

delta=meanval(theta2hat4);
sigma=theta2hat4(1:end);
mu=mufunc(X,sigma); 

% print estimation results on screen
fprintf(1,'GMM DEMAND AND SUPPLY ESTIMATION\n')
fprintf(1,'---------------------------------\n')
fprintf(1,'variable       beta  stdev   cost\n')
fprintf(1,'%s', varnames(1,:))
fprintf(1,'   % 1.3f % 1.3f % 1.3f\n',[theta1hat4(1)';theta2hat4(2)';theta1hat4(7)])
for i=2:size(X,2)
    fprintf(1,'%s', varnames(i,:))
    fprintf(1,'    % 1.3f % 1.3f % 1.3f\n',[theta1hat4(i)';theta2hat4(i+1)';theta1hat4(i+6)])
end
fprintf(1,'%s', varnames(size(X,2)+1,:))
fprintf(1,'    % 1.3f   ---    ---\n',theta2hat4(1)')
fprintf(1,'Obj. func.    %1.3f\n',fval4)
fprintf(1,'# products  %1.0f\n',length(X))
fprintf(1,'---------------------------------\n')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Q8. Another Own/Cross-Price Elasticities   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ai = theta2hat4(1)./inc;
alphai = [repmat(ai(1,:),cdindex(1),1) ; repmat(ai(2,:),cdindex(2)-cdindex(1),1)];
sij = ind_sh(delta,mu);
temp = alphai.*sij.*(1.-sij);
q8own = -(pj./sj).*(1/nsm).*sum(temp,2);

q8cross = zeros(J,J);
for jj=1:J
    for kk=1:J
        temp = alphai(jj,:).*sij(jj,:).*sij(kk,:);
        q8cross(jj,kk) = pj(kk)./sj(jj).*(1/nsm).*sum(temp,2);
    end
end

q8crossANS = q8cross([260 371 436 323 419 238 217 301],[260 371 436 323 419 238 217 301]);
r = size(q8crossANS,1) + 1 ;
q8crossANS(1:r:end) = 0;
q8ownANS = diag(q8own([260 371 436 323 419 238 217 301]));
q8ans = q8ownANS + q8crossANS;
xlswrite('q8.xlsx',q8ans)