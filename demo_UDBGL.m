function demo_UDBGL()

clear;
close all;
clc;

dataName = 'WebKB-Texas';

load(['data_',dataName,'.mat'],'X','Y'); 
n = numel(Y);
c = numel(unique(Y)); % The number of clusters
m = c; %The number of anchors
alpha = 10; beta = 0.1;
opts.Distance = 'cosine';

tic;
Label = UDBGL(X,c,m,alpha,beta,opts);
toc;

scores = NMImax(Label,Y);

disp(['NMI = ',num2str(scores)]);