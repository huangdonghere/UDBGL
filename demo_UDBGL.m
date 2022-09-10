function demo_UDBGL()

clear;
close all;
clc;

dataName = 'OutScene';

load(['data_',dataName,'.mat'],'X','Y'); 
n = numel(Y);
c = numel(unique(Y)); % The number of clusters
m = c; %The number of anchors
alpha = 1e-3; beta = 1e-5;

tic;
Label = UDBGL(X,c,m,alpha,beta);
toc;

scores = NMImax(Label,Y);

disp(['NMI = ',num2str(scores)]);