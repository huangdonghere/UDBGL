%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                               %
% This is a demo for the UDBGL algorithm, which is proposed in the paper below. %
%                                                                               %
% Si-Guo Fang, Dong Huang, Xiao-Sha Cai, Chang-Dong Wang, Chaobo He, Yong Tang. %
% Efficient Multi-view Clustering via Unified and Discrete Bipartite Graph      %
% Learning, IEEE Transactions on Neural Networks and Learning Systems, 2023.    %
%                                                                               %
% The code has been tested in Matlab R2019b on a PC with Windows 10.            %
%                                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demo_UDBGL()

clear;
close all;
clc;

dataName = 'OutScene';

load(['data_',dataName,'.mat'],'X','Y'); 
n = numel(Y);
c = numel(unique(Y)); % The number of clusters

% Parameters
m = c; %The number of anchors
alpha = 1e-3; 
beta = 1e-5;

Label = UDBGL(X,c,m,alpha,beta);

disp('Done.');
scores = NMImax(Label,Y);
disp(['NMI = ',num2str(scores)]);
