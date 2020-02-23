%% initialise data

clear; close all; clc

xlRange = 'H:S';
data = xlsread("caschool.xlsx",xlRange);

[m,n] = size(data);

x(:,1) = (data(:,1) - mean(data(:,1))) / range(data(:,1));
x(:,2) = (data(:,2) - mean(data(:,2))) / range(data(:,2));
x(:,3) = (data(:,5) - mean(data(:,5))) / range(data(:,5));
x(:,4) = (data(:,6) - mean(data(:,6))) / range(data(:,6));
x(:,5) = (data(:,9) - mean(data(:,9))) / range(data(:,9));

y = data(:,12);
  
%% gradient functions

%for this example theta will be a vector with 108 elements.
%as:
%1) input layer: 5 nodes
%2) hidden layer 1: 5 nodes
%3) hidden layer 2: 5 nodes
%4) hidden layer 3: 5 nodes
%5) output layer: 3 nodes

%% SPECIFY LAYERS HERE

[~,k] = size(x);

in_lay = k;
hid_lay1 = 5;
hid_lay2 = 5;
hid_lay3 = 5;
out_lay = max(y);

%% UNROLLING OF THETA

init_epsilon = 1.5;

theta1 = rand(hid_lay1, in_lay + 1) * (2 * init_epsilon) - init_epsilon;
theta2 = rand(hid_lay2, hid_lay1 + 1) * (2 * init_epsilon) - init_epsilon;
theta3 = rand(hid_lay3, hid_lay2 + 1) * (2 * init_epsilon) - init_epsilon;
theta4 = rand(out_lay, hid_lay3 + 1) * (2 * init_epsilon) - init_epsilon;

initialtheta = [theta1(:)' theta2(:)' theta3(:)' theta4(:)'];
[~,thetasize] = size(initialtheta);

[jval,dvec] = costfunc_neuralnetwork(initialtheta,x,y);

% 
% for i = 1:thetasize
%   thetaPlus = initialtheta;
%   thetaPlus(i) = thetaPlus(i) + init_epsilon;
%   thetaMinus = initialtheta;
%   thetaMinus(i) = thetaMinus(i) - init_epsilon;
%   [JthetaPlus,dvec1] = costfunc_neuralnetwork(thetaPlus,x,y);
%   [JthetaMinus,dvec2] = costfunc_neuralnetwork(thetaMinus,x,y);
%   gradApprox(i) = (JthetaPlus - JthetaMinus)/(2*init_epsilon);
% end
% 
% error = dvec - gradApprox;

%% optimization


options = optimset('GradObj','on','MaxIter',10000);
[optTheta, functionVal, exitFlag] = fminunc(@(theta)costfunc_neuralnetwork(theta,x,y),initialtheta,options)

%% visualisation & accuracy