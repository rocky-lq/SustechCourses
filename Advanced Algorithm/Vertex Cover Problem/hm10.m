% Optimization Toolbox
clc,clear;
load("instance_small.mat")
n=100; m=10;
% load("instance_medium.mat")
% n=100000; m=50;
% load("instance_large.mat")
% n=1000000; m=100;

A = [1, 1, 2];
B = [1, 1, 1; -1, -1, -1; 0, 0, 1];
C = [2; -1; 2];
n = size(A,2);

f = A;
lb = zeros(1, n);
ub = ones(1, n).*inf;
x = linprog(f.*-1, B, C,[], [], lb, ub);
A*x;