clc, clear;

A = [1, 1, 2];
B = [1, 1, 1; -1, -1, -1; 0, 0, 1];
C = [2; -1; 2];
n = size(A,2);


cvx_begin
    variable x(n);
    maximize(A*x);
    subject to
        B * x <= C;
        x >= 0;
cvx_end
A*x;