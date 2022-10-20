clc,clear;
hold on;
grid on;
data = load('data.txt');
k = data(:,1);
alg1 = data(:,2);
alg2 = data(:,3);
alg3 = data(:,4);

plot(k, alg1, 'Linewidth', 1.5);
plot(k, alg2, 'Linewidth', 1.5);
plot(k, alg3, 'Linewidth', 1.5);
xlabel('k','FontSize', 12)
ylabel('times of best performing', 'FontSize', 12)
title('Evaluation','FontSize', 12)
legend('alg1', 'alg2', 'random')

