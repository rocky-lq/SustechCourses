//
// Created by Luo Qi on 2022/10/21.
//
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>

#define eps 1e-2

using namespace std;

//原方程
double fx(double x) {
    return x * x * x + x - 1;
}

//导函数
double dfx(double x) {
    return 3 * x * x + 1;
}

//平分法
double binary(double a, double b) {
    double x;
    do {
        x = (a + b) / 2;
        if (fx(a) * fx(x) > 0)
            a = x;
        else b = x;

    } while (fabs(fx(x)) > eps);
    return x;

}

//试位法
double false_pos(double a, double b) {
    double x;
    do {
        x = (a * fx(b) - b * fx(a)) / (fx(b) - fx(a));
        if (fx(x) * fx(a) > 0)
            a = x;
        else b = x;
    } while (fabs(fx(x)) > eps);
    return x;
}

//牛顿法
double newton(double x) {
    do {
        x = x - fx(x) / dfx(x);
    } while (fabs(fx(x)) > eps);
    return x;
}

int main() {
    double a, b;
    a = -10.0;
    b = 10.0;
    printf("%lf\n", binary(a, b));
    printf("%lf\n", false_pos(a, b));
    printf("%lf\n", newton(3));
    return 0;
}