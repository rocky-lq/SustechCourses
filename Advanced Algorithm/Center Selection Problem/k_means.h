//
// Created by Luo Qi on 2022/10/27.
//

#ifndef CSP_K_MEANS_H
#define CSP_K_MEANS_H

#include <vector>
#include <iostream>
#include <algorithm>

template<typename T>
class Tuple {
    T x;
    T y;
public:
    Tuple(T x, T y) {
        this->x = x;
        this->y = y;
    }
};

class k_means {
private:
    int n;
    int k;
    int iterations;

public:
    k_means(int n, int k, int iterations) {
        this->n = n;
        this->k = k;
        this->iterations = iterations;
    }

    void get_avg_center(std::vector<Tuple<double>> tuples);

    void choose_site_which_minimizes_total_distance();

    void init();

    void partition();
};


#endif //CSP_K_MEANS_H
