//
// Created by 64374 on 2022/10/18.
//
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;
unordered_map<char, int> name_relation_map_q2 = {{'O', 1},
                                                 {'A', 2},
                                                 {'B', 3},
                                                 {'C', 4},
                                                 {'D', 5},
                                                 {'E', 6}};

void crossover() {

}

void mutation() {

}

void Q2() {

    freopen("../in2.txt", "r", stdin);
    int n, edge_num;
    cin >> n >> edge_num;
    vector<vector<int>> adjacency_matrix(n + 1, vector<int>(n + 1, -1));
    for (int i = 0; i < edge_num; i++) {
        char from, to;
        int val;
        cin >> from >> to >> val;
        int a = name_relation_map_q2[from];
        int b = name_relation_map_q2[to];
        adjacency_matrix[a][b] = val;
        adjacency_matrix[b][a] = val;
    }
}