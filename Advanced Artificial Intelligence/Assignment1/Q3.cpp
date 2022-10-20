//
// Created by Qi Luo on 2022/10/18.
//
#include <iostream>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <set>

using namespace std;

void show_edge_map(vector<vector<int>> edge_map, int n) {
    for (int i = 1; i <= n; i++) {
        if (edge_map[i].empty()) continue;
        cout << "City " << i << " has edges to : ";
        for (int j: edge_map[i]) {
            cout << j << ' ';
        }
        cout << endl;
    }
    cout << endl;
}

vector<int> edge_recombination_crossover(vector<vector<int>> edge_map, int n, int start) {
    cout << "Edge Recombination crossover: start with " << start << endl;
    vector<int> res;
    res.push_back(start);
    while (res.size() < n) {
        edge_map[start].clear();
        int next = 0, length = INT_MAX;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < edge_map[i].size(); j++) {
                if (edge_map[i][j] == start) {
                    edge_map[i].erase(edge_map[i].begin() + j);
                    if (edge_map[i].size() <= length) {
                        next = i;
                        length = edge_map[i].size();
                    }
                    break;
                }
            }
        }
        cout << "City " << start << " is selected" << endl;
        show_edge_map(edge_map, n);
        res.push_back(next);
        start = next;
    }

    cout << "final tour: ";
    for (int re: res) {
        cout << re << ' ';
    }
    cout << endl;
    return res;
}

void construct_edge_map(vector<vector<int>> &edge_map, int n, vector<int> parent_1, vector<int> parent_2) {
    unordered_map<int, set<int>> mp;
    for (int i = 0; i < n; i++) {
        int pre_index = (i == 0 ? n - 1 : i - 1);
        int next = (i == n - 1 ? 0 : i + 1);

        mp[parent_1[i]].insert(parent_1[pre_index]);
        mp[parent_1[i]].insert(parent_1[next]);

        mp[parent_2[i]].insert(parent_2[pre_index]);
        mp[parent_2[i]].insert(parent_2[next]);
    }

    for (auto &[key, vals]: mp) {
        for (auto &val: vals) {
            edge_map[key].push_back(val);
        }
    }
    show_edge_map(edge_map, n);
}

void Q3() {
    cout << "Question 3:" << endl;
    int n = 8;
    vector<vector<int>> edge_map(n + 1);
    vector<int> parent_1 = {1, 3, 6, 5, 4, 2, 8, 7};
    vector<int> parent_2 = {1, 4, 2, 3, 6, 5, 7, 8};
    construct_edge_map(edge_map, n, parent_1, parent_2);
    edge_recombination_crossover(edge_map, n, 1);
}