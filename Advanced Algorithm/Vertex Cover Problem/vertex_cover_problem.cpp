//
// Created by Luo Qi on 2022/11/10.
//
#include <iostream>
#include <algorithm>
#include <utility>
#include <vector>
#include <cmath>
#include <set>
#include <random>

using namespace std;

class Edges {

};

class Vertex {
public:
    Vertex(int id, double weight, vector<int> edges) {
        this->weight = weight;
        this->id = id;
        this->edges = std::move(edges);
    }

public:
    double weight;
    int id;
    vector<int> edges;  // all connected vertex id.

};

random_device rd;
mt19937 gen(rd());

set<set<int>> whole_set; // contain whole edges.

pair<vector<int>, double> greedy(vector<Vertex> vertexs) {
    vector<int> result;
    set<set<int>> cur_set;
    set<int> visited;
    double cost = 0;
    bool ok = false;
    // Stop after traversing all vertexs or finding a solution
    while (!ok || visited.size() == vertexs.size()) {
        int index = -1;
        double avg = numeric_limits<double>::max();

        for (int i = 0; i < vertexs.size(); i++) {
            if (visited.find(vertexs[i].id) != visited.end()) {
                continue; // Skipped if already visited
            } else {
                double v_cost = vertexs[i].weight;
                auto num = vertexs[i].edges.size();
                if (num == 0) {
                    visited.insert(vertexs[i].id);
                    continue;
                }

                double cur_avg = v_cost / num;
                if (cur_avg < avg) {
                    avg = cur_avg;
                    index = i;
                }

            }
        }
        if (index == -1) {
            break;
        }

        visited.insert(vertexs[index].id);
        result.push_back(vertexs[index].id);
        cost += vertexs[index].weight;

        for (auto edge: vertexs[index].edges) {
            set<int> tmp = {vertexs[index].id, edge};
            cur_set.insert(tmp);

            // 删除已经加入的
            for (int j = 0; j < vertexs.size(); j++) {
                if (j == index) continue;
                for (int k = 0; k < vertexs[j].edges.size(); k++) {
                    if (vertexs[j].edges[k] == edge) {
                        vertexs[j].edges.erase(vertexs[j].edges.begin() + k);
                    }
                }
            }
        }

        if (cur_set == whole_set) {
            ok = true;
        }
    }

    if (!ok) cost = -1;
    return make_pair(result, cost);
}

pair<vector<int>, double> pricing(vector<Vertex> vertexs) {
    // 瞎几把选算法。维护一个tight数组即可。
    // choose the tight edge one by one.

    vector<int> result;
    double min_cost = std::numeric_limits<double>::max();
    return make_pair(result, min_cost);
}

pair<vector<int>, double> optimal(vector<Vertex> vertexs) {
    vector<int> result;
    double min_cost = std::numeric_limits<double>::max();
    auto m = vertexs.size();
    auto total_space = pow(2, m);

    bool ok = false;

    for (int i = 1; i < total_space; i++) {
        set<set<int>> cur_set;
        vector<int> tmp;
        double cost = 0.0;
        for (int j = 0; j < m; j++) {
            if (i >> j & 1) {
                for (auto edge: vertexs[j].edges) {
                    set<int> edge_set = {vertexs[j].id, edge};
                    cur_set.insert(edge_set);
                }
                cost += vertexs[j].weight;
                tmp.push_back(vertexs[j].id);
            }
        }

        if (cur_set == whole_set && cost < min_cost) {
            min_cost = cost;
            result = tmp;
            ok = true;
        }
    }
    // if there is no solution.
    if (!ok) min_cost = -1;
    return make_pair(result, min_cost);
}

void lecture_test() {
    freopen("../input.txt", "r", stdin);
    // input style
    vector<Vertex> vertexs;
    int vertex_num, total_edge_num;
    cin >> vertex_num >> total_edge_num;
    for (int i = 0; i < vertex_num; i++) {
        double weight;
        int edge_num;
        cin >> weight >> edge_num;

        vector<int> edges;
        for (int j = 0; j < edge_num; j++) {
            int v;
            cin >> v;
            edges.push_back(v);

            set<int> edge_set = {i + 1, v};
            whole_set.insert(edge_set); // Construct the full collection。
        }
        Vertex tmp(i + 1, weight, edges);
        vertexs.push_back(tmp);
    }

    auto [optimal_vec, optimal_cost] = optimal(vertexs);
    cout << "Vertex Cover Problem:" << endl;
    cout << "Optimal solution:" << endl;
    cout << "The cost is " << optimal_cost << endl;
    cout << "The vertexs is : [ ";
    for (auto m: optimal_vec) {
        cout << m << ' ';
    }
    cout << "]" << endl << endl << endl;

}

int main() {
    lecture_test();
    return 0;
}