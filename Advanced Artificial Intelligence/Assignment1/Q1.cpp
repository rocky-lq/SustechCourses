//
// Created by 64374 on 2022/10/18.
//
#include <iostream>
#include <unordered_map>
#include <vector>
#include <queue>
#include <functional>

using namespace std;

unordered_map<char, int> name_relation_map_q1 = {{'S', 1},
                                                 {'A', 2},
                                                 {'B', 3},
                                                 {'C', 4},
                                                 {'D', 5},
                                                 {'E', 6},
                                                 {'F', 7},
                                                 {'G', 8}};

void show_path(vector<int> path, string title) {
    cout << title << endl;
    for (int i: path) {
        char name = 'S';
        for (auto &[key, val]: name_relation_map_q1) {
            if (val == i) {
                name = key;
            }
        }
        cout << name << ' ';
    }
    cout << endl;
}

int calculate_cost(vector<vector<int>> adjacency_matrix, vector<int> path) {
    int cost = 0;
    for (int i = 1; i < path.size(); ++i) {
        cost += adjacency_matrix[path[i - 1]][path[i]];
    }
    return cost;
}

// BFS
int breath_first_search(vector<vector<int>> adjacency_matrix, int n, int start, int goal) {
    queue<int> q;
    q.push(start);
    vector<bool> visit(n + 1, false);
    visit[start] = true;
    vector<int> path;
    // 但是如何记录正确的路径？
    while (!q.empty()) {
        for (int i = 0; i < q.size(); i++) {
            auto cur = q.front();
            path.push_back(cur);
            q.pop();
            if (cur == goal) {
                break;
            }

            for (int j = 1; j <= n; j++) {
                if (!visit[j] && adjacency_matrix[cur][j] != -1) {
                    q.push(j);
                    visit[j] = true;
                }
            }
        }
    }

    int cost = calculate_cost(adjacency_matrix, path);
    string title = "Depth-First Search, Cost is " + to_string(cost);
    show_path(path, title);
    return true;
}

// DFS
int depth_first_search(vector<vector<int>> adjacency_matrix, int n, int start, int goal) {
    vector<bool> visit(n + 1, false);

    vector<int> path = {start};
    int min_cost = INT_MAX;
    visit[start] = true;

    function<void(int, vector<int>)> dfs = [&](int cur, vector<int> tmp) {
        if (!tmp.empty() && tmp.back() == goal) {
            // 这里可以打印所有的DFS输出。
            int cost = calculate_cost(adjacency_matrix, tmp);
            if (cost < min_cost) {
                path = tmp;
                min_cost = cost;
            }
            return;
        }
        // back track
        for (int i = 1; i <= n; i++) {
            if (!visit[i] && adjacency_matrix[cur][i] != -1) {
                visit[i] = true;
                cout << cur << ' ' << i << endl;
                tmp.push_back(i);
                dfs(i, tmp);
//                tmp.pop_back();
//                visit[i] = false;
            }
        }
    };

    dfs(start, vector<int>{start});

    // failure
    if (path.back() != goal) {
        return false;
    }

    int cost = calculate_cost(adjacency_matrix, path);
    string title = "Depth-First Search, Cost is " + to_string(cost);
    show_path(path, title);
    return true;
}

void uniform_first_search(vector<vector<int>> adjacency_matrix) {

}

void Q1() {


    freopen("../in1.txt", "r", stdin);
    int n, edge_num;
    cin >> n >> edge_num;
    vector<vector<int>> adjacency_matrix(n + 1, vector<int>(n + 1, -1));
    for (int i = 0; i < edge_num; i++) {
        char from, to;
        int val;
        cin >> from >> to >> val;
        int a = name_relation_map_q1[from];
        int b = name_relation_map_q1[to];
        adjacency_matrix[a][b] = val;
        adjacency_matrix[b][a] = val;
    }

    breath_first_search(adjacency_matrix, n, 1, 8);
    depth_first_search(adjacency_matrix, n, 1, 8);

}