//
// Created by 64374 on 2022/11/6.
//
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <set>
#include <random>
#include <algorithm>

using namespace std;

const double eps = 1e-6;

class Machine {
public:
    float cost;
    set<int> jobs;
    int id;

public:
    Machine(int id, float cost, set<int> jobs) {
        this->id = id; // id就等于下标加1。
        this->cost = cost;
        this->jobs = std::move(jobs);
    }

    void print() {
        cout << "Machine id:" << id << " Machine cost " << cost << endl;
        for (auto i: jobs) {
            cout << i << ' ';
        }
        cout << endl;
    }
};

random_device rd;
mt19937 gen(rd());

set<int> whole_set;

pair<vector<int>, float> greedy(vector<Machine> machines) {
    set<int> cur_set;
    vector<int> result;
    set<int> visited;
    float cost = 0;
    bool ok = false; // 存在合理解
    while (!ok || visited.size() != machines.size()) {
        int index = -1;
        float avg = numeric_limits<float>::max();
        // 找出当前代价最小的machine
        for (int i = 0; i < machines.size(); i++) {
            if (visited.find(machines[i].id) != visited.end()) {
                continue;
            } else {
                float m_cost = machines[i].cost;
                auto num = machines[i].jobs.size();
                if (num == 0) {
                    visited.insert(machines[i].id);
                    continue;
                }
                float cur_avg = m_cost / num;
                if (cur_avg < avg) {
                    avg = cur_avg;
                    index = i;
                }
            }
        }

        if (index == -1) {
            break;
        }
        // 更新参数。
        set<int> tmp = machines[index].jobs;
        visited.insert(machines[index].id);
        result.push_back(machines[index].id);
        cost += machines[index].cost;
        cur_set.insert(tmp.begin(), tmp.end());
        if (cur_set == whole_set) {
            ok = true;
        }

        // 删除重复部分，集合取差集。
        for (auto &machine: machines) {
            if (visited.find(machine.id) != visited.end()) {
                continue;
            }
            for (int iter: tmp) {
                auto it = machine.jobs.find(iter);
                if (it != machine.jobs.end()) {
                    machine.jobs.erase(it);
                }
            }
        }

    }
    if (!ok) cost = -1;
    return make_pair(result, cost);
}

pair<vector<int>, float> optimal(vector<Machine> machines) {
    vector<int> result;
    auto m = machines.size();
    auto total_space = pow(2, m);
    float min_cost = 1000000;

    bool ok = false;
    // 遍历所有可能选取的情况。
    for (int i = 1; i < total_space; i++) {
        set<int> cur_set;
        vector<int> tmp;
        float cost = 0.0;
        for (int j = 0; j < m; ++j) {
            if (i >> j & 1) {
                cur_set.insert(machines[j].jobs.begin(), machines[j].jobs.end());
                cost += machines[j].cost;
                tmp.push_back(machines[j].id);
            }
        }
        if (cur_set == whole_set && cost < min_cost) {
            min_cost = cost;
            result = tmp;
            ok = true;
        }
    }
    if (!ok) min_cost = -1;
    return make_pair(result, min_cost);
}

double harmonic_function(int n) {
    double res = 0;
    for (int i = 1; i <= n; i++) {
        res += 1.0 / i;
    }
    return res;
}

vector<Machine> init(int machine_num, int job_num) {
    vector<Machine> machines;
    uniform_int_distribution<int> dis_job(1, job_num - 1);
    uniform_int_distribution<int> dis_job_index(1, job_num);
    uniform_real_distribution<float> dis_cost(1, 10);

    for (int i = 0; i < machine_num; i++) {
        float cost = dis_cost(gen);
        int a = dis_job(gen);
        set<int> tmp;
        while (a--) {
            int index = dis_job_index(gen);
            tmp.insert(index);
        }
        Machine machine = Machine(i + 1, cost, tmp);
        machines.push_back(machine);
    }
    return machines;
}

bool task8_1() {
    int machine_num = 4;
    int job_num = 4;
    for (int i = 1; i <= job_num; i++) {
        whole_set.insert(i);
    }
    double threshold = 2.3;

    auto machines = init(machine_num, job_num);

    int d_star = 0;
    for (auto machine: machines) {
        d_star = max(d_star, (int) machine.jobs.size());
    }
    threshold = harmonic_function(d_star);

    auto [optimal_vec, optimal_cost] = optimal(machines);
    if (optimal_cost == -1) {
        cout << "result not exist, continue next;" << endl;
        return false;
    }
    auto [greedy_vec, greedy_cost] = greedy(machines);

    set<int> optimal_set;
    for (auto m: optimal_vec) {
        optimal_set.insert(m);
    }

    set<int> greedy_set;
    for (auto m: greedy_vec) {
        greedy_set.insert(m);
    }

    set<int> intersection;
    set<int> convergence;
    // 取交集
    set_intersection(optimal_set.begin(), optimal_set.end(), greedy_set.begin(), greedy_set.end(),
                     inserter(intersection, intersection.begin()));

    // 取并集。
    set_union(optimal_set.begin(), optimal_set.end(), greedy_set.begin(), greedy_set.end(),
              inserter(convergence, convergence.begin()));


    if (greedy_cost >= optimal_cost * threshold - eps) { // task8-1-2
//    if (greedy_cost > optimal_cost * threshold && intersection.empty() && convergence.size() == machine_num) { // task8-1-1
        cout << "Set Cover Problem:" << endl;
        cout << "Optimal solution:" << endl;
        cout << "The cost is " << optimal_cost << endl;
        cout << "The machines is : [ ";
        for (auto m: optimal_vec) {
            cout << m << ' ';
        }
        cout << "]" << endl << endl << endl;

        cout << "Greedy solution:" << endl;
        cout << "The cost is " << greedy_cost << endl;
        cout << "The machines is : [ ";
        for (auto m: greedy_vec) {
            cout << m << ' ';
        }
        cout << "]" << endl << endl;

        for (auto &machine: machines) {
            machine.print();
        }
        return true;
    }
    return false;
}

int main() {
    int cnt = 0;
    while (++cnt && !task8_1()) {
        cout << cnt << endl;
    }
    return 0;
/*
    // 测试样例。
    // freopen("../input.txt", "r", stdin);
    int machine_num, job_num;
    cin >> machine_num >> job_num;
    vector<Machine> machines;
    // 构建 job 全集
    for (int i = 1; i <= job_num; i++) {
        whole_set.insert(i);
    }

    for (int i = 0; i < machine_num; i++) {
        float cost;
        int cover_num;
        cin >> cost >> cover_num;
        set<int> tmp;
        while (cover_num--) {
            int job;
            cin >> job;
            tmp.insert(job);
        }
        Machine machine = Machine(i + 1, cost, tmp);
        machines.push_back(machine);
    }
*/
}

