#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

using namespace std;
#define DOUBLE_MAX 0x3FFFFFFF
#define eps 1e-6

vector<int> k_list = {5, 10, 20, 50, 100};
int n = 1000;
int k = 100;
double low = 0.0;
double high = 100000.0;
int epoch = 200;

random_device rd;
mt19937 gen(rd());
// 随机数生成器，用于生成site坐标。
uniform_real_distribution<double> dis_double(low, high);
// 随机数生成器，用于随机选择起点。
uniform_int_distribution<int> dis_int(0, n - 1);

unordered_map<int, string> info = {
        {1, "选择的起始节点到其他所有节点的距离之和最小"},
        {2, "选择的起始节点到其他所有节点的距离之和最大"},
        {3, "随机选择起始节点"},
};

class Tuple {
public:
    double x, y;

    Tuple() {
        x = 0;
        y = 0;
    }
};


double get_distance(Tuple s1, Tuple s2) {
    return sqrt((s1.x - s2.x) * (s1.x - s2.x) + (s1.y - s2.y) * (s1.y - s2.y));
}

int select_first_site(vector<Tuple> tuples, int n, int type) {
    // task5-3,选择初始节点。没有优劣之分。
    int index = 0;
    if (type == 1) {
        // 选择的节点到其他所有节点之和最小
        double min_distance = DOUBLE_MAX;
        for (int i = 0; i < n; ++i) {
            double acc_dis = 0;
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                acc_dis += get_distance(tuples[i], tuples[j]);
            }
            if (acc_dis < min_distance) {
                min_distance = acc_dis;
                index = i;
            }
        }
    }
    if (type == 2) {
        //  选择的节点到其他所有节点的最大值最小
        double min_dis = DOUBLE_MAX;
        for (int i = 0; i < n; ++i) {
            double max_dis = 0;
            for (int j = 0; j < n; j++) {
                double dis = get_distance(tuples[i], tuples[j]);
                max_dis = max(max_dis, dis);
            }
            if (max_dis < min_dis) {
                min_dis = max_dis;
                index = i;
            }
        }
    }
    if (type == 3) {
        index = dis_int(gen);
    }

    return index;
}


double get_select_distance(vector<Tuple> sites, vector<Tuple> centers) {
    double max_dis = -1;
    for (auto &site: sites) {
        double min_dis = DOUBLE_MAX;
        // 选择最近的center。
        for (auto &center: centers) {
            double dis = get_distance(site, center);
            min_dis = min(min_dis, dis);
        }
        max_dis = max(max_dis, min_dis);
    }
    return max_dis;
}

vector<Tuple> center_selection(vector<Tuple> sites, int n, int k, int type) {
    vector<Tuple> centers;
    int choose = select_first_site(sites, n, type);
#ifdef DEBUG
    cout << "Choose:" << choose << '\t';
#endif
    vector<bool> vis(n, false);
    vis[choose] = true;

    centers.push_back(sites[choose]);

    while (--k) {
        // 选择距离当前所有centers距离最小值，最大的非center点加入center中。
        int max_distance_index = 0;
        double max_distance = 0;
        for (int i = 0; i < n; ++i) {
            if (vis[i]) {   // 已经访问过的不再考虑
                continue;
            }

            double min_dis = DOUBLE_MAX;
            for (auto &center: centers) {
                double dis = get_distance(sites[i], center);
                min_dis = min(min_dis, dis);
            }
            if (min_dis > max_distance) {
                max_distance_index = i;
                max_distance = min_dis;
            }
        }
        centers.push_back(sites[max_distance_index]);
        vis[max_distance_index] = true;
    }
    return centers;
}

double tsp(vector<Tuple> sites, int n, int k, int type) {
    vector<Tuple> selected_centers = center_selection(sites, n, k, type);
    double distance = get_select_distance(sites, selected_centers);
#ifdef DEBUG
    cout << "type:" << type << '\t' << distance << '\t';
#endif
    return distance;
}


int main() {
//    freopen("../data.txt", "w", stdout);
    for (k = 5; k <= 200; k += 5) {
        cout << k << ' ';
        vector<int> win(3, 0);
        for (int cnt = 1; cnt < epoch; cnt++) {
#ifdef DEBUG
            cout << "cnt:" << cnt << ' ';
#endif
            vector<Tuple> sites(n);
            for (int i = 0; i < n; ++i) {
                sites[i].x = dis_double(gen);
                sites[i].y = dis_double(gen);
            }

            auto dis1 = tsp(sites, n, k, 1);
            auto dis2 = tsp(sites, n, k, 2);
            auto dis3 = tsp(sites, n, k, 3);

            if (dis1 <= dis2 && dis1 <= dis3) {
                win[0]++;
            }

            if (dis2 <= dis1 && dis2 <= dis3) {
                win[1]++;
            }

            if (dis3 <= dis1 && dis3 <= dis2) {
                win[2]++;
            }
        }

        for (int i = 0; i < 3; i++) {
            cout << win[i] << ' ';
        }
        cout << endl;
    }
    return 0;
}
