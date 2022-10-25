#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <set>


using namespace std;
#define DOUBLE_MAX 0x3FFFFFFF
#define eps 1e-6

double threshold = 0.9;
int n = 9;
int k = 4;
int low = 0;
int high = 8;
int epoch = 200;

random_device rd;
mt19937 gen(rd());
// 随机数生成器，用于生成site坐标。
uniform_int_distribution<int> distribution_coordinates(low + 1, high - 1);
// 随机数生成器，用于随机选择起点。
uniform_int_distribution<int> distribution_index(0, n - 1);

unordered_map<int, string> info = {
        {1, "选择的起始节点到其他所有节点的距离之和最小"},
        {2, "选择的起始节点到其他所有节点的距离之和最大"},
        {3, "随机选择起始节点"},
};

class Tuple {
public:
    int x, y;

    Tuple() {
        x = 0;
        y = 0;
    }

    bool operator==(const Tuple other) {
        return this->x == other.x && this->y == other.y;
    }
};


double get_distance(Tuple s1, Tuple s2) {
    return sqrt(1.0 * (s1.x - s2.x) * (s1.x - s2.x) + (s1.y - s2.y) * (s1.y - s2.y));
}

int select_first_site(vector<Tuple> tuples, int tuple_num, int type) {
    // task5-3,选择初始节点。没有优劣之分。
    int index = 0;
    if (type == 1) {
        // 选择的节点到其他所有节点距离之和最小
        double min_distance = DOUBLE_MAX;
        for (int i = 0; i < tuple_num; ++i) {
            double acc_dis = 0;
            for (int j = 0; j < tuple_num; j++) {
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
        for (int i = 0; i < tuple_num; ++i) {
            double max_dis = 0;
            for (int j = 0; j < tuple_num; j++) {
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
//        随机选择起始节点
        index = distribution_index(gen);
    }

    return index;
}

// Minimize Max dist(s, C), 计算该值
// 公式1
double get_select_distance(const vector<Tuple> &tuples, const vector<Tuple> &centers, int type = 1) {
    double result = 0;
    for (auto &site: tuples) {
        double min_dis = DOUBLE_MAX;
        // 选择最近的center。
        for (auto &center: centers) {
            double dis = get_distance(site, center);
            min_dis = min(min_dis, dis);
        }
        if (type == 1) {
            result = max(result, min_dis);
        } else if (type == 2) {
            result += min_dis * min_dis;
        } else if (type == 3) {
            result += min_dis;
        }
    }
    return result;
}

// 公式2

vector<Tuple> center_selection(vector<Tuple> tuples, int tuple_num, int center_num, int type) {
    vector<Tuple> centers;
    int start = select_first_site(tuples, tuple_num, type);
#ifdef DEBUG
    cout << "Choose:" << start << '\t';
#endif
    vector<bool> visit(tuple_num, false);
    visit[start] = true;

    centers.push_back(tuples[start]);

    while (--center_num) {
        // 选择距离当前所有centers距离最小值，最大的非center点加入center中。
        int max_distance_index = 0;
        double max_distance = 0;
        for (int i = 0; i < tuple_num; ++i) {
            if (visit[i]) {   // 已经访问过的不再考虑
                continue;
            }

            double min_dis = DOUBLE_MAX;
            for (auto &center: centers) {
                double dis = get_distance(tuples[i], center);
                min_dis = min(min_dis, dis);
            }
            if (min_dis > max_distance) {
                max_distance_index = i;
                max_distance = min_dis;
            }
        }
        centers.push_back(tuples[max_distance_index]);
        visit[max_distance_index] = true;
    }
    return centers;
}

pair<int, int> select_pair(const vector<Tuple> &sites) {
    pair<int, int> res;
    double min_dis = DOUBLE_MAX;
    for (int i = 0; i < sites.size(); i++) {
        for (int j = i + 1; j < sites.size(); j++) {
            double dis = get_distance(sites[i], sites[j]);
            if (dis < min_dis) {
                res = make_pair(i, j);
                min_dis = dis;
            }
        }
    }
    return res;
}

vector<Tuple>
distance_based_greedy_removal(vector<Tuple> tuples, int tuple_num, int center_num, vector<int> &remove_order) {
    vector<Tuple> centers = tuples;
    while (centers.size() > center_num) {
        auto p = select_pair(centers);
        int a = p.first, b = p.second;
        double min_a = DOUBLE_MAX, min_b = DOUBLE_MAX;
        for (int i = 0; i < centers.size(); i++) {
            if (i != a && i != b) {
                double dis = get_distance(centers[i], centers[a]);
                min_a = min(min_a, dis);
            }
            if (i != b && i != a) {
                double dis = get_distance(centers[i], centers[b]);
                min_b = min(min_b, dis);
            }
        }

        Tuple tmp;
        if (min_a < min_b) {
            tmp = *(centers.begin() + a);
            centers.erase(centers.begin() + a);
        } else {
            tmp = *(centers.begin() + b);
            centers.erase(centers.begin() + b);
        }

        for (int i = 0; i < n; i++) {
            if (tmp == tuples[i]) {
                remove_order.push_back(i);
            }
        }
    }
    return centers;
}

vector<Tuple> greedy_inclusion(const vector<Tuple> tuples, int tuple_num, int center_num) {
    int start = select_first_site(tuples, tuple_num, 2);
    vector<Tuple> centers = {tuples[start]};
    set<int> selected = {start}; // 保存所有当前已经选择的center。
    while (selected.size() < center_num) {
        //  选择center最小化最大距离。
        int selected_index = 0;
        double min_dis = DOUBLE_MAX;
        for (int i = 0; i < tuple_num; ++i) {
            if (selected.find(i) != selected.end()) {
                continue; // 已经选过的center不考虑。
            }
            centers.push_back(tuples[i]);
            double cur_dis = get_select_distance(tuples, centers);
            if (cur_dis < min_dis) {
                min_dis = cur_dis;
                selected_index = i;
            }
            // backtrace
            centers.pop_back();
        }
        centers.push_back(tuples[selected_index]);
        selected.insert(selected_index);
    }
    return centers;
}

vector<Tuple> greedy_removal(vector<Tuple> tuples, int tuple_num, int center_num, vector<int> &remove_order) {
    vector<Tuple> centers = tuples;
    while (centers.size() > center_num) {
        int length = centers.size();
        double min_dis = DOUBLE_MAX;
        int selected_index = 0;
        for (int i = 0; i < length; i++) {
            // 每次删除一个center, 使得目标值最大
            auto tmp = centers[i];
            centers.erase(centers.begin() + i);

            double cur_dis = get_select_distance(tuples, centers);
            if (cur_dis < min_dis) {
                min_dis = cur_dis;
                selected_index = i;
            }
            centers.insert(centers.begin() + i, tmp);
        }

        auto tmp = *(centers.begin() + selected_index);
        for (int i = 0; i < n; i++) {
            if (tmp == tuples[i]) {
                remove_order.push_back(i);
            }
        }

        centers.erase(centers.begin() + selected_index);

    }
    return centers;
}

double csp(vector<Tuple> tuples, int tuple_num, int center_num, int type) {
    vector<Tuple> selected_centers = center_selection(tuples, tuple_num, center_num, type);
    double distance = get_select_distance(tuples, selected_centers);
    return distance;
}

void compare_different_start_selection() {
    for (k = 5; k <= 200; k += 500) {
        cout << k << ' ';
        vector<int> win(3, 0);
        for (int cnt = 1; cnt < epoch; cnt++) {
            vector<Tuple> sites(n);
            for (int i = 0; i < n; ++i) {
                sites[i].x = distribution_coordinates(gen);
                sites[i].y = distribution_coordinates(gen);
            }

            auto dis1 = csp(sites, n, k, 1);
            auto dis2 = csp(sites, n, k, 2);
            auto dis3 = csp(sites, n, k, 3);

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
}


vector<int> convert_to_index(vector<Tuple> sites, vector<Tuple> centers) {
    vector<int> index;
    for (auto center: centers) {
        for (int i = 0; i < sites.size(); i++) {
            if (center == sites[i]) {
                index.push_back(i);
            }
        }
    }
//    sort(index.begin(), index.end());
    return index;
}

void show_centers(vector<int> index) {
    for (auto i: index) {
        cout << i << ' ';
    }
    cout << endl;
}

void show_centers(const vector<Tuple> &centers) {
    for (auto center: centers) {
        cout << '(' << center.x << ' ' << center.y << ") ";
    }
    cout << endl;
}

// 比较两个centers的相似度，约接近1越好。
double compare_two_centers(const vector<int> &index1, const vector<int> &index2) {
    set<int> inter;
    for (int i = 0; i < index1.size(); ++i) {
        inter.insert(index1[i]);
        inter.insert(index2[i]);
    }
    return inter.size() * 1.0 / (index1.size() + index2.size());
}


void save(const vector<Tuple> &sites, int tuple_num, int center_num, vector<vector<int>> centers,
          vector<vector<int>> remove_orders, int cnt, vector<double> distances) {
    freopen("../alg_compare_data28.txt", "w", stdout);

    // 保存配置
    cout << tuple_num << ' ' << center_num << endl;

    // 保存所有城市的下标, x坐标, y坐标。
    for (int i = 0; i < tuple_num; i++) {
        cout << i << ' ' << sites[i].x << ' ' << sites[i].y << endl;
    }

    // 依次保存四个算法的centers
    for (int i = 0; i < centers.size(); i++) {
        for (int j = 0; j < centers[i].size(); j++) {
            cout << centers[i][j] << (j == centers[i].size() - 1 ? '\n' : ' ');
        }
        if (remove_orders[i].size() == 0) {
            cout << endl;
        }
        for (int j = 0; j < remove_orders[i].size(); j++) {
            cout << remove_orders[i][j] << (j == remove_orders[i].size() - 1 ? '\n' : ' ');
        }
    }

    cout << cnt << endl;

    for (int i = 0; i < distances.size(); i++) {
        cout << distances[i] << (i == distances.size() - 1 ? '\n' : ' ');
    }
}


// 返回当前最小的交集。
bool task6_1() {
    vector<Tuple> sites(n);
    set<pair<int, int>> coordinates; // 坐标去重
    for (int i = 0; i < n; ++i) {
        int x = distribution_coordinates(gen);
        int y = distribution_coordinates(gen);
        while (coordinates.find(make_pair(x, y)) != coordinates.end()) {
            x = distribution_coordinates(gen);
            y = distribution_coordinates(gen);
        }
        sites[i].x = x;
        sites[i].y = y;
        coordinates.insert(make_pair(x, y));
    }

    auto alg1_centers = center_selection(sites, n, k, 2);
    auto centers1 = convert_to_index(sites, alg1_centers);
    auto dis1 = get_select_distance(sites, alg1_centers);

    vector<int> alg2_remove_order;
    auto alg2_centers = distance_based_greedy_removal(sites, n, k, alg2_remove_order);
    auto centers2 = convert_to_index(sites, alg2_centers);
    auto dis2 = get_select_distance(sites, alg2_centers);

    auto alg3_centers = greedy_inclusion(sites, n, k);
    auto centers3 = convert_to_index(sites, alg3_centers);
    auto dis3 = get_select_distance(sites, alg3_centers);

    vector<int> alg4_remove_order;
    auto alg4_centers = greedy_removal(sites, n, k, alg4_remove_order);
    auto centers4 = convert_to_index(sites, alg4_centers);
    auto dis4 = get_select_distance(sites, alg4_centers);

    vector<vector<int>> centers = {centers1, centers2, centers3, centers4};
    vector<double> distances = {dis1, dis2, dis3, dis4};
    vector<vector<int>> remove_orders = {{}, alg2_remove_order, {}, alg4_remove_order};

    bool ok = false;
    set<int> cnt;
    for (int i = 0; i < centers.size(); i++) {
        for (int j = 0; j < centers[i].size(); j++) {
            cnt.insert(centers[i][j]);
        }
    }

    set<double> dis_cnt;
    for (auto dis: distances) {
        dis_cnt.insert(dis);
    }

    if (cnt.size() >= min(n, 4 * k - 1) && dis_cnt.size() >= 4) {
        ok = true;
    }

    if (ok) {
        save(sites, n, k, centers, remove_orders, cnt.size(), distances);
    }
    return ok;
}

void task6_3_1() {
    vector<int> xs = {1, 3, 9};
    double min_dis = DOUBLE_MAX;
    double res = 0;
    for (double i = 0; i < 10; i += 0.001) {
        double max_dis = 0;
        for (auto x: xs) {
            max_dis = max(max_dis, abs(x - i));
        }
        if (max_dis < min_dis) {
            min_dis = max_dis;
            res = i;
        }
    }
    cout << min_dis << ' ' << res << endl;
}

void task6_3_2() {
    vector<int> xs = {1, 3, 9};
    double min_dis = DOUBLE_MAX;
    double res = 0;
    for (double i = 0; i < 10; i += 0.001) {
        double max_dis = 0;
        for (auto x: xs) {
            max_dis += (abs((x - i) * (x - i)));
        }
        if (max_dis < min_dis) {
            min_dis = max_dis;
            res = i;
        }
    }
    cout << min_dis << ' ' << res << endl;
}

void task6_3_3() {
    vector<int> xs = {1, 3, 9};
    double min_dis = DOUBLE_MAX;
    double res = 0;
    for (double i = 0; i < 10; i += 0.001) {
        double max_dis = 0;
        for (auto x: xs) {
            max_dis += abs((x - i));
        }
        if (max_dis < min_dis) {
            min_dis = max_dis;
            res = i;
        }
    }
    cout << min_dis << ' ' << res << endl;
}

void task6_4_2() {
    vector<int> xs = {1, 4, 5, 6, 9};
    double min_dis = DOUBLE_MAX;
    double res1 = 0, res2 = 0;
    for (double i = 1; i < 5; i += 0.001) {
        for (double j = 5; j < 10; j += 0.001) {
            double max_dis = 0;
            for (auto x: xs) {
                double dis1 = (x - i) * (x - i);
                double dis2 = (x - j) * (x - j);
                max_dis += min(dis1, dis2);
            }
            if (max_dis <= min_dis) {
                min_dis = max_dis;
                res1 = i;
                res2 = j;
            }
        }
    }

    cout << min_dis << ' ' << res1 << ' ' << res2 << endl;
}


int main() {
    task6_4_2();
//    task6_3_1();
//    task6_3_2();
//    task6_3_3();
//    int cnt = 0;
//    while (++cnt) {
//        cout << cnt << endl;
//        auto ok = task6_1();
//        if (ok) break;
//    }

//    compare_different_start_selection();
    return 0;
}
