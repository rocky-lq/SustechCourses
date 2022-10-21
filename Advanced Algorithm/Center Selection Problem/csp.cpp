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
int n = 20;
int k = 5;
int low = 0;
int high = 20;
int epoch = 200;

random_device rd;
mt19937 gen(rd());
// 随机数生成器，用于生成site坐标。
uniform_int_distribution<int> distribution_coordinates(low, high);
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
double get_select_distance(const vector<Tuple> &sites, const vector<Tuple> &centers) {
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
    int start = select_first_site(sites, n, type);
#ifdef DEBUG
    cout << "Choose:" << start << '\t';
#endif
    vector<bool> visit(n, false);
    visit[start] = true;

    centers.push_back(sites[start]);

    while (--k) {
        // 选择距离当前所有centers距离最小值，最大的非center点加入center中。
        int max_distance_index = 0;
        double max_distance = 0;
        for (int i = 0; i < n; ++i) {
            if (visit[i]) {   // 已经访问过的不再考虑
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

vector<Tuple> distance_based_greedy_removal(vector<Tuple> sites, int tuple_num, int center_num) {
    vector<Tuple> centers = sites;
    while (centers.size() > center_num) {
        auto p = select_pair(centers);
        int a = p.first, b = p.second;
        double min_a = DOUBLE_MAX, min_b = DOUBLE_MAX;
        for (int i = 0; i < centers.size(); i++) {
            if (i != a) {
                double dis = get_distance(centers[i], centers[a]);
                min_a = min(min_a, dis);
            }
            if (i != b) {
                double dis = get_distance(centers[i], centers[b]);
                min_b = min(min_b, dis);
            }
        }
        if (min_a < min_b) {
            centers.erase(centers.begin() + a);
        } else {
            centers.erase(centers.begin() + b);
        }
    }
    return centers;
}

vector<Tuple> greedy_inclusion(const vector<Tuple> sites, int tuple_num, int center_num) {
    int start = select_first_site(sites, tuple_num, 2);
    vector<Tuple> centers = {sites[start]};
    set<int> selected = {start}; // 保存所有当前已经选择的center。
    while (selected.size() < center_num) {
        //  选择center最小化最大距离。
        int selected_index = 0;
        double min_dis = DOUBLE_MAX;
        for (int i = 0; i < tuple_num; ++i) {
            if (selected.find(i) != selected.end()) {
                continue; // 已经选过的center不考虑。
            }
            centers.push_back(sites[i]);
            double cur_dis = get_select_distance(sites, centers);
            if (cur_dis < min_dis) {
                min_dis = cur_dis;
                selected_index = i;
            }
            // backtrace
            centers.pop_back();
        }
        centers.push_back(sites[selected_index]);
        selected.insert(selected_index);
    }
    return centers;
}

vector<Tuple> greedy_removal(vector<Tuple> sites, int tuple_num, int center_num) {
    vector<Tuple> centers = sites;
    while (centers.size() > center_num) {
        int length = centers.size();
        double min_dis = DOUBLE_MAX;
        int selected_index = 0;
        for (int i = 0; i < length; i++) {
            // 每次删除一个center
            auto tmp = centers[i];
            centers.erase(centers.begin() + i);

            double cur_dis = get_select_distance(sites, centers);
            if (cur_dis < min_dis) {
                min_dis = cur_dis;
                selected_index = i;
            }
            centers.insert(centers.begin() + i, tmp);
        }
        centers.erase(centers.begin() + selected_index);
    }
    return centers;
}

double csp(const vector<Tuple> sites, int n, int k, int type) {
    vector<Tuple> selected_centers = center_selection(sites, n, k, type);
    double distance = get_select_distance(sites, selected_centers);
#ifdef DEBUG
    cout << "type:" << type << '\t' << distance << '\t';
#endif
    return distance;
}

void compare_different_start_selection() {
    for (k = 5; k <= 200; k += 500) {
        cout << k << ' ';
        vector<int> win(3, 0);
        for (int cnt = 1; cnt < epoch; cnt++) {
#ifdef DEBUG
            cout << "cnt:" << cnt << ' ';
#endif
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
    set<int> intersections;
    for (int i = 0; i < index1.size(); ++i) {
        intersections.insert(index1[i]);
        intersections.insert(index2[i]);
    }
    return intersections.size() * 1.0 / (index1.size() + index2.size());
}


void save(const vector<Tuple> &sites, int n, int k, vector<vector<int>> centers, int cnt) {
    freopen("../alg_compare_data.txt", "w", stdout);

    // 保存配置
    cout << n << ' ' << k << endl;

    // 保存所有城市的下标, x坐标, y坐标。
    for (int i = 0; i < n; i++) {
        cout << i << ' ' << sites[i].x << ' ' << sites[i].y << endl;
    }

    // 依次保存四个算法的centers
    for (auto &center: centers) {
        for (int j = 0; j < center.size(); j++) {
            cout << center[j] << (j == center.size() - 1 ? '\n' : ' ');
        }
    }

    cout << cnt << endl;
}


// 返回当前最小的交集。
bool create_samples() {
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

    auto alg2_centers = distance_based_greedy_removal(sites, n, k);
    auto centers2 = convert_to_index(sites, alg2_centers);

    auto alg3_centers = greedy_inclusion(sites, n, k);
    auto centers3 = convert_to_index(sites, alg3_centers);

    auto alg4_centers = greedy_removal(sites, n, k);
    auto centers4 = convert_to_index(sites, alg4_centers);

    vector<vector<int>> centers = {centers1, centers2, centers3, centers4};

    bool ok = false;
    set<int> cnt;
    for (int i = 0; i < centers.size(); i++) {
        for (int j = 0; j < centers[i].size(); j++) {
            cnt.insert(centers[i][j]);
        }
    }
    if (cnt.size() >= n - 1) {
        ok = true;
    }

    if (ok) {
        save(sites, n, k, centers, cnt.size());
    }
    return ok;
}


int main() {
    int cnt = 0;
    while (++cnt) {
        auto ok = create_samples();
        cout << cnt << endl;
        if (ok) break;
    }

//    compare_different_start_selection();
    return 0;
}
