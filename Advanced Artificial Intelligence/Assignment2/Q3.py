import csv

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = csv.reader(open('input3.csv'))

mp = {
    'hazel': 1,
    'brown': 2,
    'blue': 3,
    'black': 1,
    'blond': 3,
    'Europe': 1,
    'Asia': 2,
    'America': 3
}
X = []
y = []
for line in data:
    if line[0] == 'name':
        continue
    # print(line)
    x = line[1:5]
    # print(x)
    x[0] = int(x[0])
    x[1] = int(x[1])
    x[2] = mp[x[2]]
    x[3] = mp[x[3]]
    X.append(x)
    yt = mp[line[-1]]
    y.append(yt)

print(X)
print(y)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

res = clf.predict(X)
print(res)

feature_names = ['height', 'weight', 'eye-color', 'hair-color']
target_names = ['Europe', 'Asia', 'America']
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=target_names,
                                filled=True, rounded=True,
                                special_characters=True)

with open("Q3.gv", 'w') as f:
    f.write(dot_data)
