from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)

clf1 = LogisticRegression(random_state=1)
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = GaussianNB()

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft', weights=[2, 2, 1]
)

for clf, label in zip([clf1, clf2, clf3, eclf],
                      ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Voting Ensemble']):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Accuracy: %0.4f, [%s]" % (score, label))
