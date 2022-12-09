import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
# 禁止warning输出
import warnings

warnings.filterwarnings('ignore')


# Based on the training set, you need to train a model out of the following candidate models:
# decision tree,
# nearest neighbor classifier,
# support vector machine,
# neural network as well as
# ensemble models.

def datasets():
    # 读取diabetes/traindata.txt的数据
    x_train = np.loadtxt('diabetes/traindata.txt')
    y_train = np.loadtxt('diabetes/trainlabel.txt')
    x_test = np.loadtxt('diabetes/testdata.txt')
    print(x_train.shape, y_train.shape, x_test.shape)
    # 读取diabetes/testdata.txt的数据
    return x_train, y_train, x_test, None


def neural_network():
    # 用神经网络训练模型
    model = MLPClassifier(max_iter=10000, hidden_layer_sizes=(32, 64, 32))
    model.fit(x_train, y_train)

    # 训练集的准确率
    print('neural_network train accuracy: ', model.score(x_train, y_train))

    # val集的准确率
    print('neural_network val accuracy: ', model.score(x_val, y_val))

    # 换行
    print()

    # 用训练好的模型预测
    y_pred = model.predict(x_test)

    # 将预测结果保存到文件
    np.savetxt('diabetes/predict_NeuralNetwork.txt', y_pred, fmt='%d')


def nearest_neighbor_classifier():
    # 用nearest neighbor classifier训练模型
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)

    # 训练集的准确率
    print('nearest_neighbor_classifier train accuracy: ', model.score(x_train, y_train))

    # val集的准确率
    print('nearest_neighbor_classifier val accuracy: ', model.score(x_val, y_val))

    # 换行
    print()

    # 用训练好的模型预测
    y_pred = model.predict(x_test)

    # 将预测结果保存到文件
    np.savetxt('diabetes/predict_KNN.txt', y_pred, fmt='%d')


def support_vector_machine():
    # 用支持向量机训练模型
    model = SVC()
    model.fit(x_train, y_train)

    # 训练集的准确率
    print('support_vector_machine train accuracy: ', model.score(x_train, y_train))

    # val集的准确率
    print('support_vector_machine val accuracy: ', model.score(x_val, y_val))

    # 换行
    print()

    # 用训练好的模型预测
    y_pred = model.predict(x_test)

    # 将预测结果保存到文件
    np.savetxt('diabetes/predict_SVM.txt', y_pred, fmt='%d')


def decision_tree():
    # 用决策树训练模型
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(x_train, y_train)

    # 训练集的准确率
    print('decision_tree train accuracy: ', model.score(x_train, y_train))

    # val集的准确率
    print('decision_tree val accuracy: ', model.score(x_val, y_val))

    # 换行
    print()

    # 用训练好的模型预测
    y_pred = model.predict(x_test)

    # 将预测结果保存到文件
    np.savetxt('diabetes/predict_DecisionTree.txt', y_pred, fmt='%d')


# 使用上述5个模型做集成学习，使用投票的方式得到最终的结果
def ensemble():
    # 用集成学习的方式训练模型
    model1 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(32, 64, 32))
    model2 = KNeighborsClassifier()
    model3 = SVC()
    model4 = DecisionTreeClassifier(max_depth=5)
    model = VotingClassifier(estimators=[('neural_network', model1), ('nearest_neighbor_classifier', model2),
                                         ('support_vector_machine', model3), ('decision_tree', model4)],
                             voting='hard')
    model.fit(x_train, y_train)

    # 训练集的准确率
    print('ensemble train accuracy: ', model.score(x_train, y_train))

    # val集的准确率
    print('ensemble val accuracy: ', model.score(x_val, y_val))

    # 换行
    print()

    # 用训练好的模型预测
    y_pred = model.predict(x_test)

    # 将预测结果保存到文件
    np.savetxt('diabetes/predict_ensemble.txt', y_pred, fmt='%d')


# main调取datasets函数
if __name__ == '__main__':
    x_train, y_train, x_test, _ = datasets()
    # 统计y_train里面的0.0和1.0的个数
    print('y_train 0.0: ', np.sum(y_train == 0.0))
    print('y_train 1.0: ', np.sum(y_train == 1.0))
    x_train, y_train = SMOTE(random_state=42).fit_resample(x_train, y_train)
    # 统计y_train里面的0.0和1.0的个数
    print('y_train 0.0: ', np.sum(y_train == 0.0))
    print('y_train 1.0: ', np.sum(y_train == 1.0))

    # 将训练集分为训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

    # 处理数据集的imbalanced问题
    decision_tree()
    nearest_neighbor_classifier()
    support_vector_machine()
    neural_network()
    ensemble()
