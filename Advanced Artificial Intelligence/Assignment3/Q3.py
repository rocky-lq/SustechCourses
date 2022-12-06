import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


def ensemble_models():
    # 用emsemble方法训练模型
    # 读取预测结果
    y_pred1 = np.loadtxt('diabetes/predict_NeuralNetwork.txt')
    y_pred2 = np.loadtxt('diabetes/predict_KNN.txt')
    y_pred3 = np.loadtxt('diabetes/predict_SVM.txt')
    y_pred4 = np.loadtxt('diabetes/predict_DecisionTree.txt')
    y_pred5 = np.loadtxt('diabetes/predict_LogisticRegression.txt')

    # 将预测结果进行投票
    y_pred = np.zeros(y_pred1.shape)
    for i in range(y_pred1.shape[0]):
        y_pred[i] = np.argmax(np.bincount([y_pred1[i], y_pred2[i], y_pred3[i], y_pred4[i], y_pred5[i]]))

    # 将预测结果保存到文件
    np.savetxt('diabetes/predict_ensemble.txt', y_pred, fmt='%d')


def neural_network():
    # 用神经网络训练模型
    model = MLPClassifier()
    model.fit(x_train, y_train)

    # 训练集的准确率
    print('neural_network train accuracy: ', model.score(x_train, y_train))

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

    # 用训练好的模型预测
    y_pred = model.predict(x_test)

    # 将预测结果保存到文件
    np.savetxt('diabetes/predict_SVM.txt', y_pred, fmt='%d')


def decision_tree():
    # 用决策树训练模型
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    # 训练集的准确率
    print('decision_tree train accuracy: ', model.score(x_train, y_train))

    # 用训练好的模型预测
    y_pred = model.predict(x_test)

    # 将预测结果保存到文件
    np.savetxt('diabetes/predict_DecisionTree.txt', y_pred, fmt='%d')


def regression():
    # 用逻辑回归模型训练
    model = LogisticRegression(max_iter=10000)
    model.fit(x_train, y_train)

    # 训练集的准确率
    print('regression train accuracy: ', model.score(x_train, y_train))

    # 用训练好的模型预测
    y_pred = model.predict(x_test)

    # 将预测结果保存到文件
    np.savetxt('diabetes/predict_LogisticRegression.txt', y_pred, fmt='%d')


# main调取datasets函数
if __name__ == '__main__':
    x_train, y_train, x_test, _ = datasets()
    regression()
    decision_tree()
    support_vector_machine()
    nearest_neighbor_classifier()
    neural_network()
    ensemble_models()
