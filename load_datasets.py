import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.datasets import fetch_covtype, fetch_kddcup99, fetch_mldata, make_blobs
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from keras.datasets import mnist as load_mnist


def synthetic(size=10, random_state=1):
    n_samples = int(950*size)
    centers = 1
    num_features = 512
    normal, y = make_blobs(n_samples=n_samples, n_features=num_features, cluster_std=1.0,
                           centers=centers, shuffle=True, random_state=random_state)
    mu, sigma = 0, 5.0
    np.random.seed(random_state)
    anomalies = np.random.normal(mu, sigma, (int(50*size), num_features))

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((normal, anomalies), axis=0))
    normal = scaler.transform(normal)
    anomalies = scaler.transform(anomalies)

    test_normal = normal[int(len(normal) / 2):]
    normal = normal[:int(len(normal) / 2)]

    test_anomalies = anomalies[int(len(anomalies) / 2):]
    anomalies = anomalies[:int(len(anomalies) / 2)]

    x_train = np.concatenate((normal, anomalies), axis=0)
    y_train = np.concatenate(([1] * len(normal), [-1] * len(anomalies)), axis=0)
    x_train, y_train = shuffle(x_train, y_train, random_state=1)

    x_test = np.concatenate((test_normal, test_anomalies), axis=0)
    y_test = np.concatenate(([1] * len(test_normal), [-1] * len(test_anomalies)), axis=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=1)

    return x_train, y_train, x_test, y_test, test_normal, test_anomalies


def kddcup(percent10, random_state=1):
    data = fetch_kddcup99(percent10=percent10)

    x = data.data
    y_ori = data.target
    y = np.array([1 if l == b'normal.' else -1 for l in y_ori])
    labelencoder_x_1 = LabelEncoder()
    labelencoder_x_2 = LabelEncoder()
    labelencoder_x_3 = LabelEncoder()
    x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
    x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
    x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])
    onehotencoder_1 = OneHotEncoder(categorical_features=[1])
    x = onehotencoder_1.fit_transform(x).toarray()
    onehotencoder_2 = OneHotEncoder(categorical_features=[4])
    x = onehotencoder_2.fit_transform(x).toarray()
    onehotencoder_3 = OneHotEncoder(categorical_features=[70])
    x = onehotencoder_3.fit_transform(x).toarray()

    normal = x[np.where(y == 1)]
    anomalies = x[np.where(y == -1)]
    anomalies = shuffle(anomalies, random_state=1)
    anomalies = anomalies[:int(len(normal)/19)]

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((normal, anomalies), axis=0))
    normal = scaler.transform(normal)
    anomalies = scaler.transform(anomalies)

    x = np.concatenate((normal, anomalies), axis=0)
    y = np.concatenate(([1] * len(normal), [-1] * len(anomalies)), axis=0)
    x, y = shuffle(x, y, random_state=random_state)

    normal = x[np.where(y == 1)]
    test_normal = normal[int(len(normal) / 2):]
    normal = normal[:int(len(normal) / 2)]

    anomalies = x[np.where(y == -1)]
    test_anomalies = anomalies[int(len(anomalies) / 2):]
    anomalies = anomalies[:int(len(anomalies) / 2)]

    x_train = np.concatenate((normal, anomalies), axis=0)
    y_train = np.concatenate(([1] * len(normal), [-1] * len(anomalies)), axis=0)
    x_train, y_train = shuffle(x_train, y_train, random_state=1)

    x_test = np.concatenate((test_normal, test_anomalies), axis=0)
    y_test = np.concatenate(([1] * len(test_normal), [-1] * len(test_anomalies)), axis=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=1)

    return x_train, y_train, x_test, y_test


def usps(random_state=1, normal_num=1, anomaly_num=7):
    data = fetch_mldata('USPS')

    x = data.data
    x = MinMaxScaler().fit_transform(x)
    y = data.target

    normal = x[np.where(y == normal_num + 1)][:950]
    anomalies = x[np.where(y == anomaly_num + 1)][:50]

    x = np.concatenate((normal, anomalies), axis=0)
    y = np.concatenate(([1]*len(normal), [-1]*len(anomalies)), axis=0)
    x, y = shuffle(x, y, random_state=random_state)

    normal = x[np.where(y == 1)]
    test_normal = normal[int(len(normal)/2):]
    normal = normal[:int(len(normal)/2)]

    anomalies = x[np.where(y == -1)]
    test_anomalies = anomalies[int(len(anomalies)/2):]
    anomalies = anomalies[:int(len(anomalies)/2)]

    x_train = np.concatenate((normal, anomalies), axis=0)
    y_train = np.concatenate(([1]*len(normal), [-1]*len(anomalies)), axis=0)
    x_train, y_train = shuffle(x_train, y_train, random_state=1)

    x_test = np.concatenate((test_normal, test_anomalies), axis=0)
    y_test = np.concatenate(([1]*len(test_normal), [-1]*len(test_anomalies)), axis=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=1)

    return x_train, y_train, x_test, y_test


def forestcover(random_state=1):
    data = fetch_covtype()

    x = data.data
    x = MinMaxScaler().fit_transform(x)
    y = data.target
    y = np.array([1 if l == 2 else -1 if l == 4 else 0 for l in y])

    normal = x[np.where(y == 1)]
    anomalies = x[np.where(y == -1)]

    x = np.concatenate((normal, anomalies), axis=0)
    y = np.concatenate(([1]*len(normal), [-1]*len(anomalies)), axis=0)
    x, y = shuffle(x, y, random_state=random_state)

    normal = x[np.where(y == 1)]
    test_normal = normal[int(len(normal)/2):]
    normal = normal[:int(len(normal)/2)]

    anomalies = x[np.where(y == -1)]
    test_anomalies = anomalies[int(len(anomalies)/2):]
    anomalies = anomalies[:int(len(anomalies)/2)]

    x_train = np.concatenate((normal, anomalies), axis=0)
    y_train = np.concatenate(([1]*len(normal), [-1]*len(anomalies)), axis=0)
    x_train, y_train = shuffle(x_train, y_train, random_state=1)

    x_test = np.concatenate((test_normal, test_anomalies), axis=0)
    y_test = np.concatenate(([1]*len(test_normal), [-1]*len(test_anomalies)), axis=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=1)

    return x_train, y_train, x_test, y_test


def shuttle(random_state=1):
    data = fetch_mldata('shuttle')

    x = data.data
    x = MinMaxScaler().fit_transform(x)
    y = data.target
    y = np.array([1 if l == 1 else -1 if l in [2, 3, 5, 6, 7] else 0 for l in y])

    normal = x[np.where(y == 1)]
    anomalies = x[np.where(y == -1)]

    x = np.concatenate((normal, anomalies), axis=0)
    y = np.concatenate(([1]*len(normal), [-1]*len(anomalies)), axis=0)
    x, y = shuffle(x, y, random_state=random_state)

    normal = x[np.where(y == 1)]
    test_normal = normal[int(len(normal)/2):]
    normal = normal[:int(len(normal)/2)]

    anomalies = x[np.where(y == -1)]
    test_anomalies = anomalies[int(len(anomalies)/2):]
    anomalies = anomalies[:int(len(anomalies)/2)]

    x_train = np.concatenate((normal, anomalies), axis=0)
    y_train = np.concatenate(([1]*len(normal), [-1]*len(anomalies)), axis=0)
    x_train, y_train = shuffle(x_train, y_train, random_state=1)

    x_test = np.concatenate((test_normal, test_anomalies), axis=0)
    y_test = np.concatenate(([1]*len(test_normal), [-1]*len(test_anomalies)), axis=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=1)

    return x_train, y_train, x_test, y_test


def musk(random_state=1):
    data = sio.loadmat('musk.mat')

    x = np.array(data['X'])
    x = MinMaxScaler().fit_transform(x)

    y = np.array(data['y']).T[0]
    y = np.array([1 if label == 0 else -1 for label in y])

    normal = x[np.where(y == 1)]
    anomalies = x[np.where(y == -1)]

    x = np.concatenate((normal, anomalies), axis=0)
    y = np.concatenate(([1]*len(normal), [-1]*len(anomalies)), axis=0)
    x, y = shuffle(x, y, random_state=random_state)

    normal = x[np.where(y == 1)]
    test_normal = normal[int(len(normal)/2):]
    normal = normal[:int(len(normal)/2)]

    anomalies = x[np.where(y == -1)]
    test_anomalies = anomalies[int(len(anomalies)/2):]
    anomalies = anomalies[:int(len(anomalies)/2)]

    x_train = np.concatenate((normal, anomalies), axis=0)
    y_train = np.concatenate(([1]*len(normal), [-1]*len(anomalies)), axis=0)
    x_train, y_train = shuffle(x_train, y_train, random_state=1)

    x_test = np.concatenate((test_normal, test_anomalies), axis=0)
    y_test = np.concatenate(([1]*len(test_normal), [-1]*len(test_anomalies)), axis=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=1)

    return x_train, y_train, x_test, y_test


def mnist(random_state=1, normal_classes=[4], anomaly_classes=[0, 7, 9], anomaly_count=100):
    (x_train, x_trainLabels), (x_test, x_testLabels) = load_mnist.load_data()

    labels = x_trainLabels
    data = x_train
    data = np.reshape(data, (len(data), 784))
    data = MinMaxScaler().fit_transform(data)
    data, labels = shuffle(data, labels, random_state=random_state)
    
    normal_set = []
    for c in normal_classes:
        normal_set.append(data[np.where(labels == c)])

    normal = np.concatenate(normal_set, axis=0)
    normal = np.reshape(normal, (len(normal), 784))
    
    anomaly_set = []
    for c in anomaly_classes:
        anomaly_set.append(data[np.where(labels == c)])

    anomalies = np.concatenate(anomaly_set, axis=0)

    anomalies = shuffle(anomalies, random_state=1)[:anomaly_count]
    anomalies = np.reshape(anomalies, (len(anomalies), 784))

    test_normal = normal[int(len(normal)/2):]
    normal = normal[:int(len(normal)/2)]

    test_anomalies = anomalies[int(len(anomalies)/2):]
    anomalies = anomalies[:int(len(anomalies)/2)]

    x_train = np.concatenate((normal, anomalies), axis=0)
    y_train = np.concatenate(([1]*len(normal), [-1]*len(anomalies)), axis=0)
    x_train, y_train = shuffle(x_train, y_train, random_state=1)

    x_test = np.concatenate((test_normal, test_anomalies), axis=0)
    y_test = np.concatenate(([1]*len(test_normal), [-1]*len(test_anomalies)), axis=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=1)

    return x_train, y_train, x_test, y_test
