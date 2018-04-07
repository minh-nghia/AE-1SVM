from pprint import pprint

import numpy as np
import tensorflow as tf
import time
from collections import Counter

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import matplotlib
import matplotlib.pyplot as plt

from Refactor.load_datasets import synthetic
from Refactor.metrics import metrics
from Refactor.models.AE1SVM import AEOneClassSVM
from Refactor.models.DEC import DEC
from Refactor.models.RDA import RobustL21Autoencoder

tf.set_random_seed(2018)

x_train, y_train, x_test, y_test, x_test_normal, x_test_anomalies = synthetic(size=1, random_state=1)

counter = Counter(y_train)
print('Anomalies ratio:', 100*counter[-1]/(counter[1]+counter[-1]), '%')

autoencoder_layers = [512, 128, 32]
batch_size = 32

data_input = tf.placeholder(tf.float32, shape=[None, 512])

ae1svm = AEOneClassSVM(data_input, batch_size, 'test', autoencoder_layers[1:], 0.4, 1e3, 3.0, 500,
                       autoencoder_activation='sigmoid',
                       full_op=tf.train.AdamOptimizer(1e-2),
                       svm_op=tf.train.AdamOptimizer(1e-2))

ae_only = AEOneClassSVM(data_input, batch_size, 'test', autoencoder_layers[1:], 0.4, 1e3, 3.0, 500,
                        autoencoder_activation='sigmoid', ae_op=tf.train.AdamOptimizer(1e-2))

plt.figure(figsize=(20, 10))

matplotlib.rc('font', size=14)

gs = matplotlib.gridspec.GridSpec(2, 3)

# Train conventional OCSVM
print('OCSVM-RBF')
libsvm = OneClassSVM(nu=0.07, verbose=True, shrinking=True)
t0 = time.time()
libsvm.fit(x_train)
print('Train time:', time.time() - t0)

t0 = time.time()
out_y = libsvm.predict(x_test)
print('Test time:', time.time() - t0)
pprint(metrics(y_test, out_y))
splt = plt.subplot(gs[0])
splt.hist(libsvm.decision_function(x_test_normal), color='blue', alpha=0.5, bins=50, label='Normal')
splt.hist(libsvm.decision_function(x_test_anomalies), color='red', alpha=0.5, bins=50, label='Anomaly')
splt.set_title('OC-SVM RBF')
splt.legend(fontsize=18)

# Train conventional OCSVM
print('OCSVM-Linear')
libsvm = OneClassSVM(nu=0.07, verbose=True, shrinking=True, kernel='linear')
t0 = time.time()
libsvm.fit(x_train)
print('Train time:', time.time() - t0)
t0 = time.time()
out_y = libsvm.predict(x_test)
print('Test time:', time.time() - t0)
pprint(metrics(y_test, out_y))
splt = plt.subplot(gs[1])
splt.hist(libsvm.decision_function(x_test_normal), color='blue', alpha=0.5, bins=50)
splt.hist(libsvm.decision_function(x_test_anomalies), color='red', alpha=0.5, bins=50)
splt.set_title('OC-SVM Linear')

# Train Isolation Forest
print('IsolationForest')
iforest = IsolationForest(contamination=0.05, verbose=1)
t0 = time.time()
iforest.fit(x_train)
print('Train time:', time.time() - t0)

t0 = time.time()
out_y = iforest.predict(x_test)
print('Test time:', time.time() - t0)
pprint(metrics(y_test, out_y))
splt = plt.subplot(gs[2])
splt.hist(iforest.decision_function(x_test_normal), color='blue', alpha=0.5, bins=50)
splt.hist(iforest.decision_function(x_test_anomalies), color='red', alpha=0.5, bins=50)
splt.set_title('Isolation Forest')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Autoencoder-OneclassSVM
    t0 = time.time()
    ae1svm.fit(sess, x_train, x_train, y_train, epochs_1=20, epochs_2=0)
    print('Train time:', time.time() - t0)

    t0 = time.time()
    out_y = ae1svm.predict(sess, x_test)
    print('Test time:', time.time() - t0)

    pprint(metrics(y_test, out_y))

    # Train autoencoder for conventional methods
    t0 = time.time()
    ae_only.fit_ae(sess, x_train, epochs=6)
    print('AE time:', time.time() - t0)

    splt = plt.subplot(gs[3])
    splt.hist(ae1svm.decision_function(sess, x_test_normal), color='blue', alpha=0.5, bins=50)
    splt.hist(ae1svm.decision_function(sess, x_test_anomalies), color='red', alpha=0.5, bins=50)
    splt.set_title('AE-1SVM')

    x_train_encoded = ae_only.encode(sess, x_train)
    x_test_encoded = ae_only.encode(sess, x_test)

    x_train_rff = ae_only.encode_rff(sess, x_train)
    x_test_rff = ae_only.encode_rff(sess, x_test)

    # Robust Deep Autoencoder
    rae = RobustL21Autoencoder(sess=sess, lambda_=2, layers_sizes=autoencoder_layers, learning_rate=1e-2)
    t0 = time.time()
    L, S = rae.fit(x_train, sess=sess, inner_iteration=20, iteration=5, verbose=True, batch_size=batch_size)
    print('Train time:', time.time() - t0)

    t0 = time.time()
    L_test, S_test = rae.predict(x_test, sess=sess)
    print('Test time:', time.time() - t0)

    s_sum = np.linalg.norm(S, axis=1)
    s_sum_test = np.linalg.norm(S_test, axis=1)
    out_y = [1 if s == 0 else -1 for s in s_sum_test]
    pprint(metrics(y_test, out_y))

    splt = plt.subplot(gs[4])
    splt.hist(s_sum_test[np.where(y_test == 1)], color='blue', alpha=0.5, bins=50)
    splt.hist(s_sum_test[np.where(y_test == -1)], color='red', alpha=0.5, bins=50)
    splt.set_yscale('log')
    splt.set_title('RDA')

    dec = DEC(dims=autoencoder_layers, n_clusters=5)
    t0 = time.time()
    dec.pretrain(x=x_train, epochs=10)
    dec.compile(loss='kld')
    y_pred = dec.fit(x_train, update_interval=10, batch_size=batch_size)
    print('Train time:', time.time() - t0)

    t0 = time.time()
    scores = dec.cluster_score(x_test)
    print('Test time:', time.time() - t0)
    threshold = np.partition(scores.flatten(), int(counter[-1]))[int(counter[-1])]
    out_y = [1 if s > threshold else -1 for s in scores]
    pprint(metrics(y_test, out_y))
    splt = plt.subplot(gs[5])
    splt.hist(scores[np.where(y_test == 1)], color='blue', alpha=0.5, bins=50)
    splt.hist(scores[np.where(y_test == -1)], color='red', alpha=0.5, bins=50)
    splt.set_yscale('log')
    splt.set_title('DEC')

proxy = [plt.Rectangle((0, 0), 1, 1, fc='blue'), plt.Rectangle((0, 0), 1, 1, fc='red')]
plt.savefig('synthetic_histograms.eps', bbox_inches='tight')
