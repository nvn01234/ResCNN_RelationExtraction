import numpy as np
from sklearn.metrics import f1_score


def test(testing_data, input_x, input_p1, input_p2, s, p, dropout_keep_prob, datamanager, sess, num_epoch):
    x_test = datamanager.generate_x(testing_data)
    p1, p2 = datamanager.generate_p(testing_data)
    y_test = datamanager.generate_y(testing_data)
    y_true = np.argmax(y_test, axis=1)
    scores, pre = sess.run([s, p], {input_x: x_test, input_p1:p1, input_p2:p2, dropout_keep_prob: 1.0})
    print(np.shape(scores))
    print(np.shape(pre))
    y_pred = np.argmax(scores)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f1)
