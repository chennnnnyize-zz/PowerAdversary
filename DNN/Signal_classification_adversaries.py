import tensorflow as tf
import keras
from keras.optimizers import SGD
from Neural_Net_Module import dnn_model
import csv
from numpy import shape
import numpy as np
import matplotlib.pyplot as plt

batch_size=32
nb_epoch=15
eps=0.5
gamma=80

def scaled_gradient(x, y, predictions):
    #loss: the mean of loss(cross entropy)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    grad, = tf.gradients(loss, x)
    signed_grad = tf.sign(grad)
    return grad, signed_grad

if __name__ == '__main__':
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')

    sess = tf.Session()
    keras.backend.set_session(sess)


    with open('normal.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows=np.array(rows, dtype=float)
    data=rows
    label=np.zeros((200,1))
    with open('sag.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows=np.array(rows, dtype=float)
    data=np.concatenate((data, rows))
    labels=np.ones((200,1))
    label=np.concatenate((label, labels))
    with open('distortion.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows=np.array(rows, dtype=float)
    data=np.concatenate((data, rows))
    labels=2*np.ones((200,1))
    label=np.concatenate((label, labels))
    with open('impulse.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows=np.array(rows, dtype=float)
    data=np.concatenate((data, rows))
    labels=3*np.ones((200,1))
    label=np.concatenate((label, labels))
    label=label.reshape(-1,1)
    label=keras.utils.to_categorical(label, num_classes=None)
    print("Input label shape", shape(label))
    print("Input data shape", shape(data))

    index = np.arange(len(label))
    np.random.shuffle(index)
    label = label[index]
    data = data[index]

    trX=data[:600]
    trY=label[:600]
    teX=data[600:]
    teY=label[600:]

    x = tf.placeholder(tf.float32, shape=(None, 1000))
    y = tf.placeholder(tf.float32, shape=(None, 4))

    model = dnn_model(input_dim=1000)
    predictions = model(x)
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(trX, trY, batch_size=batch_size, epochs=nb_epoch, shuffle=True)  # validation_split=0.1
    # model.save_weights('dnn_clean.h5')
    score = model.evaluate(teX, teY, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    with sess.as_default():
        adv_sample=[]
        counter = 0
        # Initialize the SGD optimizer
        grad, sign_grad = scaled_gradient(x, y, predictions)
        for q in range(200):
            if counter % 50 == 0 and counter > 0:
                print("Attack on samples" + str(counter))
            X_new_group=np.copy(teX[counter])
            gradient_value, signed_grad = sess.run([grad, sign_grad], feed_dict={x: X_new_group.reshape(-1,1000),
                                                                                 y: teY[counter].reshape(-1,4),
                                                                                 keras.backend.learning_phase(): 0})
            saliency_mat = np.abs(gradient_value)
            saliency_mat = (saliency_mat > np.percentile(np.abs(gradient_value), [gamma])).astype(int)
            X_new_group = X_new_group + np.multiply(eps * signed_grad, saliency_mat)
            adv_sample.append(X_new_group)
            '''print("Ground truth", teY[counter])
            print(model.predict(teX[counter].reshape(-1, 1000)))
            print(model.predict(X_new_group.reshape(-1,1000)))
            plt.plot(teX[counter])
            plt.show()
            plt.plot(X_new_group.reshape(-1,1),'r')
            plt.show()'''


            counter+=1
        adv_sample=np.array(adv_sample, dtype=float).reshape(-1,1000)
        score=model.evaluate(adv_sample, teY, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        teY_pred=np.argmax(model.predict(teX, batch_size=32), axis=1)
        adv_pred=np.argmax(model.predict(adv_sample, batch_size=32), axis=1)
        '''with open('test_true.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.argmax(teY, axis=1).reshape(-1, 1))

        with open('test_pred.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(teY_pred.reshape(-1, 1))

        with open('test_adversary.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(adv_pred.reshape(-1, 1))'''

