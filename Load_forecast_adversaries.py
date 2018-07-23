import tensorflow as tf
import keras
from keras.optimizers import SGD
from Neural_Net_Module import rnn_model
import csv
import matplotlib.pyplot as plt
from utils import *

lr = 0.01
batch_size = 200
nb_epoch = 10
controllable_dim = 16
seq_length = 10
TEMP_MAX = 24
TEMP_MIN = 19
eps=0.03
gamma=90


def scaled_gradient(x, predictions, target):
    loss = tf.square(predictions - target)
    # Take gradient with respect to x_{T}, since it contains all the x value needs to be updated
    grad, = tf.gradients(loss, x)
    signed_grad = tf.sign(grad)
    # Define the gradient of log barrier function on constraints
    #grad_comfort_high = 1 / ((tref_high - tset))
    #grad_comfort_low = 1 / ((tset - tref_low))
    #grad_contrained = grad[:, :, 0:16] + 0.000000001 * (grad_comfort_high + grad_comfort_low)
    return grad, signed_grad



if __name__ == '__main__':
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')

    sess = tf.Session()
    keras.backend.set_session(sess)

    with open('building_data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows = rows[1:43264]
    print("Dataset shape", shape(rows))
    rows = np.array(rows[1:], dtype=float)

    feature_dim = rows.shape[1]
    print("Feature dimension", feature_dim)

    # Normalize the feature and response
    max_value = np.max(rows, axis=0)
    print("Max power values: ", max_value)
    min_value = np.min(rows, axis=0)
    rows2 = (rows - min_value) / (max_value - min_value)

    # Reorganize to the RNN-like sequence
    X_train, Y_train = reorganize(rows2[:, 0:feature_dim - 1], rows2[:, feature_dim - 1], seq_length=seq_length)
    print("Training data shape", shape(X_train))
    print("X_train None:", np.argwhere(np.isnan(X_train)))
    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)

    # Test data: change here for real testing data
    Y_test = np.copy(Y_train[3500:])
    X_test = np.copy(X_train[3500:])
    X_train =  X_train[:35000]
    Y_train = Y_train[:35000]
    print('Number of testing samples', Y_test.shape[0])
    print('Number of training samples', Y_train.shape[0])

    # Define tensor
    x = tf.placeholder(tf.float32, shape=(None, seq_length, feature_dim - 1))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    tset = tf.placeholder(tf.float32, shape=(None, seq_length, controllable_dim))
    target = tf.placeholder(tf.float32, shape=(None, 1))

    # Define the tempture setpoint upper and lower bound
    temp_low = TEMP_MIN * np.ones((1, controllable_dim))  # temp setpoint lowest as 20
    temp_low = (temp_low - min_value[0:controllable_dim]) / (max_value[0:controllable_dim] - min_value[0:controllable_dim])
    temp_high = TEMP_MAX * np.ones((1, controllable_dim))  # temp setpoint highest as 25
    temp_high = (temp_high - min_value[0:controllable_dim]) / (max_value[0:controllable_dim] - min_value[0:controllable_dim])

    # Define the RNN model, establish the graph and SGD solver
    model = rnn_model(seq_length=seq_length, input_dim=feature_dim - 1)
    predictions = model(x)
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    # Fit the RNN model with training data and save the model weight
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True)  # validation_split=0.1
    # model.save_weights('rnn_clean.h5')
    model.load_weights('rnn_clean.h5')
    y_value = model.predict(X_test[0:5000], batch_size=32)

    # Record the prediction result
    with open('predicted_rnn2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(y_value)

    with open('truth_rnn2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(Y_test[0:5000])

    # Plot the prediction result. This is the same as Building_Load_Forecasting.py
    t = np.arange(0, 2016)
    plt.plot(t, Y_test[216:216 + 2016], 'r--', label="True")
    plt.plot(t, y_value[216:216 + 2016], 'b', label="predicted")
    plt.legend(loc='northeast')
    ax = plt.gca()  # grab the current axis
    ax.set_xticks(144 * np.arange(0, 14))  # choose which x locations to have ticks
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
                        "Sun"])  # set the labels to display at those ticks
    plt.title("Building electricity consumption")
    plt.show()
    print("Clean training completed!")
    print("Training percentage error:", np.mean(np.divide(abs(y_value - Y_train[0:5000]), Y_train[0:5000])))
    #model.save_weights('rnn_clean.h5')

    # Optimization step starts here!


    X_new = []
    grad_new = []
    mpc_scope = seq_length
    X_train2 = np.copy(X_test)
    with sess.as_default():
        counter = 0
        # Initialize the SGD optimizer
        grad, sign_grad = scaled_gradient(x, predictions, target)
        for q in range(1000 - seq_length):
            if counter % 100 == 0 and counter > 0:
                print("Optimization Time step" + str(counter))

            # Define the control output target
            #Y_target = (0 * Y_test[counter:counter + mpc_scope]).reshape(-1, 1)
            Y_target =  Y_test[counter:counter + mpc_scope].reshape(-1, 1)

            # upper and lower bound for controllable features
            X_upper_bound = np.tile(temp_high, (mpc_scope, seq_length, 1))
            X_lower_bound = np.tile(temp_low, (mpc_scope, seq_length, 1))

            # Define input: x_t, x_{t+1},...,x_{t+pred_scope}
            X_input = X_train2[counter:counter + mpc_scope]
            #X_input = check_control_constraint(X_input, controllable_dim, X_upper_bound, X_lower_bound)
            X_controllable = X_input[:, :, 0:controllable_dim]
            # the uncontrollable part needs to be replaced by prediction later!!!
            X_uncontrollable = X_input[:, :, controllable_dim:feature_dim - 1]

            X_new_group = X_input
            #print("X_new_group shape", shape(X_new_group))
            gradient_value, signed_grad = sess.run([grad, sign_grad], feed_dict={x: X_new_group,
                                                                          target: Y_target,
                                                                          tset: X_controllable,
                                                                          keras.backend.learning_phase(): 0})
            #print("sign_grad", signed_grad)
            #print(np.shape(signed_grad))
            saliency_mat=np.abs(gradient_value)
            saliency_mat=(saliency_mat>np.percentile(np.abs(gradient_value),[gamma])).astype(int)
            random_num=np.random.randint(0,2)
            if random_num==0:
                X_new_group = X_new_group + np.multiply(eps * signed_grad, saliency_mat)
                #X_new_group = X_new_group + eps * signed_grad
            else:
                X_new_group = X_new_group - np.multiply(eps * signed_grad, saliency_mat)
                #X_new_group = X_new_group - eps * signed_grad

            # check the norm constraints on input
            #X_new_group = check_control_constraint(X_new_group, controllable_dim, X_upper_bound, X_lower_bound)
            y_new_group = model.predict(X_new_group)

            if X_new == []:
                X_new = X_new_group[0].reshape([1, seq_length, feature_dim - 1])
                grad_new = gradient_value[0]
            else:
                X_new = np.concatenate((X_new, X_new_group[0].reshape([1, seq_length, feature_dim - 1])), axis=0)
                grad_new = np.concatenate((grad_new, gradient_value[0]), axis=0)

            # Update the x value in the training data
            X_train2[counter] = X_new_group[0].reshape([1, seq_length, feature_dim - 1])
            for i in range(1, seq_length):
                X_train2[counter + i, 0:seq_length - i, :] = X_train2[counter, i:seq_length, :]

            # Next time step
            counter += 1


    X_new = np.array(X_new, dtype=float)
    print("Adversarial X shape", shape(X_new))
    dime=55
    y_new = model.predict(X_new, batch_size=64)* (max_value[dime] - min_value[dime]) + min_value[dime]
    y_val=model.predict(X_test[:1000], batch_size=32)* (max_value[dime] - min_value[dime]) + min_value[dime]
    y_orig=Y_test[:1000]* (max_value[dime] - min_value[dime]) + min_value[dime]
    plt.plot(y_new,'r')
    plt.plot(y_val, 'g')
    plt.plot(y_orig,'b')
    plt.show()


    print("Adversary Forecast Error:", np.mean(np.clip(np.abs(y_new - y_orig[:990])/y_orig[:990], 0,3)))
    with open('test_true.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(y_orig.reshape(-1,1))

    with open('test_pred.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(y_val.reshape(-1,1))

    with open('test_adversary.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(y_new.reshape(-1,1))


    #Observe the difference on input features and visualize
    deviation_all=0
    '''for dime in range(30):
        X_temp = rows[0:len(X_new), dime]
        X_temp_new = X_new[0:len(X_new), 0, dime] * (max_value[dime] - min_value[dime]) + min_value[dime]
        deviation=np.mean(np.abs(X_temp_new-X_temp)/(X_temp+0.0001))
        print(deviation)
        deviation_all+=deviation
        plt.plot(X_temp, 'r--', label="previous")
        plt.plot(X_temp_new, 'b', label="adversarial")
        plt.show()

    print("The overall input features deviation: ", deviation_all/30.0)'''

    dime=26
    X_temp = rows[0:len(X_new), dime].reshape(-1,1)
    X_temp_new = (X_new[0:len(X_new), 0, dime] * (max_value[dime] - min_value[dime]) + min_value[dime]).reshape(-1,1)
    plt.plot(X_temp, 'r--', label="previous")
    plt.plot(X_temp_new, 'b', label="adversarial")
    plt.show()
    with open('fea10_orig.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(X_temp)

    with open('fea10_adv.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(X_temp_new)





