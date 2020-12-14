import sys
sys.path.append("../")

from utils import *

import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    assert len(data["user_id"]) == len(data["question_id"])
    assert len(data["user_id"]) == len(data["is_correct"])
    length = len(data["user_id"])
    for s in range(length):
        i = data["user_id"][s]
        j = data["question_id"][s]
        cij = data["is_correct"][s]
        
        theta_i = theta[i]
        beta_j = beta[j]

        theta_i_minus_beta_j = theta_i - beta_j
        log_lklihood += cij*theta_i_minus_beta_j - np.log(1.0+np.exp(theta_i_minus_beta_j))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    dtheta = []
    dbeta = []
    
    N = len(theta)
    M = len(beta)
    
    theta_mid = {x: [] for x in range(N)}
    beta_mid = {x: [] for x in range(M)}
    
    assert len(data["user_id"]) == len(data["question_id"])
    assert len(data["user_id"]) == len(data["is_correct"])
    
    length = len(data["user_id"])
    for s in range(length):
        i = data["user_id"][s]
        j = data["question_id"][s]
        cij = data["is_correct"][s]
        theta_i = theta[i]
        beta_j = beta[j]
        theta_mid[i].append(cij - sigmoid(theta_i-beta_j))
        beta_mid[j].append(-1*cij+sigmoid(theta_i-beta_j))
    
    for i in range(N):
        dtheta.append(-1*sum(theta_mid[i]))
    for j in range(M):
        dbeta.append(-1*sum(beta_mid[j]))

    theta -= lr * np.array(dtheta, dtype=np.float32)
    beta -= lr * np.array(dbeta, dtype=np.float32)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.array([0 for i in range(542)], dtype=np.float32)
    beta = np.array([0 for j in range(1774)], dtype=np.float32)

    train_acc_lst = []
    val_acc_lst = []
    train_nlld = []
    val_nlld = []
    train_lld = []
    val_lld = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_nlld.append(train_neg_lld)
        val_nlld.append(val_neg_lld)
        train_lld.append(-1*train_neg_lld)
        val_lld.append(-1*val_neg_lld)

        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)
        
        train_score = evaluate(data=data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)

        print("Step: {} \n Train NLLK: {}, Train Score: {} \n Val NLLK: {}, Val Score: {}".format(i, train_neg_lld, train_score, val_neg_lld, val_score))
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_lst, val_acc_lst, train_nlld, val_nlld, train_lld, val_lld


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    # print(sparse_matrix.shape)
    # print(sparse_matrix)
    val_data = load_valid_csv("../data")
    # print(len(val_data["user_id"]))
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    NUM_ITERS = 70
    LEARNING_RATE = 0.005
    theta, beta, train_acc_lst, val_acc_lst, train_nlld, val_nlld, train_lld, val_lld = irt(train_data, val_data, LEARNING_RATE, NUM_ITERS)
    # print(theta)
    
    plt.figure(figsize=(18, 6))
    x = [i for i in range(NUM_ITERS)]
    plt.subplot(1, 3, 1)
    train_line, = plt.plot(x, train_acc_lst)
    val_line, = plt.plot(x, val_acc_lst)
    plt.title("Accuracy")
    l1 = plt.legend([train_line, val_line], ["train", "validation"])

    plt.subplot(1, 3, 2)
    train_line, = plt.plot(x, train_nlld)
    val_line, = plt.plot(x, val_nlld)
    plt.title("Negative log-likelihood")
    l2 = plt.legend([train_line, val_line], ["train", "validation"])

    plt.subplot(1, 3, 3)
    train_line, = plt.plot(x, train_lld)
    val_line, = plt.plot(x, val_lld)
    plt.title("log-likelihood")
    l3 = plt.legend([train_line, val_line], ["train", "validation"])
    plt.show()

    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Validation Accuracy is: {} \n Test Accuracy is: {}".format(val_acc, test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    sample_questions = random.sample([i for i in range(1774)], 5)
    theta = np.linspace(-5.0, 5.0, 100)
    lines = []
    for q in sample_questions:
        q_beta = beta[q]
        p = sigmoid(theta-q_beta)
        line, = plt.plot(theta, p, label="Question " + str(q) + " with beta {:.2f}".format(q_beta))
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
