import sys
sys.path.append("../")
from utils import *
import random
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import copy
import torch

import random
import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import grad

import pandas as pd

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
    return -log_lklihood

# def loss_auto_grad(data, c, theta, beta, k):
#   def theta


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

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_nlld.append(train_neg_lld)
        val_nlld.append(val_neg_lld)

        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)
        
        train_score = evaluate(data=data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)

        print("Step: {} \n Train NLLK: {}, Train Score: {} \n Val NLLK: {}, Val Score: {}".format(i, train_neg_lld, train_score, val_neg_lld, val_score))
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_lst, val_acc_lst, train_nlld, val_nlld


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


def loss(data, c, theta, beta, k):
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
        k_j = k[j]

        useful_term = k_j*(theta_i - beta_j)
        term1 = cij*np.log(c+(1-c)*sigmoid(useful_term))
        term2 = (1-cij)*np.log(1-c-(1-c)*sigmoid(useful_term))
        new_term = term1+term2
        log_lklihood += new_term
    return -log_lklihood


def update_params(data, c, lr, theta, beta, k):
    dtheta = []
    dbeta = []
    dk = []
    
    N = len(theta)
    M = len(beta)
    
    theta_mid = {x: [] for x in range(N)}
    beta_mid = {x: [] for x in range(M)}
    k_mid = {x: [] for x in range(M)}
    
    assert len(data["user_id"]) == len(data["question_id"])
    assert len(data["user_id"]) == len(data["is_correct"])
    
    length = len(data["user_id"])
    for s in range(length):
        i = data["user_id"][s]
        j = data["question_id"][s]
        cij = data["is_correct"][s]
        theta_i = theta[i]
        beta_j = beta[j]
        k_j = k[j]
        useful_term = k_j*(theta_i - beta_j)

        theta_term1 = cij*(1-c)*sigmoid(useful_term)*(1-sigmoid(useful_term))*k_j/(c+(1-c)*sigmoid(useful_term))
        theta_term2 = (1-cij)*(-1*(1-c)*sigmoid(useful_term)*(1-sigmoid(useful_term))*k_j)/(1-c-(1-c)*sigmoid(useful_term))
        
        beta_term1 = cij*(1-c)*sigmoid(useful_term)*(1-sigmoid(useful_term))*(-1)*k_j/(c+(1-c)*sigmoid(useful_term))
        beta_term2 = (1-cij)*(-1)*(1-c)*sigmoid(useful_term)*(1-sigmoid(useful_term))*(-1)*k_j/(1-c-(1-c)*sigmoid(useful_term))

        k_term1 = cij*(1-c)*sigmoid(useful_term)*(1-sigmoid(useful_term))*(theta_i-beta_j)/(c+(1-c)*sigmoid(useful_term))
        k_term2 = (1-cij)*(-1)*(1-c)*sigmoid(useful_term)*(1-sigmoid(useful_term))*(theta_i-beta_j)/(1-c-(1-c)*sigmoid(useful_term))

        theta_mid[i].append(theta_term1+theta_term2)
        beta_mid[j].append(beta_term1+beta_term2)
        k_mid[j].append(k_term1+k_term2)
    
    for i in range(N):
        dtheta.append(-1*sum(theta_mid[i]))
    for j in range(M):
        dbeta.append(-1*sum(beta_mid[j]))
    for j in range(M):
        dk.append(-1*sum(k_mid[j]))

    theta -= lr * np.array(dtheta, dtype=np.float32)
    beta -= lr * np.array(dbeta, dtype=np.float32)
    k -= lr * np.array(dk, dtype=np.float32)

    return theta, beta, k


def new_eval(data, c, theta, beta, k):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (k[q]*(theta[u] - beta[q])).sum()
        p_a = sigmoid(x)*(1-c)+c
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def new_irt(data, val_data, c, lr, iterations, theta, beta, k):
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

    train_acc_lst = []
    val_acc_lst = []
    train_nlld = []
    val_nlld = []

    for i in range(iterations):
        train_neg_lld = loss(data, c, theta, beta, k)
        val_neg_lld = loss(val_data, c, theta, beta, k)
        train_nlld.append(train_neg_lld)
        val_nlld.append(val_neg_lld)

        val_score = new_eval(val_data, c, theta, beta, k)
        val_acc_lst.append(val_score)
        
        train_score = new_eval(data, c, theta, beta, k)
        train_acc_lst.append(train_score)

        print("Step: {} \n Train NLLK: {}, Train Score: {} \n Val NLLK: {}, Val Score: {}".format(i, train_neg_lld, train_score, val_neg_lld, val_score))
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, k = update_params(data, c, lr, theta, beta, k)
        # print(theta, beta, k)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, k, train_acc_lst, val_acc_lst, train_nlld, val_nlld


def calc_correctness_percentage(data):
  length = len(data['question_id'])
  result = {i:{"total":0, "correct":0} for i in range(1774)}
  for s in range(length):
    if data["is_correct"][s] == 1:
      result[data["question_id"][s]]["correct"] += 1
    result[data["question_id"][s]]["total"] += 1
  ret = []
  for i in range(1774):
    if result[i]["total"] == 0:
      ret.append(1.)
    else:
      ret.append(result[i]["correct"]/result[i]["total"])
  return np.array(ret)


def load_question_meta(path):
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    
    # Return a dictionary whose keys are question_id and values are the category of that question
    data = {}
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                question_id = int(row[0])
                cate = eval(row[1])
                if question_id in data:
                    print("GG")
                if question_id not in data:
                    data[question_id] = []
                data[question_id].extend(cate)
            except ValueError:
                pass
    return data

def split_data(data):
    number = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    algebra = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    geometry = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    statistics = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    other = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    NUMBER = 1
    ALGEBRA = 17
    GEOMETRY = 39
    STATISTICS = 68

    number_questions = []
    algebra_questions = []
    geometry_questions = []
    statistics_questions = []
    other_questions = []
    question_meta = load_question_meta("../data/question_meta.csv")

    for question_id in question_meta:
        cate = question_meta[question_id]
        flag = False
        if NUMBER in cate:
            number_questions.append(question_id)
            flag = True
        if ALGEBRA in cate:
            algebra_questions.append(question_id)
            flag = True
        if GEOMETRY in cate:
            geometry_questions.append(question_id)
            flag = True
        if STATISTICS in cate:
            statistics_questions.append(question_id)
            flag = True
        if not flag:
            other_questions.append(question_id)

    length = len(data["question_id"])
    for s in range(length):
        curr_question = data["question_id"][s]
        flag = False
        if curr_question in number_questions:
            number["user_id"].append(data["user_id"][s])
            number["question_id"].append(curr_question)
            number["is_correct"].append(data["is_correct"][s])
            flag = True
        if curr_question in geometry_questions:
            geometry["user_id"].append(data["user_id"][s])
            geometry["question_id"].append(curr_question)
            geometry["is_correct"].append(data["is_correct"][s])
            flag = True
        if curr_question in algebra_questions:
            algebra["user_id"].append(data["user_id"][s])
            algebra["question_id"].append(curr_question)
            algebra["is_correct"].append(data["is_correct"][s])
            flag = True
        if curr_question in statistics_questions:
            statistics["user_id"].append(data["user_id"][s])
            statistics["question_id"].append(curr_question)
            statistics["is_correct"].append(data["is_correct"][s])
            flag = True
        if not flag:
            # print(curr_question)
            other["user_id"].append(data["user_id"][s])
            other["question_id"].append(curr_question)
            other["is_correct"].append(data["is_correct"][s])
    return number, algebra, geometry, statistics, other

def test_cluster_model(data, c, theta, beta, k):
    number_questions = []
    algebra_questions = []
    geometry_questions = []
    statistics_questions = []
    other_questions = []
    question_meta = load_question_meta("../data/question_meta.csv")

    for question_id in question_meta:
        cate = question_meta[question_id]
        flag = False
        if NUMBER in cate:
            number_questions.append(question_id)
            flag = True
        if ALGEBRA in cate:
            algebra_questions.append(question_id)
            flag = True
        if GEOMETRY in cate:
            geometry_questions.append(question_id)
            flag = True
        if STATISTICS in cate:
            statistics_questions.append(question_id)
            flag = True
        if not flag:
            other_questions.append(question_id)
    length = len(data["user_id"])
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        if q in number_questions:
            curr_theta = theta["number"]
            curr_beta = beta["number"]
            curr_k = k["number"]
        elif q in algebra_questions:
            curr_theta = theta["algebra"]
            curr_beta = beta["algebra"]
            curr_k = k["algebra"]
        elif q in geometry_questions:
            curr_theta = theta["geometry"]
            curr_beta = beta["geometry"]
            curr_k = k["geometry"]
        elif q in statistics_questions:
            curr_theta = theta["statistics"]
            curr_beta = beta["statistics"]
            curr_k = k["statistics"]
        else:
            curr_theta = theta["other"]
            curr_beta = beta["other"]
            curr_k = k["other"]
        x = (curr_k[q]*(curr_theta[u]-curr_beta[q])).sum()
        p_a = sigmoid(x)*(1-c)+c
        pred.append(p_a >= 0.5)
    return np.sum(test_data["is_correct"] == np.array(pred)) \
            / len(test_data["is_correct"])


def train_category(train, valid, iterations, lr, c):
  correctness = (0.1+calc_correctness_percentage(train))*10
  init_theta = np.array([0 for i in range(542)], dtype=np.float32)
  init_beta = np.array([0 for j in range(1774)], dtype=np.float32)
  init_k = correctness
  return new_irt(train, valid, c, lr, iterations, init_theta, init_beta, init_k)


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    # print(sparse_matrix.shape)
    # print(sparse_matrix)
    val_data = load_valid_csv("../data")
    # print(len(val_data["user_id"]))
    test_data = load_public_test_csv("../data")


    number_train, algebra_train, geometry_train, statistics_train, other_train = split_data(train_data)
    number_valid, algebra_valid, geometry_valid, statistics_valid, other_valid = split_data(val_data)
    number_test, algebra_test, geometry_test, statistics_test, other_test = split_data(test_data)

    print("Algebra:")
    alg_theta, alg_beta, alg_k, train_acc_lst, val_acc_lst, train_nlld, val_nlld = train_category(algebra_train, algebra_valid, 20, 0.0015, 0.25)
    print("test acc:")
    print(new_eval(algebra_test, 0.25, alg_theta, alg_beta, alg_k))
    print("--------------------------------------")

    print("Number:")
    num_theta, num_beta, num_k, train_acc_lst, val_acc_lst, train_nlld, val_nlld = train_category(number_train, number_valid, 20, 0.002, 0.25)
    print("test acc:")
    print(new_eval(number_test, 0.25, num_theta, num_beta, num_k))
    print("--------------------------------------")

    print("Statistics:")
    sta_theta, sta_beta, sta_k, train_acc_lst, val_acc_lst, train_nlld, val_nlld = train_category(statistics_train, statistics_valid, 23, 0.0015, 0.25)
    print("test acc:")
    print(new_eval(statistics_test, 0.25, sta_theta, sta_beta, sta_k))
    print("--------------------------------------")

    print("Geometry:")
    geo_theta, geo_beta, geo_k, train_acc_lst, val_acc_lst, train_nlld, val_nlld = train_category(geometry_train, geometry_valid, 12, 0.0015, 0.25)
    print("test acc:")
    print(new_eval(geometry_test, 0.25, geo_theta, geo_beta, geo_k))
    print("--------------------------------------")

    print("Other:")
    other_theta, other_beta, other_k, train_acc_lst, val_acc_lst, train_nlld, val_nlld = train_category(other_train, other_valid, 20, 0.0015, 0.25)
    print("test acc:")
    print(new_eval(other_test, 0.25, other_theta, other_beta, other_k))
    print("--------------------------------------")

    print("Train all:")
    all_theta, all_beta, all_k, train_acc_lst, val_acc_lst, train_nlld, val_nlld = train_category(train_data, val_data, 20, 0.001, 0.25)
    print("test acc:")
    theta = {
        "number": num_theta,
        "algebra": alg_theta,
        "geometry": all_theta,
        "statistics": all_theta,
        "other": all_theta
    }
    beta = {
        "number": num_beta,
        "algebra": alg_beta,
        "geometry": all_beta,
        "statistics": all_beta,
        "other": all_beta
    }
    k = {
        "number": num_k,
        "algebra": alg_k,
        "geometry": all_k,
        "statistics": all_k,
        "other": all_k
    }

    print(test_cluster_model(test_data, 0.25, theta, beta, k))

if __name__ == "__main__":
    main()