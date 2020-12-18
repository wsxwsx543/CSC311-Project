# TODO: complete this file.
import sys
sys.path.append("../")

from utils import *
from part_a.item_response import * 

def single_round_prediction_IRT(data, val_data, test_data, base_size, lr = 0.003, iterations = 50):
    length = len(data['user_id'])
    theta = np.array([0 for i in range(542)])
    beta = np.array([0 for j in range(1774)])
    indexs = np.random.randint(length, size=(base_size))
    
    new_data = {}
    new_data['user_id'] = list(np.array(data['user_id'])[indexs])
    new_data['question_id'] = list(np.array(data['question_id'])[indexs])
    new_data['is_correct'] = list(np.array(data['is_correct'])[indexs])
    for i in range(iterations):
        theta, beta = update_theta_beta(new_data, lr, theta, beta)
    
    val_pred = []
    for i, q in enumerate(val_data["question_id"]):
        u = val_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        val_pred.append(int(p_a >= 0.5))
        
    test_pred = []
    for i, q in enumerate(test_data["question_id"]):
        u = test_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        test_pred.append(int(p_a >= 0.5))
    
    return np.array(val_pred), np.array(test_pred)
        
    
def bagging_IRT_avg_pred(data, val_data, test_data, lr = 0.003, iterations = 50):
    length = len(data['user_id'])
    val_length = len(val_data['user_id'])
    test_length = len(test_data['user_id'])
    
    val_overall_pred = np.zeros((val_length))
    test_overall_pred = np.zeros((test_length))
    base_size = length
    
    for i in range(3):
        val_single_round_pred, test_single_round_pred = single_round_prediction_IRT(data, val_data, test_data, base_size, lr, iterations)
        val_overall_pred += val_single_round_pred
        test_overall_pred += test_single_round_pred
    
    return 1*(val_overall_pred/3 >= 0.5), 1*(test_overall_pred/3 >= 0.5)
 
    
def bagging_IRT_evaluate(data, val_data, test_data, lr = 0.003, iterations = 50):
    val_avg_pred, test_avg_pred = bagging_IRT_avg_pred(data, val_data, test_data, lr, iterations)
    
    val_accu = np.sum((val_data["is_correct"] == np.array(val_avg_pred))) / len(val_data["is_correct"])
    test_accu = np.sum((test_data["is_correct"] == np.array(test_avg_pred))) / len(test_data["is_correct"])
    
    return val_accu, test_accu


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    LEARNING_RATE = 0.003
    NUM_ITERATIONS = 30
    
    val_accu, test_accu = bagging_IRT_evaluate(train_data, val_data, test_data, LEARNING_RATE, NUM_ITERATIONS)
    print("Validation Accuracy is: {} \n Test Accuracy is: {}".format(val_accu, test_accu))

if __name__ == "__main__":
    main()
    