from utils import *
from torch.autograd import Variable

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        h = torch.nn.functional.sigmoid(self.g(inputs))
        out = torch.nn.functional.sigmoid(self.h(h))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.
    # On line 123-125 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    
    train_acc_list = []
    valid_acc_list = []
    train_loss_list = []
    valid_loss_list = []
    train_dict = load_train_csv('../data')

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            reg_loss = model.get_weight_norm() * lamb / 2.
            loss = torch.sum((output - target) ** 2.) + reg_loss
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        if epoch % 5 == 0:
            valid_acc, valid_loss = evaluate(model, zero_train_data, valid_data, lamb)
            #train_acc = evaluate_train(model, train_dict, zero_train_data)
            train_acc = 0
            train_acc_list.append((epoch, train_acc))
            valid_acc_list.append((epoch, valid_acc))
            train_loss_list.append((epoch, train_loss))
            valid_loss_list.append((epoch, valid_loss))
            print("Epoch: {} \t"
              "Training Accuracy: {:.6f}\t "
              "Training Cost: {:.6f}\t "
              "Validation Accuracy: {:.6f}\t "
              "Validation Cost: {:.6f}\t ".format(epoch, train_acc, train_loss, valid_acc, valid_loss))

    display_plot(valid_acc_list, "Validation Accuracy")
    display_plot(valid_loss_list, "Validation Loss")
    display_plot(train_acc_list, "Training Accuracy")
    display_plot(train_loss_list, "Training Loss")
    
def evaluate_train(model, train_dict, train_data):
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(train_dict["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][train_dict["question_id"][i]].item() >= 0.5
        if guess == train_dict["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def display_plot(data_list, label):
    """ Displays curve.
    :param valid: Validation accuracy
    :param y_label: Y-axis label of the plot
    :param number: The number of the plot
    :return: None
    """
    plt.clf()
    data = np.array(data_list)
    plt.plot(data[:, 0], data[:, 1], "g")
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data, lamb):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: A tuple (acc, loss)
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    loss = 0.

    reg_loss = model.get_weight_norm() * lamb / 2.

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        target = valid_data["is_correct"][i]
        if guess == target:
            correct += 1
        
        loss_tensor = torch.sum((output - target) ** 2.) + reg_loss
        loss += loss_tensor.item()

        total += 1
    return (correct / float(total), loss)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data("../data")

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = 10
    D = train_matrix.shape[1]
    model = AutoEncoder(D, k)

    # Set optimization hyperparameters.

    lr = 0.007
    num_epoch = 250 # 150 is sufficient
    lamb = 0.0

    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

