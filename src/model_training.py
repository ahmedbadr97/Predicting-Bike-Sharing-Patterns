import sys

import torch

from torch.optim import Adam
from torch.nn import MSELoss
from src.model import BikeSharingModel
from torch import no_grad
from datetime import datetime
from time import time


def test_model(model: BikeSharingModel, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_size = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    for data, actual_output in test_loader:
        with no_grad():
            output = model.forward(data)
            loss = criterion(output, actual_output)
            test_loss += loss.item() * batch_size
    model.train()
    return test_loss / test_size


def train_model(model: BikeSharingModel, train_loader, test_loader, learning_rate=0.01, epochs=None,
                time_to_train=None):
    """

    :param model:
    :param train_loader:
    :param test_loader:
    :param learning_rate:
    :param epochs: no of epochs
    :param time_to_train: train the model for specific time not epochs in minutes
    :return: (train_losses list,test_losses list)
    """

    if epochs is None and time_to_train is None:
        return

    model_saving_path = "../model_weights/training"
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    train_losses, test_losses = [], []
    train_size = len(train_loader.dataset)
    min_train_loss = min_test_loss = 5000
    batch_size = train_loader.batch_size

    t_start = time()
    train_loss = test_model(model, train_loader, criterion)
    test_loss = test_model(model, test_loader, criterion)
    t_end = time()
    avg_time_epoch = t_end - t_start

    print(f"avg time for epoch= {int(avg_time_epoch / 60)}m and {round(avg_time_epoch % 60,2)}s")
    if epochs is None:
        epochs = int((time_to_train * 60) / avg_time_epoch)

    print(f"model before training Train loss={str(train_loss)[:8]} Test Loss={str(test_loss)[:8]}")
    for e in range(epochs):
        train_loss = 0
        t_start = time()
        for data, actual_output in train_loader:
            optimizer.zero_grad()
            output = model.forward(data)

            loss = criterion(output, actual_output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size

        train_loss /= train_size
        train_losses.append(train_loss)

        test_loss = test_model(model, test_loader, criterion)
        test_losses.append(test_loss)

        t_end = time()

        sys.stdout.write(
            f"\repoch:{e + 1}/{epochs} time per epoch {round((t_end - t_start), 2)} Train Loss={train_loss} Test Loss={test_loss}")
        sys.stdout.flush()
        # print(f"epoch:{e + 1}/{epochs} time per epoch {round((t_end - t_start),2)} Train Loss={train_loss} Test Loss={test_loss}")
        if train_loss < min_train_loss and test_loss < min_test_loss:
            full_path = save_train_weights(model, train_loss, test_loss, model_saving_path)
            print(
                f"new minimum test loss {str(train_loss)[:8]} and train loss {str(test_loss)[:8]} achieved model weights will be saved in\n{full_path}")
            min_train_loss=train_loss
            min_test_loss=test_loss

        if train_loss < test_loss:
            print("!!!Warning Overfitting!!!")
        print()
    return train_losses, test_losses


def save_train_weights(model, train_loss, test_loss, saving_path):
    """
    saves model weights with file name format Day_Month Hour_minute train_(train_loss) test_(test_loss)
    :param model: model object
    :param train_loss: train loss (float)
    :param test_loss: test loss (float)
    :param saving_path: the path you want to save the weights in
    :return: the full path of the saved file (saving_path+filename)
    """
    weight_file_name = f"{datetime.now().strftime('%m_%d %H_%M')} Train_({str(train_loss)[:8]} Test_({str(test_loss)[:8]}).pt"
    full_path = f"{saving_path}/{weight_file_name}"

    torch.save(model.state_dict(), full_path)
    return full_path
