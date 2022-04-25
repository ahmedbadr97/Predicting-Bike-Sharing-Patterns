import torch

from torch.optim import Adam
from torch.nn import MSELoss
from src.model import BikeSharingModel
from torch import no_grad

from datetime import datetime


def test_model(model: BikeSharingModel, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_size = len(test_loader)
    for data, actual_output in test_loader:
        with no_grad():
            output = model.forward(data)
            loss = criterion(output, actual_output)
            test_loss += loss.item()
    model.train()
    return test_loss / test_size


def train_model(model: BikeSharingModel, epochs, learning_rate, train_loader, test_loader):
    model_saving_path = "../model_weights/training"
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    train_losses, test_losses = [], []
    train_size = len(train_loader)
    min_train_loss = min_test_loss = 10000.0

    print(f"model test_loss before training ={str(test_model(model, test_loader, criterion))[:8]}")
    for e in range(epochs):
        train_loss = 0
        for data, actual_output in train_loader:
            optimizer.zero_grad()
            output = model.forward(data)

            loss = criterion(output, actual_output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= train_size
        train_losses.append(train_loss)

        test_loss = test_model(model, test_loader, criterion)
        test_losses.append(test_loss)

        print(f"epoch:{e + 1}/{epochs} Train Loss={train_loss} Test Loss={test_loss}")

        if train_loss < min_train_loss and test_loss < min_test_loss:
            weight_file_name = f"{datetime.now().strftime('%m_%d %H_%M')} Train_({str(train_loss)[:8]} Test_({str(test_loss)[:8]}).pt"
            full_path = f"{model_saving_path}/{weight_file_name}"

            torch.save(model.state_dict(), full_path)
            print(
                f"new minimum test loss {str(train_loss)[:8]} and train loss {str(test_loss)[:8]} achieved model weights will be saved in\n ")

        if train_loss < test_loss:
            print("!!!Warning Overfitting!!!")
        print()
