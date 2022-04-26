import matplotlib.pyplot as plt


def plot_train_validation(*curves: tuple):
    for data, label in curves:
        plt.plot(data, label=label)
    plt.legend(frameon=False)
