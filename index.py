from nn import NN

TRAINING_DATA = (
    ((0.5, 0.7), 0.2),
    ((0.35, 0.56), 0.9),
    # ((0.1, 0.1), 0.8),
    # ((0.2, 0.4), 0.4),
) * 10


def main():
    nn = NN([2, 4, 1])
    nn.train(TRAINING_DATA)


if __name__ == "__main__":
    main()
