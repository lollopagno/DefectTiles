class EarlyStopping:
    r"""
    Class that implements early stopping to end network training to avoid the overfitting.
    """

    def __init__(self, patience=5):
        r"""
        Constructor of class.
        :param patience: number of epochs to wait to finish the training.
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):

        if self.best_loss is None or val_loss > self.best_loss:
            # Update the best loss field
            self.best_loss = val_loss

        elif self.best_loss - val_loss < 0:

            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")

            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
