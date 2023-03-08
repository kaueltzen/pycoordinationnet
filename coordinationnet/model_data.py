import torch

## ----------------------------------------------------------------------------

class CoordinationFeaturesData(torch.utils.data.Dataset):

    def __init__(self, X, y = None) -> None:

        if y is None:
            self.y = len(X)*[None]
        else:
            if isinstance(y, torch.Tensor):
                self.y = y
            else:
                self.y = torch.tensor(y)

        self.X = X

        for i, features in enumerate(X):
            if not features.encoded:
                self.X[i] = features.encode()

    def __len__(self):
        return len(self.X)

    # Called by pytorch DataLoader to collect items that
    # are joined later by collate_fn into a batch
    def __getitem__(self, index):
        return self.X[index], self.y[index]
