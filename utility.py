from torch.utils.data import Dataset, DataLoader

class IndexedDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx  # return index too

def tensor_to_str(t):
    return ", ".join([f"{x:.6f}" for x in t.detach().cpu().numpy().flatten()])