from torch.utils.data import Subset, Dataset

class ClientSet(Dataset):        
    def __init__(self,clients):
        super().__init__()
        self.clients = clients;
        
    def __getitem__(self, index):
        return self.clients[index];
    
    def __len__(self):
        return len(self.clients);
    
def client_collate(batch):    
    return batch;