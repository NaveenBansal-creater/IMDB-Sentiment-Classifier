import torch
from torchvision import datasets, transforms

## Dataset and Dataloader
class IMDBDataset:
    
    def __init__(self,sents,labels):
        self.sents = torch.tensor(sents)
        self.labels = torch.tensor(labels,dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        
        current_sample = self.sents[idx]
        current_target = self.labels[idx]
        return {
            "x": current_sample.reshape(-1),
            "y": current_target
        }   
    

def get_dataloader(df,feature_col,label_col,batch_size=50):
    
    dataset = IMDBDataset(df[feature_col].tolist(),df[label_col].tolist())
    
    data_loader  = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               shuffle = True,
                                               drop_last=True)
    
    return data_loader