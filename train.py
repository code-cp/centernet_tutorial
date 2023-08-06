import torch 
from dataset import ToyDataset 
import os 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(dataloader, epoch): 
    
    for batch_idx, (img_batch, mask_batch) in enumerate(dataloader): 
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        
if __name__ == "__main__": 
    img_size = 128 
    batch_size = 8 
    epoch_num = 20 
    
    dataset = ToyDataset(img_shape=(img_size, img_size))
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)
    
    output_folder = './outputs'
    os.makedirs(output_folder, exist_ok=True)
    for epoch in range(epoch_num): 
        train_model(dataloader, epoch)