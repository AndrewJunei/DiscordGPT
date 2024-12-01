import torch
import numpy as np  
from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight
from GPT2.config_med import GPT2Config

class EncodedData:
    def __init__(self, batch_size, ctxlen, device):
        self.device = device
        full_data = np.load('encoded_data.npy')
        full_data = torch.from_numpy(full_data).to(self.device) # int64

        n = int(0.1*len(full_data)) 
        self.train_data = full_data[n:] # 90%  
        self.val_data = full_data[:n] # 10% 
        print('train size: ', self.train_data.shape, flush=True)
        print('val size: ', self.val_data.shape, flush=True)

        self.batch_size = batch_size
        self.ctxlen = ctxlen

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        # inputs x and targets y
        ix = torch.randint(len(data) - self.ctxlen, (self.batch_size,))
        x = torch.stack([data[i:i+self.ctxlen] for i in ix])
        y = torch.stack([data[i+1:i+self.ctxlen+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
    
class GPTTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        batch_size = 8 # see what fills memory 
        self.grad_accumulation_steps = 4
        ctxlen = 512 # GPT2 = 1024
        self.data = EncodedData(batch_size, ctxlen, self.device)
        
        self.num_epochs = 20
        self.lr = 3e-5 
        self.iter_per_epoch = 1100 # roughly
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
    @torch.no_grad()
    def estimate_losses(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.iter_per_epoch // 8) # because of val size
            for i in range(self.iter_per_epoch // 8): 
                x, y = self.data.get_batch(split)
                loss = self.model(x, lm_labels=y)
                losses[i] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def training_loop(self):
        train_losses, val_losses = [], []

        # epoch 0, untrained losses
        losses = self.estimate_losses()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        print(f"epoch 0: train loss {losses['train']:.5f}, val loss {losses['val']:.5f}", flush=True)

        for epoch in range(self.num_epochs):
            for step in range(self.iter_per_epoch):
                x, y = self.data.get_batch('train')
                loss = self.model(x, lm_labels=y)
                loss = loss / self.grad_accumulation_steps
                loss.backward()

                if (step + 1) % self.grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            losses = self.estimate_losses()
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            print(f"epoch {epoch+1}: train loss {losses['train']:.5f}, val loss {losses['val']:.5f}", flush=True)

            torch.save(self.model.state_dict(), f'gpt_checkpoint_{epoch+1}.pth')
            savelosses = np.array([train_losses, val_losses])
            np.save('gpt_losses.npy', savelosses)

seed = 1000
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush=True)

config = GPT2Config()
model = GPT2LMHeadModel(config)
state_dict = torch.load('pytorch_model_med.bin', map_location=device)
model = load_weight(model, state_dict)
model = model.to(device)
print('model loaded', flush=True)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters', flush=True)

trainer = GPTTrainer(model, device)
trainer.training_loop()