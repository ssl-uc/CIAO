import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from functools import lru_cache

@lru_cache(maxsize=16)
def load_binary(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


class PathIndexedContrastiveDataset(IterableDataset):
    def __init__(self, arch1_index_path, arch2_dir, func_name_map, max_per_label):
        with gzip.open(arch1_index_path, "rb") as f:
            self.label_index = pickle.load(f)

        self.arch2_dir = arch2_dir
        self.max_per_label = max_per_label
        self.func_cache = {}  # reuse loaded files

    def load_func(self, path, index):
        return load_binary(path)[index]

    def __iter__(self):
        for fname in os.listdir(self.arch2_dir):
            path = os.path.join(self.arch2_dir, fname)
            try:
                with gzip.open(path, "rb") as f:
                    funcs = pickle.load(f)
                    for func2 in funcs:
                        label = func_name_map[func2.get("label")]
                        if not label or label not in self.label_index:
                            continue
                        matches = self.label_index[label]
                        if not matches:
                            continue
                        samples = random.sample(matches, min(len(matches), self.max_per_label))
                        for match in samples:
                            func1 = self.load_func(match["path"], match["index"])
                            if not func1.get("tokens") or not func2.get("tokens"):
                                continue
                            yield (
                                torch.tensor(func1["tokens"], dtype=torch.float32),
                                torch.tensor(func2["tokens"], dtype=torch.float32)
                            )
            except Exception as e:
                print(f"Failed to stream {path}: {e}")
 
 def contrastive_collate_fn(batch):
    x1, x2 = zip(*batch)

    # Clamp token count before padding
    x1 = [t[:MAX_LEN] for t in x1]
    x2 = [t[:MAX_LEN] for t in x2]

    # Pad sequences
    x1 = pad_sequence(x1, batch_first=True) 
    x2 = pad_sequence(x2, batch_first=True)  
    # Final clamp to avoid model crash
    x1 = x1[:, :MAX_LEN, :]
    x2 = x2[:, :MAX_LEN, :]

    return x1, x2
    

# like SimCLR, custom contrastive loss with positive pairs to pull positive pairs closer    
def contrastive_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(len(z1)).to(z1.device)
    return F.cross_entropy(logits, labels)
    
    
class FunctionEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, max_len, num_layers, nhead):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, model_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True 
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        x = self.input_linear(x)                      
        x = x + self.pos_embed[:, :x.size(1)]       
        x = x.permute(1, 0, 2)                      
        x = self.encoder(x)                         
        x = x.permute(1, 2, 0)                      
        x = self.pool(x).squeeze(-1)                
        return self.out_proj(x)       
        
def train(input_dim, model_dim, MAX_LEN, num_layers, nhead, arch1_index_path, arch2_dir, func_name_map, max_per_label, max_step, device, epochs, lr, model_out):


    dataset = PathIndexedContrastiveDataset( arch1_index_path, arch2_dir, func_name_map, max_per_label)
    loader = DataLoader(dataset, batch_size=16, collate_fn=contrastive_collate_fn)

    model = FunctionEncoder(input_dim, model_dim, MAX_LEN, num_layers, nhead)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    

    scaler = GradScaler(device=device)
    for epoch in range(epochs):
        total_loss = 0
        step_count = 0
        for step, (x1, x2) in enumerate(dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)

            optimizer.zero_grad()


            with autocast(device):
                z1 = model(x1)
                z2 = model(x2)
                loss = contrastive_loss(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            total_loss += loss.item()
            step_count += 1
            if step % 10 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

            if step_count >= max_step:
                break

        if epoch % 5 == 0:
            #save
            torch.save(model.state_dict(), model_out+"contrastive_function_encoder_"+str(epoch)+"_.pt")


        
        avg_loss = total_loss / step_count if step_count > 0 else 0.0
        print(f"Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), model_out+"contrastive_function_encoder_final.pt")
           
