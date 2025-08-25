import torch
from torch_geometric.loader import DataLoader
import model as md
from sklearn.metrics import precision_score, recall_score
import torch.nn.functional as F
from torch_geometric.data import Dataset

class FCGDatasetImproved(Dataset):
    def __init__(self, root_dir, labels, transform=None, pre_transform=None):
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.files = []

        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for fname in os.listdir(class_folder):
                self.files.append({
                    "path": os.path.join(class_folder, fname),
                    "label": labels[class_name],
                    "name": fname
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        data = torch.load(item["path"], weights_only=False)

        data.name = item["name"]
        return data

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == data.y).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, precision, recall


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            total_loss += loss.item()

            preds = out.argmax(dim=1)
            correct += (preds == data.y).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    # Calculate precision and recall (macro for multi-class)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy, precision, recall


def run_training(dataset, train_ds, val_ds,  test_ds, device_config, out_folder, hidden_dim=64, batch_size=32, epochs=200, lr=0.001):
    

    device = torch.device(device_config)
    num_classes = cls_number
    optimizer_config = torch.optim.Adam
    
    # Get input feature dimension from first graph in the dataset
    first_graph = torch.load(dataset.files[0]["path"], weights_only=False)  # add weidhts_only FALSE if trusted data source
    input_dim = first_graph.x.size(1)


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=1)


    model = md.GIN(in_channels=input_dim, hidden_channels=hidden_dim, num_classes=num_classes).to(device)
    optimizer = optimizer_config(model.parameters(), lr=lr)



    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_precision, train_recall = train(model, train_loader, optimizer, device)
        val_loss, val_acc, val_precision, val_recall = evaluate(model, val_loader, device)

        

        print(f"Epoch {epoch:03d}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f} |"
              )

        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        if epoch % 10 == 0:
            checkpoint_path = f"{out_folder}/model_epoch_{epoch}.pt"
            
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Saved model to {checkpoint_path}")

    # Final test accuracy
    test_loss, test_acc, test_precision, test_recall = evaluate2(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"\nFinal Precision: {test_precision:.4f}")
    print(f"\nFinal Recall: {test_recall:.4f}")
    return model
    
def prepare_dataset(data_dir, train_arm_label, train_mips_label, val_arm_label, val_mips_label, test_arm_label, test_mips_label):
    dataset = FCGDatasetImproved(data_dir)
    train_ds = [data for data in datset_transformer_epoch5 if data.name in train_arm_label + train_mips_label]
    test_ds = [data for data in datset_transformer_epoch5 if data.name in test_arm_label]
    val_ds = [data for data in datset_transformer_epoch5 if data.name in val_arm_label]
    
    return dataset, train_ds, val_ds, test_ds
