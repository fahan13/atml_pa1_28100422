import torch.nn as nn
import torch
from common.seed import set_seed
import copy

def train_linear_head(train_feats, train_labels, val_feats, val_labels,
                       num_classes=10, seed=6304, max_epochs=50, patience=5, device='cuda'):
    set_seed(seed)
    
    d = train_feats.shape[1]  # feature dimension: 2048 for ResNet, 768 for ViT, 512 for CLIP
    head = nn.Linear(d, num_classes).to(device)  
    
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()  
    
    train_feats_t = torch.tensor(train_feats, dtype=torch.float32).to(device)
    train_labels_t = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_feats_t = torch.tensor(val_feats, dtype=torch.float32).to(device)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long).to(device)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_state = None
    
    for epoch in range(max_epochs):
        head.train()
        optimizer.zero_grad()               
        logits = head(train_feats_t)         
        loss = loss_fn(logits, train_labels_t)
        loss.backward()                      
        optimizer.step()                   
        
        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats_t)
            val_preds = val_logits.argmax(dim=1)   
            val_acc = (val_preds == val_labels_t).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            break   
    
    head.load_state_dict(best_state)  
    return head