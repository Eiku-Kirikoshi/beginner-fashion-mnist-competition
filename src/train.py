
# uv run src/train.py

# import pickle
# from pathlib import Path

# from load_fashion_mnist import load_train_data
# from network import NetworkConfig, SimpleMLP

# OUTPUT_PATH = Path("sample_weight.pkl")

# # Dropoutの導入により収束が遅くなるためエポックを増加
# EPOCHS = 40
# HIDDEN_SIZE1 = 512
# HIDDEN_SIZE2 = 256
# LEARNING_RATE = 0.001
# BATCH_SIZE = 64
# SEED = 42

# def main() -> int:
#     (x_train, t_train), (x_valid, t_valid) = load_train_data()

#     model = SimpleMLP(
#         NetworkConfig(
#             input_size=x_train.shape[1],
#             hidden_size1=HIDDEN_SIZE1,
#             hidden_size2=HIDDEN_SIZE2,
#             output_size=10,
#             learning_rate=LEARNING_RATE,
#             batch_size=BATCH_SIZE,
#             seed=SEED,
#         )
#     )

#     for epoch in range(1, EPOCHS + 1):
#         # 【追加3】学習の終盤で学習率を10分の1に下げる (Learning Rate Decay)
#         if epoch == 30:
#             model.config.learning_rate *= 0.1
#             print(f"  -> Learning rate dynamically decayed to {model.config.learning_rate:.4f}")

#         loss = model.train_epoch(x_train, t_train, epoch=epoch)
#         train_acc = model.evaluate_accuracy(x_train, t_train)
#         valid_acc = model.evaluate_accuracy(x_valid, t_valid)
#         print(
#             f"Epoch {epoch:02d}/{EPOCHS} "
#             f"loss={loss:.4f} train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}"
#         )

#     with OUTPUT_PATH.open("wb") as f:
#         pickle.dump(model.to_state(), f)

#     print(f"Saved model: {OUTPUT_PATH.resolve()}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())

# uv run src/train.py

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

from load_fashion_mnist import load_train_data
from network import NetworkConfig, FashionCNN

OUTPUT_PATH = Path("sample_weight.pkl")
EPOCHS = 35
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42

def main() -> int:
    # 1. デバイスの自動判定 (M1/M2 MacならMPS、NVIDIAならCUDA、それ以外はCPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. データの準備
    (x_train, t_train), (x_valid, t_valid) = load_train_data()
    
    x_train_t = torch.from_numpy(x_train).float().view(-1, 1, 28, 28)
    t_train_t = torch.from_numpy(t_train).long()
    x_valid_t = torch.from_numpy(x_valid).float().view(-1, 1, 28, 28)
    t_valid_t = torch.from_numpy(t_valid).long()

    # 3. データ拡張 (Data Augmentation)
    # 左右反転と、わずかな回転を加えて擬似的にデータ量を増やす
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])

    train_dataset = TensorDataset(x_train_t, t_train_t)
    valid_dataset = TensorDataset(x_valid_t, t_valid_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. モデルと最適化アルゴリズムのセットアップ
    config = NetworkConfig(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED)
    model = FashionCNN(config).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 25エポック目で学習率を1/10にする
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.1)

    # 5. 学習ループ
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_t in train_loader:
            # データ拡張の適用
            batch_x = torch.stack([train_transform(img) for img in batch_x])
            batch_x, batch_t = batch_x.to(device), batch_t.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_t).sum().item()
            total_train += batch_t.size(0)
            
        scheduler.step()
        
        # 検証ループ
        model.eval()
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for batch_x, batch_t in valid_loader:
                batch_x, batch_t = batch_x.to(device), batch_t.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                correct_valid += (preds == batch_t).sum().item()
                total_valid += batch_t.size(0)
                
        epoch_loss = total_loss / total_train
        train_acc = correct_train / total_train
        valid_acc = correct_valid / total_valid
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{EPOCHS} [LR: {current_lr:.5f}] loss={epoch_loss:.4f} train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}")

    # 6. モデルの保存
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(model.to_state(), f)

    print(f"Saved model: {OUTPUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())