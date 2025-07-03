import math
import os
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from Bio import SeqIO
from collections.abc import Mapping, Sequence

from model.RNA_Model import Model
from API.dataloader_gtrans import DataLoader_GTrans
from parser import create_parser
from API.featurizer import featurize
def cuda(obj, *args, **kwargs):
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, *args, **kwargs)
    raise TypeError("Can't transfer object type `%s`" % type(obj))

class CoordTransform:
    @staticmethod
    def random_rotation(coords):
        device = torch.device('cuda:0')
        coords_tensor = torch.from_numpy(coords).float().to(device)
        angle = np.random.uniform(0, 2 * math.pi)
        # 绕z轴随机旋转骨架原子三维坐标
        rot_mat = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ], device=device)
        return (coords_tensor @ rot_mat.T).cpu().numpy()

class RNADataset(torch.utils.data.Dataset):
    def __init__(self, coords_dir, seqs_dir, augment=False):
        self.samples = []
        self.augment = augment
        for fname in os.listdir(coords_dir):
            # 加载坐标
            coord = np.load(os.path.join(coords_dir, fname))
            coord = np.nan_to_num(coord, nan=0.0)

            # 数据增强
            if self.augment and np.random.rand() > 0.5:
                coord = CoordTransform.random_rotation(coord)

            # 加载序列
            seq_id = os.path.splitext(fname)[0]
            seq_path = os.path.join(seqs_dir, f"{seq_id}.fasta")
            seq = str(next(SeqIO.parse(seq_path, "fasta")).seq)

            # 列表，里面是字典
            self.samples.append({
                'coords': coord,
                'seq': seq,
            })
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch in tqdm(loader, desc="Training Batch-ing"):
        optimizer.zero_grad()
        X, S, mask, lengths = batch
        X, S, mask, lengths = cuda((X, S, mask, lengths), device=device)
        logits, S, graph_prjs = model(X, S, mask)
        loss = criterion(logits, S)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()

    with torch.no_grad():
        total_correct = total_nodes = 0
        for batch in tqdm(loader, desc="Evaluate-ing"):
            X, S, mask, lengths = batch
            X, S, mask, lengths = cuda((X, S, mask, lengths), device=device)

            logits, S, graph_prjs = model(X, S, mask)
            log_probs = F.softmax(logits, dim=-1)
            preds = log_probs.argmax(dim=-1)
            total_correct += (preds == S).sum().item()
            total_nodes += S.size(0)
    return total_correct / total_nodes

if __name__ == '__main__':
    args = create_parser()
    config = args.__dict__

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    coords_dir = "./RNAdesignv2/train/coords"
    seqs_dir = "./RNAdesignv2/train/seqs"

    train_set = RNADataset(
        coords_dir,
        seqs_dir,
        augment=True
    )

    # 划分数据集
    train_size = int(0.8 * len(train_set))
    val_size = (len(train_set) - train_size) // 2
    test_size = len(train_set) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(
        train_set, [train_size, val_size, test_size])

    train_loader = DataLoader_GTrans(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                collate_fn=featurize)
    # 跑验证集卡住是因为batch_size大了，显存不足
    val_loader = DataLoader_GTrans(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=featurize)
    test_loader = DataLoader_GTrans(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=featurize)

    # 模型
    model = Model(args).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # log记录
    log_path = './logs/062801_log.csv'
    if not Path(log_path).exists():
        pd.DataFrame(columns=['epoch', 'train_loss', 'val_acc']).to_csv(log_path, index=False)

    # 训练
    best_acc = 0
    for epoch in tqdm(range(args.epochs), desc='Epoch-ing'):
        train_loss = train(model, train_loader, optimizer,  args.device)
        val_acc = evaluate(model, val_loader, args.device)

        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        row = pd.DataFrame([[epoch + 1, train_loss, val_acc]],
                           columns=['epoch', 'train_loss', 'val_acc'])
        row.to_csv(log_path, mode='a', header=False, index=False)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./pth/best_model.pth")

    # 最后的验证
    model.load_state_dict(torch.load("./pth/best_model.pth"))
    test_acc = evaluate(model, test_loader, args.device)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
