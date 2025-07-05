import glob
import math
import os
import pandas as pd
import torch.nn.functional as F
from model.RNA_Model import Model
from Bio import SeqIO
import numpy as np
import torch
from parser import create_parser
from API.featurizer import featurize
from collections.abc import Mapping, Sequence

args = create_parser()
config = args.__dict__
seq_vocab = "AUCG"

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


class RNASequenceGenerator:
    def __init__(self, model_path):
        self.model = Model(args).to(args.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=args.device, weights_only=True)
        )
        self.model.eval()

    def generate_sequences(self, coord_data, num_seq=5, temperature=1.0, top_k=3):
        """
        生成候选RNA序列
        :param coord_data: numpy数组 [L, 7, 3]
        :param num_seq: 需要生成的序列数量
        :param temperature: 温度参数控制多样性
        :param top_k: 每个位置只考虑top_k高概率的碱基
        :return: 生成的序列列表
        """
        # 拿到数据，一个数据一个数据处理
        seq = self._preprocess_data(coord_data)
        X, S, mask, lengths = featurize([{
                'coords': coord_data,
                'seq': seq,
            }])
        X, S, mask, lengths = cuda((X, S, mask, lengths), device=args.device)
        # 获取概率分布
        with torch.no_grad():
            logits, S, graph_prjs = self.model(X, S, mask)
            probs = F.softmax(logits / temperature, dim=1)

        # 生成候选序列
        sequences = set()
        while len(sequences) < num_seq:
            seq = self._sample_sequence(probs, top_k)
            sequences.add(seq)

        return list(sequences)[:num_seq]
    def _preprocess_data(self, coord):
        """预处理坐标数据为图结构"""
        # 创建伪序列（实际不会被使用）
        dummy_seq = "A" * coord.shape[0]
        return dummy_seq

    def _sample_sequence(self, probs, top_k):
        """基于概率分布进行采样"""
        seq = []
        for node_probs in probs:
            # 应用top-k筛选
            topk_probs, topk_indices = torch.topk(node_probs, top_k)
            # 重新归一化
            norm_probs = topk_probs / topk_probs.sum()
            # 采样
            chosen = np.random.choice(topk_indices.cpu().numpy(), p=norm_probs.cpu().numpy())
            seq.append(seq_vocab[chosen])
        return "".join(seq)

if __name__ == "__main__":
    # 加载生成器
    generator = RNASequenceGenerator("./pth/best_model.pth")
    result={
     "pdb_id": [],
     "seq": []
    }
    # 推理
    npy_files=glob.glob("./saisdata/coords/*.npy")
    for npy in npy_files:
        id_name=os.path.basename(npy).split(".")[0]
        coord = np.load(npy)  # [L, 7, 3]
        coord = np.nan_to_num(coord, nan=0.0)

        # 生成1个候选序列
        candidates = generator.generate_sequences(
            coord,
            num_seq=1,
            temperature=0.8, # 适度多样性
            top_k=4 # 每个位置考虑前4个可能
        )
        result["pdb_id"].append(id_name)
        result["seq"].append(candidates[0])
    result=pd.DataFrame(result)
    result.to_csv("./saisresult/submit.csv", index=False)
