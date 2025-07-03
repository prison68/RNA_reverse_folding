import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # 数据集参数
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    # 训练参数
    parser.add_argument('--epochs', default=200, type=int, help='end epoch')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # 边、节点特征
    parser.add_argument('--node_feat_types', default=['node_d', 'angle', 'distance', 'direction'], type=list)
    parser.add_argument('--edge_feat_types', default=['orientation', 'distance', 'direction'], type=list)

    # 模型参数
    parser.add_argument('--num_encoder_layers', default=4, type=int)
    parser.add_argument('--num_decoder_layers', default=3, type=int)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--k_neighbors', default=30, type=int)
    parser.add_argument('--vocab_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    
    return parser.parse_args()