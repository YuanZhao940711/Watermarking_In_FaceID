import os
import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from utils.common import l2_norm, alignment, calculatie_correlation, evaluation
from utils.dataset import EvaluationDataset
from face_modules.model import Backbone, l2_norm



def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print("[*]Running on device: {}".format(device))

    print("[*]Loading Face Recognition Model {} from {}".format(args.facenet_mode, args.facenet_dir))
    if args.facenet_mode == 'arcface':
        facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(device)
        facenet.load_state_dict(torch.load(os.path.join(args.facenet_dir, 'model_ir_se50.pth'), map_location=device), strict=True)
    elif args.facenet_mode == 'circularface':
        facenet = Backbone(input_size=112, num_layers=100, drop_ratio=0.4, mode='ir', affine=False).to(device)
        facenet.load_state_dict(torch.load(os.path.join(args.facenet_dir, 'CurricularFace_Backbone.pth'), map_location=device), strict=True)
    else:
        raise ValueError("Invalid Face Recognition Model. Must be one of [arcface, CurricularFace]")    
    facenet.eval()

    dataset = EvaluationDataset(pos_root=args.imgpos_dir, neg_root=args.imgneg_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False
    )

    print("[*]Loading injected sequence from {}".format(args.seq_dir))
    seq_path = os.path.join(args.seq_dir, 'sequence.txt')
    seq = np.loadtxt(seq_path, delimiter=',')
    
    label_all = []
    pred_all = []

    for batch in tqdm(dataloader):
        img_input, label_input = batch
        img_input = img_input.to(device)

        id_input = facenet(alignment(img_input))
        id_input_norm = l2_norm(id_input)

        pred_input = calculatie_correlation(id_input_norm, seq, args.peak_threshold)
        
        label_all.extend(label_input.numpy())
        pred_all.extend(pred_input)

    accuracy, precision, recall, f1_score, tn, fp, fn, tp = evaluation(label_all, pred_all)

    print("[*]Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1_Score: {:.4f}".format(accuracy, precision, recall, f1_score))
    print("[*]True Negative: {}, False Positive: {}, False Negative: {}, True Positive: {}".format(tn, fp, fn, tp))
    
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    print("[*]True Positive Rate: {:.4f}, False Positive Rate: {:.4f}".format(tpr, fpr))



def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--peak_threshold', type=int, default=5)
    parser.add_argument('--seq_dir', type=str, default='./experiment/mls_weight_0.1')
    parser.add_argument('--facenet_mode', type=str, default='arcface')
    parser.add_argument('--facenet_dir', type=str, default='./experiment/mls_weight_0.1/best_models')
    parser.add_argument('--imgpos_dir', type=str, default='./experiment/mls_weight_0.1/real')
    parser.add_argument('--imgneg_dir', type=str, default='./experiment/mls_weight_0.1/fake')
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)