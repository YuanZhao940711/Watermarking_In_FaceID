import os
import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from utils.common import l2_norm, alignment, statistic_correlation, visualize_correlation
from utils.dataset import AnalysisDataset
from face_modules.model import Backbone



def main(args):
    global_step = 0

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

    dataset = AnalysisDataset(root=args.img_dir, perturbation=args.perturbation)
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
    
    corpeak_list = []
    coravg_list = []

    if args.visual_correlation == 'Yes':
        visual_dir = os.path.join(args.output_dir, 'visual_correlation')
        print("[*]Outputing visualized correlation results to {}".format(visual_dir))
        os.makedirs(visual_dir, exist_ok=True)

    for image_input in tqdm(dataloader):
        image_input = image_input.to(device)

        id_input = facenet(alignment(image_input))
        id_input_norm = l2_norm(id_input)

        peak_list, avg_list = statistic_correlation(id_input_norm, seq)
        
        corpeak_list.extend(peak_list)
        coravg_list.extend(avg_list)

        if args.visual_correlation == 'Yes':
            for i in range(len(image_input)):
                img_name = os.path.basename(dataset.image_paths[global_step])
                save_path = os.path.join(visual_dir, img_name)

                visualize_correlation(seq, id_input_norm[i], image_input[i], save_path)

                global_step += 1
    
    peak_mean = np.mean(corpeak_list)
    avg_mean = np.mean(coravg_list)

    print("[*]Average peak value of autocorrelation is {:.4f}".format(peak_mean))
    print("[*]Residual values' average of autocorrelation is {:.4f}".format(avg_mean))
    print("[*]Average Peak-to-Average Ratio is {:.4f}".format(peak_mean/avg_mean))



def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--perturbation', type=str, default='No')
    parser.add_argument('--visual_correlation', type=str, default='Yes')
    parser.add_argument('--seq_dir', type=str, default='./experiment/mls_weight_0.1')
    parser.add_argument('--img_dir', type=str, default='./experiment/mls_weight_0.1/injected')
    parser.add_argument('--output_dir', type=str, default='./experiment/mls_weight_0.1/verification')
    parser.add_argument('--facenet_mode', type=str, default='arcface')
    parser.add_argument('--facenet_dir', type=str, default='./experiment/mls_weight_0.1/best_models')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)