import os
import sys
import json
import pprint

sys.path.append(".")
sys.path.append("..")

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.signal import max_len_seq

import torch
from torch.utils.data import DataLoader

from options.options import InjectionOptions

from network.AAD import AADGenerator
from network.MAE import MLAttrEncoder

from face_modules.model import Backbone

from utils.common import l2_norm, alignment, tensor2img, generate_seqstate
from utils.dataset import InjectionDataset



class Inject:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        torch.backends.deterministic = True
        SEED = self.opts.seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        self.opts.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        print("[*]Running on device: {}".format(self.opts.device))


        ### Initialize networks and load pretrained models ###
        self.aadblocks = AADGenerator(c_id=512).to(self.opts.device)
        print("[*]Loading AAD Blocks pre-trained model from {}".format(self.opts.aadblocks_dir))
        self.aadblocks.load_state_dict(torch.load(self.opts.aadblocks_dir, map_location=self.opts.device), strict=True)
        self.aadblocks.eval()

        self.attencoder = MLAttrEncoder().to(self.opts.device)
        print("[*]Loading Attributes Encoder pre-trained model from {}".format(self.opts.attencoder_dir))
        self.attencoder.load_state_dict(torch.load(self.opts.attencoder_dir, map_location=self.opts.device), strict=True)
        self.attencoder.eval()

        print("[*]Loading Face Recognition Model {} from {}".format(self.opts.facenet_mode, self.opts.facenet_dir))
        if self.opts.facenet_mode == 'arcface':
            self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(self.opts.device)
            self.facenet.load_state_dict(torch.load(os.path.join(self.opts.facenet_dir, 'model_ir_se50.pth'), map_location=self.opts.device), strict=True)
        elif self.opts.facenet_mode == 'circularface':
            self.facenet = Backbone(input_size=112, num_layers=100, drop_ratio=0.4, mode='ir', affine=False).to(self.opts.device)
            self.facenet.load_state_dict(torch.load(os.path.join(self.opts.facenet_dir, 'CurricularFace_Backbone.pth'), map_location=self.opts.device), strict=True)
        else:
            raise ValueError("Invalid Face Recognition Model. Must be one of [arcface, CurricularFace]")
        self.facenet.eval()


        ### Generate and save watermark sequence ###
        if self.opts.seq_type == 'mls':
            state = generate_seqstate()
            mls = max_len_seq(nbits=9, state=state)[0]*2.0 - 1.0
            seq = np.insert(mls, -1, 0)
        elif self.opts.seq_type == 'gold':
            state_01 = generate_seqstate()
            state_02 = generate_seqstate()
            mls_01 = max_len_seq(nbits=9, state=state_01)[0]
            mls_02 = max_len_seq(nbits=9, state=state_02)[0]
            gcs = (np.logical_xor(mls_01, mls_02) * 1) * 2.0 - 1.0 
            seq = np.insert(gcs, -1, 0)
        elif self.opts.seq_type == 'gaussian':
            gaussian = np.random.normal(loc=0.0, scale=1.0, size=512)
            seq = gaussian
        elif self.opts.seq_type == 'laplace':
            laplace = np.random.laplace(loc=0, scale=1.0, size=512)
            seq = laplace
        else:
            raise ValueError('Unexpected Generator training mode {}'.format(self.opts.genloss_mode))

        self.seq_path = os.path.join(self.opts.output_dir, 'sequence.txt')
        print("[*]Outputing {} sequence to {}".format(self.opts.seq_type, self.seq_path))
        np.savetxt(self.seq_path, seq, fmt='%f', delimiter=',')

        self.seq = torch.from_numpy(seq).float().to(self.opts.device)

        self.dataset = InjectionDataset(root=self.opts.img_dir, max_num=self.opts.max_num, rand_select=self.opts.rand_select, rand_seed=self.opts.seed)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=int(self.opts.num_workers),
            drop_last=False,
        )



    def running(self):
        imgin_dir = os.path.join(self.opts.output_dir, 'original')
        imgout_dir = os.path.join(self.opts.output_dir, 'reconstructed')
        os.makedirs(imgin_dir, exist_ok=True)
        os.makedirs(imgout_dir, exist_ok=True)

        self.seq_weighted = self.seq * self.opts.seq_weight

        for input_batch in tqdm(self.dataloader):
            with torch.no_grad():
                img_org = input_batch
                img_org = img_org.to(self.opts.device)
                
                id_org = self.facenet(alignment(img_org))
                id_org_norm = l2_norm(id_org)

                id_input = id_org_norm * self.opts.idvec_weight + self.seq_weighted
                
                att_org = self.attencoder(Xt=img_org)
                
                img_rec = self.aadblocks(inputs=(att_org, id_input))

            for i in range(len(input_batch)):
                img_input = tensor2img(img_org[i])
                img_output = tensor2img(img_rec[i])

                img_name = self.dataset.image_paths[self.global_step]

                Image.fromarray(np.array(img_input)).save(os.path.join(imgin_dir, os.path.basename(img_name)))
                Image.fromarray(np.array(img_output)).save(os.path.join(imgout_dir, os.path.basename(img_name)))

                self.global_step += 1



def main():
    opts = InjectionOptions().parse()

    output_dir = os.path.join(opts.exp_dir, 'sw({})_st({})_{}'.format(opts.seq_weight, opts.seq_type, opts.facenet_mode))
    
    os.makedirs(output_dir, exist_ok=True)
    print("[*]Generating injection results at {}".format(output_dir))
    
    opts.output_dir = output_dir
    
    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(output_dir, 'inject_opts.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    inject = Inject(opts)
    inject.running()



if __name__ == '__main__':
    main()