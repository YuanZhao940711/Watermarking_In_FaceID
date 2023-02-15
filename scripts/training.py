import os
import sys
import time
import json
import pprint

sys.path.append(".")
sys.path.append("..")

import numpy as np
from scipy.signal import max_len_seq 

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from options.options import TrainingOptions

from face_modules.model import Backbone
from network.AAD import AADGenerator
from network.MAE import MLAttrEncoder
from network.Discriminator import MultiscaleDiscriminator

from criteria import loss_functions

from utils.common import l2_norm, alignment, visualize_train_results, generate_seqstate
from utils.dataset import TrainingDataset



class Train:
    def __init__(self, opts):
        self.opts = opts

        torch.backends.deterministic = True
        SEED = self.opts.seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        self.opts.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        print("[*]Running on device: {}".format(self.opts.device))


        ### Initialize networks and load pretrained models ###
        self.aadblocks = AADGenerator(c_id=512).to(self.opts.device)
        try:
            self.aadblocks.load_state_dict(torch.load(self.opts.aadblocks_dir, map_location=self.opts.device), strict=True)
            print("[*]Successfully loaded AAD Generator's pre-trained model")
        except:
            print("[*]Training AAD Generator from scratch")

        self.attencoder = MLAttrEncoder().to(self.opts.device)
        try:
            self.attencoder.load_state_dict(torch.load(self.opts.attencoder_dir, map_location=self.opts.device), strict=True)
            print("[*]Successfully loaded Attributes Encoder's pre-trained model")
        except:
            print("[*]Training Attributes Encoder from scratch")

        self.discriminator = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).to(self.opts.device)
        try:
            self.discriminator.load_state_dict(torch.load(self.opts.discriminator_dir, map_location=self.opts.device), strict=True)
            print("[*]Successfully loaded Discriminator's pre-trained model")
        except:
            print("[*]Training Discriminator from scratch")

        print("[*]Loading Face Recognition Model {} from {}".format(self.opts.facenet_mode, self.opts.facenet_dir))
        if self.opts.facenet_mode == 'arcface':
            self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(self.opts.device)
            self.facenet.load_state_dict(torch.load(os.path.join(self.opts.facenet_dir, 'model_ir_se50.pth'), map_location=self.opts.device), strict=True)
        elif self.opts.facenet_mode == 'circularface':
            self.facenet = Backbone(input_size=112, num_layers=100, drop_ratio=0.4, mode='ir', affine=False).to(self.opts.device)
            self.facenet.load_state_dict(torch.load(os.path.join(self.opts.facenet_dir, 'CurricularFace_Backbone.pth'), map_location=self.opts.device), strict=True)
        else:
            raise ValueError("Invalid Face Recognition Model. Must be one of [arcface, CurricularFace]")


        ### Initialize optimizers ###
        self.opt_aad = optim.Adam(self.aadblocks.parameters(), lr=self.opts.lr_aad, betas=(0.9, 0.999))
        self.opt_att = optim.Adam(self.attencoder.parameters(), lr=self.opts.lr_att, betas=(0.9, 0.999))
        self.opt_dis = optim.Adam(self.discriminator.parameters(), lr=self.opts.lr_dis, betas=(0.9, 0.999))


        ### Initialize result directories and folders ###
        self.trainpics_dir = os.path.join(self.opts.output_dir, 'TrainPics')
        os.makedirs(self.trainpics_dir, exist_ok=True)
        self.valpics_dir = os.path.join(self.opts.output_dir, 'ValidationPics')
        os.makedirs(self.valpics_dir, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.opts.output_dir, 'CheckPoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.best_checkpoints_dir = os.path.join(self.opts.output_dir, 'BestResult')
        os.makedirs(self.best_checkpoints_dir, exist_ok=True)
        self.log_dir = os.path.join(self.opts.output_dir, 'Logs')
        os.makedirs(self.log_dir, exist_ok=True)


        ### Initialize loss functions ###
        self.adv_loss = loss_functions.GANLoss(adv_weight=self.opts.adv_weight).to(self.opts.device)
        self.att_loss = loss_functions.AttLoss(self.opts.att_weight).to(self.opts.device)
        self.id_loss = loss_functions.IdLoss(self.opts.id_weight, self.opts.idloss_mode).to(self.opts.device)
        self.rec_loss = loss_functions.RecLoss(self.opts.rec_weight, self.opts.recloss_mode, self.opts.device)


        ### Initialize data loaders ###
        train_dataset = TrainingDataset(root=self.opts.trainimg_dir)
        val_dataset = TrainingDataset(root=self.opts.valimg_dir)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=int(self.opts.num_workers),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=int(self.opts.num_workers),
            drop_last=True,
        )

        ### Initialize logger ###
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.best_loss = None


    def train_dis(self, img_org):
        self.facenet.eval()
        self.aadblocks.eval()
        self.attencoder.eval()

        self.discriminator.train()

        loss_dis = 0.0

        id_org = self.facenet(alignment(img_org))
        id_org_norm = l2_norm(id_org) 

        id_input = id_org_norm * self.opts.idvec_weight + self.seq_weighted

        att_org = self.attencoder(Xt=img_org)

        img_rec = self.aadblocks(inputs=(att_org, id_input))
        
        dis_real = self.discriminator(img_org)
        dis_fake = self.discriminator(img_rec)

        self.opt_dis.zero_grad()

        loss_dis = self.cal_dis_loss(dis_real, dis_fake)

        loss_dis.backward()
        self.opt_dis.step()

        return loss_dis


    def train_gen(self, img_org):
        self.facenet.eval()
        self.discriminator.eval()

        self.aadblocks.train()
        self.attencoder.train()

        vis_dict = {}
        loss_gen = 0.0
        
        id_org = self.facenet(alignment(img_org))
        id_org_norm = l2_norm(id_org)

        id_input = id_org_norm * self.opts.idvec_weight + self.seq_weighted

        att_org = self.attencoder(Xt=img_org)

        img_rec = self.aadblocks(inputs=(att_org, id_input))

        id_rec = self.facenet(alignment(img_rec))
        id_rec_norm = l2_norm(id_rec)

        att_rec = self.attencoder(Xt=img_rec)

        dis_rec = self.discriminator(img_rec)

        self.opt_att.zero_grad()
        self.opt_aad.zero_grad()

        loss_gen, loss_gen_dict = self.cal_gen_loss(dis_rec, img_org, img_rec, id_input, id_rec_norm, att_org, att_rec)

        loss_gen.backward()
        self.opt_att.step()
        self.opt_aad.step()

        vis_dict = {
            'img_org': img_org,
            'img_rec': img_rec,
            'id_org': id_org_norm,
            'id_input': id_input,
            'id_rec': id_rec_norm,
            'seq': self.seq,
            'seq_len': self.seq_len,
            'seq_weighted': self.seq_weighted
        }
        return vis_dict, loss_gen_dict


    def training(self, epoch, dataloader):
        for train_iter, img_org in enumerate(dataloader):
            random_state = generate_seqstate()
                
            seq = max_len_seq(nbits=9, state=random_state)[0]*2.0 - 1.0
            seq = np.insert(seq, -1, 0)
            self.seq = torch.from_numpy(seq).float().to(self.opts.device)
            self.seq_weighted = self.seq * self.opts.seq_weight
            self.seq_len = len(self.seq)

            img_org = img_org.to(self.opts.device)

            ### Training Discriminator ###
            loss_dis = self.train_dis(img_org)

            ### Training Generator ###
            vis_dict, loss_train_dict = self.train_gen(img_org)

            loss_train_dict['Total_DisLoss'] = float(loss_dis)

            if (train_iter+1) % self.opts.board_interval == 0:
                self.print_metrics(loss_train_dict, train_iter+1, epoch, prefix='train')
                self.log_metrics(loss_train_dict, self.train_steps+1, prefix='train')

            if (train_iter+1) % self.opts.image_interval == 0:
                fig = visualize_train_results(cor_dict=vis_dict, dis_num=self.opts.display_num)
                imgoutput_dir = os.path.join(self.trainpics_dir, 'epoch_{:05d}_iteration_{:05d}_steps_{:05d}.png'.format(epoch, train_iter+1, self.train_steps+1))
                fig.savefig(imgoutput_dir)
                plt.close(fig)

            self.train_steps += 1
            
            if train_iter == self.opts.max_train_iters-1:
                break


    def validation(self, epoch, dataloader):
        self.aadblocks.eval()
        self.attencoder.eval()
        self.facenet.eval()
        self.discriminator.eval()

        vis_dict = {}
        val_loss = 0.0

        for val_iter, img_val in enumerate(dataloader):
            img_val = img_val.to(self.opts.device)

            id_org = self.facenet(alignment(img_val))
            id_org_norm = l2_norm(id_org)

            id_input = id_org_norm * self.opts.idvec_weight + self.seq_weighted

            att_val = self.attencoder(Xt=img_val)

            img_rec = self.aadblocks(inputs=(att_val, id_input))

            id_rec = self.facenet(alignment(img_rec))
            id_rec_norm = l2_norm(id_rec)

            att_rec = self.attencoder(Xt=img_rec)

            dis_rec = self.discriminator(img_rec)

            loss_val_dict = self.cal_val_loss(dis_rec, img_val, img_rec, id_input, id_rec_norm, att_val, att_rec)
            
            self.print_metrics(loss_val_dict, val_iter+1, epoch, prefix='validate')
            self.log_metrics(loss_val_dict, self.validate_steps+1, prefix='validate')

            val_loss += loss_val_dict['Total_ValLoss']

            self.validate_steps += 1

            if val_iter == self.opts.max_val_iters-1:
                break

        vis_dict = {
            'img_org': img_val,
            'img_rec': img_rec,
            'id_org': id_org_norm,
            'id_input': id_input,
            'id_rec': id_rec_norm,
            'seq': self.seq,
            'seq_len': self.seq_len,
            'seq_weighted': self.seq_weighted
        }
        fig = visualize_train_results(cor_dict=vis_dict, dis_num=self.opts.display_num)
        imgoutput_dir = os.path.join(self.valpics_dir, 'epoch_{:05d}.png'.format(epoch))
        fig.savefig(imgoutput_dir)
        plt.close(fig)

        val_loss_avg = val_loss/self.opts.max_val_iters

        return vis_dict, val_loss_avg


    def running(self):
        print("Start Training...")

        self.train_steps = 0
        self.validate_steps = 0

        for epoch in range(self.opts.max_epoch):
            self.training(epoch, self.train_loader)
            
            with torch.no_grad():
                vis_dict, total_valloss = self.validation(epoch, self.val_loader)

            self.save_checkpoint(Att=self.attencoder, AAD=self.aadblocks, Dis=self.discriminator, epoch=epoch, is_best=False)

            if (self.best_loss is None) or (total_valloss < self.best_loss):
                self.best_loss = total_valloss
                self.save_checkpoint(Att=self.attencoder, AAD=self.aadblocks, Dis=self.discriminator, epoch=epoch, is_best=True)
                
                fig = visualize_train_results(cor_dict=vis_dict, dis_num=self.opts.display_num)
                bestoutput_dir = os.path.join(self.best_checkpoints_dir, 'best.png')
                fig.savefig(bestoutput_dir)
                plt.close(fig)
            
        print("Training Finish")


    def cal_dis_loss(self, dis_real, dis_fake):
        loss = 0.0

        loss_real = self.adv_loss(dis_real, target_is_real=True)
        loss_fake = self.adv_loss(dis_fake, target_is_real=False)

        loss = (loss_real + loss_fake)/2
        return loss

    
    def cal_gen_loss(self, dis_rec, img_input, img_output, id_input, id_output, att_input, att_output):
        loss = 0.0
        loss_dict = {}

        loss_adv = self.adv_loss(dis_rec, target_is_real=True)
        loss_dict['loss_adv'] = float(loss_adv)
        loss += loss_adv

        loss_att = self.att_loss(att_output, att_input)
        loss_dict['loss_att'] = float(loss_att)
        loss += loss_att

        loss_id = self.id_loss(id_output, id_input)
        loss_dict['loss_id'] = float(loss_id)
        loss += loss_id

        loss_rec = self.rec_loss(img_output, img_input)
        loss_dict['loss_rec'] = float(loss_rec)
        loss += loss_rec

        loss_dict['Total_GenLoss'] = float(loss)
        return loss, loss_dict


    def cal_val_loss(self, dis_rec, img_input, img_output, id_input, id_output, att_input, att_output):
        loss = 0.0
        loss_dict = {}

        loss_adv = self.adv_loss(dis_rec, target_is_real=True)
        loss_dict['val_advloss'] = float(loss_adv)
        loss += loss_adv

        loss_att = self.att_loss(att_output, att_input)
        loss_dict['val_attloss'] = float(loss_att)
        loss += loss_att

        loss_id = self.id_loss(id_output, id_input)
        loss_dict['val_idloss'] = float(loss_id)
        loss += loss_id

        loss_rec = self.rec_loss(img_output, img_input)
        loss_dict['val_recloss'] = float(loss_rec)
        loss += loss_rec

        loss_dict['Total_ValLoss'] = float(loss)
        return loss_dict        


    def print_metrics(self, loss_dict, iteration, epoch, prefix):
        if prefix == 'train':
            print('Metrics for train, iteration {:05d}, epoch {:04d}'.format(iteration, epoch))
            for key, value in loss_dict.items():
                print('\t{}: {:.6f}'.format(key, value))
        elif prefix == 'validate':
            print('Validate, iteration {:05d}, epoch {:04d} are'.format(iteration, epoch), ['{}: {:.6f}'.format(key, value) for key, value in loss_dict.items()])
        else:
            raise ValueError('Unexpected prefix mode {}'.format(prefix))


    def save_checkpoint(self, Att, AAD, Dis, epoch, is_best):
        if is_best:
            torch.save(AAD.state_dict(), os.path.join(self.best_checkpoints_dir, 'AAD_best.pth'))
            torch.save(Att.state_dict(), os.path.join(self.best_checkpoints_dir, 'Att_best.pth'))
            torch.save(Dis.state_dict(), os.path.join(self.best_checkpoints_dir, 'Dis_best.pth'))
        else:
            torch.save(AAD.state_dict(), os.path.join(self.checkpoints_dir, 'AAD_{:05d}.pth'.format(epoch)))
            torch.save(Att.state_dict(), os.path.join(self.checkpoints_dir, 'Att_{:05d}.pth'.format(epoch)))
            torch.save(Dis.state_dict(), os.path.join(self.checkpoints_dir, 'Dis_{:05d}.pth'.format(epoch)))


    def log_metrics(self, loss_dict, iteration, prefix):
        for key, value in loss_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, (iteration))



def main():
    opts = TrainingOptions().parse()

    cur_time = time.strftime('%Y%m%d_H%H%M%S', time.localtime())
    output_dir = os.path.join(opts.exp_dir, '{}_sw({})_im({})_rm({})_{}'.format(cur_time, opts.seq_weight, opts.idloss_mode, opts.recloss_mode, opts.facenet_mode))

    os.makedirs(output_dir, exist_ok=True)
    print("[*]Exporting experiment results at {}".format(output_dir))
    
    opts.output_dir = output_dir

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(output_dir, 'train_opts.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    train = Train(opts)
    train.running()



if __name__ == '__main__':
	main()