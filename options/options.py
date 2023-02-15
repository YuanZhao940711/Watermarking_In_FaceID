from argparse import ArgumentParser



class TrainingOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self): 
        self.parser.add_argument('--seed', default=0, type=int)

        self.parser.add_argument('--max_epoch', default=15, type=int)
        self.parser.add_argument('--display_num', default=8, type=int)
        self.parser.add_argument('--batch_size', default=8, type=int)
        self.parser.add_argument('--num_workers', default=2, type=int)
        self.parser.add_argument('--max_train_iters', default=2500, type=int)
        self.parser.add_argument('--max_val_iters', default=50, type=int)

        self.parser.add_argument('--seq_weight', default=0.1, type=float)
        self.parser.add_argument('--idvec_weight', default=1.0, type=float)

        self.parser.add_argument('--facenet_mode', default='arcface', type=str)
        self.parser.add_argument('--facenet_dir', default='./saved_models', type=str)
        
        self.parser.add_argument('--aadblocks_dir', default='./saved_models', type=str)
        self.parser.add_argument('--attencoder_dir', default='./saved_models', type=str)
        self.parser.add_argument('--discriminator_dir', default='./saved_models', type=str)

        self.parser.add_argument('--lr_aad', default=1e-4, type=float)
        self.parser.add_argument('--lr_att', default=1e-4, type=float)
        self.parser.add_argument('--lr_dis', default=1e-4, type=float)

        self.parser.add_argument('--adv_weight', default=0.1, type=float)
        self.parser.add_argument('--att_weight', default=10.0, type=float)
        self.parser.add_argument('--id_weight', default=1.0, type=float)
        self.parser.add_argument('--rec_weight', default=10.0, type=float)

        self.parser.add_argument('--idloss_mode', default='Cos', type=str)
        self.parser.add_argument('--recloss_mode', default='lpips', type=str)

        self.parser.add_argument('--board_interval', default=50, type=int)
        self.parser.add_argument('--save_interval', default=5000, type=int)
        self.parser.add_argument('--image_interval', default=1000, type=int)

        self.parser.add_argument('--exp_dir', default='./experiment', type=str)
        self.parser.add_argument('--trainimg_dir', default='./train_image', type=str)
        self.parser.add_argument('--valimg_dir', default='./validate_image', type=str)
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts



class InjectionOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--seed', default=0, type=int)
        self.parser.add_argument('--rand_select', default='Yes', type=str)
        
        self.parser.add_argument('--max_num', default=10, type=int)
        self.parser.add_argument('--batch_size', default=10, type=int)
        self.parser.add_argument('--num_workers', default=2, type=int)

        self.parser.add_argument('--seq_weight', default=0.1, type=float)
        self.parser.add_argument('--idvec_weight', default=1.0, type=float)  
        
        self.parser.add_argument('--facenet_mode', default='arcface', type=str)
        self.parser.add_argument('--facenet_dir', default='./saved_models', type=str)
        
        self.parser.add_argument('--aadblocks_dir', default='./saved_models', type=str)
        self.parser.add_argument('--attencoder_dir', default='./saved_models', type=str)

        self.parser.add_argument('--seq_type', default='mls', type=str)

        self.parser.add_argument('--exp_dir', default='./experiment', type=str)
        self.parser.add_argument('--img_dir', default='./image_datasets', type=str)

    def parse(self):
        opts = self.parser.parse_args()
        return opts