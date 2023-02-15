import os
import numpy as np
from PIL import Image
from scipy import spatial 
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix



def alignment(images):
  return F.interpolate(images[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True)


def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def tensor2img(var):
    # var: 3 x 256 x 256 --> 256 x 256 x 3
    var = var.cpu().detach().numpy().transpose([1,2,0])
    #var = ((var+1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def visualize_train_results(cor_dict, dis_num):
    fig = plt.figure(figsize=(52, 4*dis_num))
    gs = fig.add_gridspec(nrows=dis_num, ncols=13)

    seq_len = cor_dict['seq_len']
    seq = cor_dict['seq'].cpu().detach().numpy()
    seq_weighted = cor_dict['seq_weighted'].cpu().detach().numpy()

    for img_idx in range(dis_num):
        fig.add_subplot(gs[img_idx, 0])
        img_org = tensor2img(cor_dict['img_org'][img_idx])
        plt.imshow(img_org)
        plt.title('Original Image')

        fig.add_subplot(gs[img_idx, 1])
        img_rec = tensor2img(cor_dict['img_rec'][img_idx]) 
        plt.imshow(img_rec)
        plt.title('Reconstrcted Image')

        fig.add_subplot(gs[img_idx, 2])
        img_residual = Image.fromarray(np.abs(np.array(img_org) - np.array(img_rec)))
        img_residual = img_residual.convert('L')
        plt.imshow(img_residual, cmap=plt.get_cmap('gray'))
        plt.title('Residual Image')

        fig.add_subplot(gs[img_idx, 3])
        org_gray = img_org.convert('L')
        org_f = np.fft.fft2(np.array(org_gray))
        org_fshift = np.fft.fftshift(org_f)
        forg_img = np.log(np.abs(org_fshift))
        plt.imshow(forg_img, cmap='jet')
        plt.title('Original Fourier')

        fig.add_subplot(gs[img_idx, 4])
        rec_gray = img_rec.convert('L')
        rec_f = np.fft.fft2(np.array(rec_gray))
        rec_fshift = np.fft.fftshift(rec_f)
        frec_img = np.log(np.abs(rec_fshift))
        plt.imshow(frec_img, cmap='jet')
        plt.title('Reconstrcted Fourier')

        fig.add_subplot(gs[img_idx, 5])
        id_input = cor_dict['id_input'][img_idx].cpu().detach().numpy()
        input_acorr = np.correlate(id_input, seq, 'full')
        plt.plot(np.arange(-seq_len+1, seq_len), input_acorr, '.-')
        plt.grid()
        plt.title('Input_Id AutoCorrelation')

        fig.add_subplot(gs[img_idx, 6])
        id_rec = cor_dict['id_rec'][img_idx].cpu().detach().numpy()
        rec_acorr = np.correlate(id_rec, seq, 'full')
        plt.plot(np.arange(-seq_len+1, seq_len), rec_acorr, '.-')
        plt.grid()
        plt.title('Rec_Id AutoCorrelation')

        fig.add_subplot(gs[img_idx, 7])
        id_org = cor_dict['id_org'][img_idx].cpu().detach().numpy()
        org_acorr = np.correlate(id_org, seq, 'full')
        plt.plot(np.arange(-seq_len+1, seq_len), org_acorr, '.-')
        plt.grid()
        plt.title('Origin_Id AutoCorrelation')

        fig.add_subplot(gs[img_idx, 8])
        id_recres = id_rec - id_org
        recres_acorr = np.correlate(id_recres, seq, 'full')
        plt.plot(np.arange(-seq_len+1, seq_len), recres_acorr, '.-')
        plt.grid()
        plt.title('Rec_Residual AutoCorrelation')

        fig.add_subplot(gs[img_idx, 9])
        plt.plot(id_input)
        plt.grid()
        orgin_similarity = 1 - spatial.distance.cosine(id_org, id_input)
        noiin_similarity = 1 - spatial.distance.cosine(seq_weighted, id_input)
        plt.title('Input_Id OrgIn:{:.2f} NoiIn:{:.2f}'.format(orgin_similarity, noiin_similarity))

        fig.add_subplot(gs[img_idx, 10])
        plt.plot(id_rec)
        plt.grid()
        inrec_similarity = 1 - spatial.distance.cosine(id_input, id_rec)
        noirec_similarity = 1 - spatial.distance.cosine(seq_weighted, id_rec)
        plt.title('Rec_Id InRec:{:.2f} NoiRec:{:.2f}'.format(inrec_similarity, noirec_similarity))

        fig.add_subplot(gs[img_idx, 11])
        plt.plot(id_recres)
        plt.grid()
        recres_similarity = 1 - spatial.distance.cosine(id_rec, id_recres)
        noirec_similarity = 1 - spatial.distance.cosine(seq_weighted, id_recres)
        plt.title('Rec_Residual RecRes:{:.2f} NosRec:{:.2f}'.format(recres_similarity, noirec_similarity))

        fig.add_subplot(gs[img_idx, 12])
        plt.plot(seq_weighted)
        plt.grid()
        seqnoi_similarity = 1 - spatial.distance.cosine(seq, seq_weighted)
        plt.title('Input_seq_weighted seqNoi:{:.2f}'.format(seqnoi_similarity))
    plt.tight_layout()
    return fig


def visualize_correlation(seq, id, img, save_path):
    seq_len = len(seq)
    
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(nrows=1, ncols=2)

    fig.add_subplot(gs[0, 0])
    img = tensor2img(img)
    plt.imshow(img)
    plt.title('Input Image')
    
    fig.add_subplot(gs[0, 1])
    id = id.to(torch.device('cpu')).detach().numpy()
    correlation = np.correlate(id, seq, 'full')
    plt.plot(np.arange(-seq_len+1, seq_len), correlation, '.-')
    plt.grid()
    plt.title('Correlation')

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return fig


def generate_seqstate(size=9):
    randint = np.random.randint(low=0, high=2, size=size)
    random_state = np.ones(shape=9) if sum(randint) == 0 else randint
    return random_state


def statistic_correlation(id_vector, seq):
    seq_len = len(seq)

    peak_list = []
    avg_list = []

    for idx in range(len(id_vector)):
        id_vec = id_vector[idx].to(torch.device('cpu')).detach().numpy()
        
        corr_abs = np.abs(np.correlate(id_vec, seq, 'full'))

        peak = corr_abs[seq_len-1]

        corr_delpeak = np.delete(corr_abs, obj=seq_len-1, axis=None)

        avg = np.mean(corr_delpeak)

        peak_list.append(peak)
        avg_list.append(avg)
        
    return peak_list, avg_list


def calculatie_correlation(id_vector, seq, threshold):
    seq_len = len(seq)

    verify_list = []

    for idx in range(len(id_vector)):
        id_vec = id_vector[idx].to(torch.device('cpu')).detach().numpy()

        corr_abs = np.abs(np.correlate(id_vec, seq, 'full'))

        peak = corr_abs[seq_len-1]

        corr_delpeak = np.delete(corr_abs, obj=seq_len-1, axis=None)

        avg = np.mean(corr_delpeak)

        if peak/avg >= threshold:
            verify_list.append(1)
        else:
            verify_list.append(0)

    return verify_list


def evaluation(label, pred):
    
    accuracy = accuracy_score(y_true=label, y_pred=pred)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true=label, y_pred=pred, average='binary')

    tn, fp, fn, tp = confusion_matrix(y_true=label, y_pred=pred).ravel()

    return accuracy, precision, recall, f1_score, tn, fp, fn, tp