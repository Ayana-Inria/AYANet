import datasets
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from thop import profile

from models.AYANet import AYANet
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
import misc.utils as utils
from misc.utils import de_norm


class CDEval():
    def __init__(self, arguments):
        self.args = arguments

        self.device = torch.device("cuda:%s" % self.args.gpu_ids[0] if torch.cuda.is_available() and len(self.args.gpu_ids)>0
                                   else "cpu")

        self.net_G = AYANet(self.args.encoder_arch, self.args.decoder_arch)

        self.dataset_eval_loader = DataLoader(datasets.bcd_eval(pjoin(self.args.datadir), self.args.test_split),
                                          num_workers=self.args.num_workers, batch_size=self.args.batch_size,
                                          shuffle=False)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        self.ckpt_list = []
        if self.args.checkpoint != 'All':
            ckpt_path = pjoin(self.args.checkpointdir, self.args.checkpoint)
            self.ckpt_list.append(ckpt_path)
        else:
            for el in os.listdir(self.args.checkpointdir):
                if el.endswith('.pt'):
                    ckpt_path = pjoin(self.args.checkpointdir, el)
                    self.ckpt_list.append(ckpt_path)

        logger_path = os.path.join(self.args.checkpointdir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.batch = None
        self.batch_id = 0


    def _load_checkpoint(self):

        try:
            self.logger.write('loading checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(self.checkpoint, map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        except:
            raise FileNotFoundError('no such checkpoint %s' % self.checkpoint)


    def _clear_cache(self):
        self.running_metric.clear()

    def _update_metric(self):
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataset_eval_loader)

        if np.mod(self.batch_id, 100) == 1:
            message = '[%d,%d],  running_mf1: %.5f\n' %\
                        (self.batch_id, m, running_acc)
            self.logger.write(message)

        img_t0, img_t1 = torch.split(self.batch['I'], 3, 1)
        vis_input = utils.make_numpy_grid(de_norm(img_t0))
        vis_input2 = utils.make_numpy_grid(de_norm(img_t1))

        vis_pred = utils.make_numpy_grid(self._visualize_pred())

        vis_gt = utils.make_numpy_grid(self.batch['L'])
        val, ct = np.unique(vis_gt, return_counts=True)

        if (1 in val):
            if ct[np.where(val==1)] > 16384:
            
                diff = vis_pred - vis_gt

                mask_tp = np.all(diff == [254, 254, 254], axis=-1)
                mask_fp = np.all(diff == [255, 255, 255], axis=-1)
                mask_fn = np.all(diff == [-1, -1, -1], axis=-1)
                mask_tn = np.all(diff == [0, 0, 0], axis=-1)

                diff[mask_tp] = [1, 1, 1]
                diff[mask_fp] = [0.4, 1, 1]
                diff[mask_fn] = [1, 0.4, 0.4]

                vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt, diff], axis=0)
                vis = np.clip(vis, a_min=0.0, a_max=1.0)
                file_name = os.path.join(
                    self.args.visdir, 'eval_' + str(self.batch_id)+'.jpg')
                plt.imsave(file_name, vis)

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.args.checkpointdir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

        ckpt = self.checkpoint.split('/')[-1]
        message += ('\n' + str(ckpt) + '\n')
        with open(os.path.join(self.args.checkpointdir, 'report.txt' % (self.epoch_acc)), mode='a') as file:
            file.write(message)


    def _forward(self, inputs, label):
        data = {}
        data['I'] = inputs
        data['L'] = label
        self.batch = data
        
        inputs_img = self.batch['I'].to(self.device)
        # macs, params = profile(self.net_G, inputs=(inputs_img, ))
        # print('MACs : ')
        # print(macs)
        # print('PARAMs : ')
        # print(params)

        self.G_pred = self.net_G(inputs_img)
        # self.G_pred, self.G_mask1, self.G_mask2 = self.net_G(inputs_img)


    def eval(self):
        for ckpt in self.ckpt_list:
            self.checkpoint = ckpt
            self._load_checkpoint()

            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.net_G.eval()

            for self.batch_id, (inputs_test, mask_test) in enumerate(tqdm(self.dataset_eval_loader)):
                with torch.no_grad():
                    self._forward(inputs_test, mask_test)
                self._collect_running_batch_states()
            self._collect_epoch_states()


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='START EVALUATING...')
    parser.add_argument('--gpu_ids', default=0)
    parser.add_argument('--dataset', type=str, default='bcd', required=True)
    parser.add_argument('--test_split', type=str, default='test', required=True)
    parser.add_argument('--datadir',required=True)
    parser.add_argument('--resultdir',required=True)
    parser.add_argument('--checkpointroot', type=str, required=True)
    parser.add_argument('--visroot', type=str, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--encoder-arch', type=str, required=True)
    parser.add_argument('--decoder-arch', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--store-imgs', action='store_true')
    parser.add_argument('--multi-gpu', action='store_true', help='processing with multi-gpus')
    parser.add_argument('--project_name', type=str)

    args = parser.parse_args()
    args.checkpointdir = pjoin(args.checkpointroot, args.project_name)
    os.makedirs(args.checkpointdir, exist_ok=True)

    args.visdir = pjoin(args.visroot, args.project_name)
    os.makedirs(args.visdir, exist_ok=True)


    if args.dataset == 'bcd':
        eval = CDEval(args)
        eval.eval()
    else:
        print('Error: Cannot identify the dataset...(dataset: pcd or vl_cmu_cd)')
        exit(-1)
