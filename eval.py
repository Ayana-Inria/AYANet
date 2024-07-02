import datasets
from models.AYANet import AYANet, AYANet2
import os
import csv
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

import datasets
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
import misc.utils as utils
from misc.utils import de_norm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

from thop import profile

class CDEval():
    def __init__(self, arguments):
        self.args = arguments

        self.device = torch.device("cuda:%s" % self.args.gpu_ids[0] if torch.cuda.is_available() and len(self.args.gpu_ids)>0
                                   else "cpu")

        # self.net_G = TANet(self.args.encoder_arch, self.args.local_kernel_size, self.args.attn_stride,
        #                    self.args.attn_padding, self.args.attn_groups, self.args.drtam, self.args.refinement)

        self.net_G = AYANet2(self.args.encoder_arch, self.args.decoder_arch, self.args.local_kernel_size, self.args.attn_stride,
                           self.args.attn_padding, self.args.attn_groups, self.args.drtam, self.args.refinement)

        self.dataset_eval_loader = DataLoader(datasets.bcd_eval(pjoin(self.args.datadir), self.args.test_split),
                                          num_workers=self.args.num_workers, batch_size=self.args.batch_size,
                                          shuffle=False)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        self.ckpt_list = []
        if self.args.checkpoint != 'All':
            # self.checkpoint = pjoin(self.args.checkpointdir, self.args.checkpoint)
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
    
    def _visualize_weight(self):
        checkpoint = torch.load(self.checkpoint, map_location=self.device)

        weights = checkpoint['model_G_state_dict']
        
        ### GFN ###
        # count = 24
        # convs = []
        # gabors = []
        # for weight in list(weights):
        #     if count == -1 : break
        #     print(weight)
        #     kernel = weights[weight]
        #     if 'MFilters' in weight:
        #         kernel = np.asarray(kernel.cpu()).astype('f').transpose(1, 2, 0)
        #         gabors.append(kernel)
        #         print('kernel size : ', kernel.shape)
        #     elif 'gfn' in weight:
        #         _, _, c, k, _ = kernel.size()
        #         kernel = kernel.view(c, k, k)
        #         kernel = np.asarray(kernel.cpu()).astype('f').transpose(1, 2, 0)
        #         convs.append(kernel)
        #         print('kernel size : ', kernel.shape)
            
            # vis.append(weights[weight])
            # print(weights[weight].size())
        #     count -= 1

        # file_name = './weights_with_gabor.npz'
        # np.savez(file_name, conv=convs, gabor=gabors)
        # print('Saved..')

        count = 24


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

    def _extract_features(self):
        from PIL import Image
        import numpy as npi
        import matplotlib.pyplot as plt

        t0_path = '/home/priscilla/Codes/AYANet/features/test_34_1_3.png'
        t1_path = '/home/priscilla/Codes/AYANet/features/test_18_0_3.png'
        destination = '/home/priscilla/Codes/AYANet/features/'

        img_t0 = np.asarray(Image.open(t0_path).convert('RGB'))
        img_t1 = np.asarray(Image.open(t1_path).convert('RGB'))
        img_t0 = np.asarray(img_t0).astype('f').transpose(2, 0, 1)
        img_t1 = np.asarray(img_t1).astype('f').transpose(2, 0, 1)
        img_t0 = torch.from_numpy(img_t0)
        C, H, W = img_t0.size()
        img_t0 = img_t0.view(1, C, H, W)
        img_t1 = torch.from_numpy(img_t1)
        img_t1 = img_t1.view(1, C, H, W)

        inputs = torch.cat((img_t0, img_t1), axis=1).to(self.device)
        # inputs = torch.from_numpy(np.concatenate((img_t0, img_t1), axis=0)).to(self.device)
        print('Image size : ', inputs.size())
        self.G_pred, gabor_feat0, gabor_feat1 = self.net_G(inputs)

        for i, feat in enumerate(gabor_feat0):
            feat = feat.to('cpu').detach()
            _, C, H, W = feat.size()
            feat = feat.view(C, H, W)
            print('features size : ', feat.size())
            feat0 = np.asarray(feat).astype('f').transpose(1, 2, 0)
            # feat1 = np.asarray(gabor_feat1).astype('f').transpose(1, 2, 0)
            
            px = 8
            py = int(C/px)
            ix = 1
            for _ in range(px*py):
                # ax = plt.subplot(px, py, ix)
                # ax.set_xticks([])
                # ax.set_yticks([])
                plt.imshow(feat0[:,:,ix-1], cmap='gray')
                ix += 1
            # plt.imshow(feat0, cmap='gray')
            # plt.title("Features stage %d" % (i))
                name = "feat_stage=%d_filter_%d.jpg" % (i,ix)
                name = os.path.join(destination, name)
                plt.savefig(name)
                print('saved to : ', name)



    def _visualize_grad(self):
        class EncoderExtractor(torch.nn.Module):
            def __init__(self, model):
                super(EncoderExtractor, self).__init__()
                self.model = model
                self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
                self.encoder = self.feature_extractor[0]

            def __call__(self,x):
                # self.encoder(x) in layer 4 of encoder returns list of 4 tensors (presumably the features of each block)
                # for some reasons, this __call__ part is not overriding the original output which makes the error of 'tuple does not have .cpu() method'
                # because in the library, it is expected that the result is in tensor type
                return self.encoder(x)[0]

                
        class ChangeDetectionTarget:
            def __init__(self, category, mask):
                self.category = category
                self.mask = torch.from_numpy(mask)
                if torch.cuda.is_available():
                    self.mask = self.mask.cuda()
                
            def __call__(self, model_output):
                # return (model_output[self.category, :, : ] * self.mask).sum()
                return torch.argmax(model_output, dim=1)
        
        model = EncoderExtractor(self.net_G)
        inputs_img = self.batch['I'].to(self.device)
        img_t0,img_t1 = torch.split(inputs_img,3,1)
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1).to('cpu')
        # change_mask = G_pred * 255
        change_mask_float = np.float32(G_pred.numpy())

        target_layers = [self.net_G.encoder1.layer4[-1]]
        targets = [ChangeDetectionTarget(1, change_mask_float)]

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        grayscale_cam = cam(input_tensor = inputs_img, targets=targets)[0,:]
        cam_image = show_cam_on_image(img_t0, grayscale_cam, use_rgb=True)
    
        img_grad = Image.fromarray(cam_image)
        file_name = os.path.join(
            self.args.visdir, 'grad_' + str(self.batch_id)+'.jpg')
        img_grad.save(file_name)

    def eval(self):
        for ckpt in self.ckpt_list:
            self.checkpoint = ckpt
            self._load_checkpoint()

            self.logger.write('Begin evaluation...\n')
            self._clear_cache()

            # self._extract_features()
            self.net_G.eval()
            # self._visualize_weight()


            for self.batch_id, (inputs_test, mask_test) in enumerate(tqdm(self.dataset_eval_loader)):
                with torch.no_grad():
                    self._forward(inputs_test, mask_test)
                    # self._visualize_weight()
                # self._visualize_grad()
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
    parser.add_argument('--local-kernel-size',type=int, default=1)
    parser.add_argument('--attn-stride', type=int, default=1)
    parser.add_argument('--attn-padding', type=int, default=0)
    parser.add_argument('--attn-groups', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--drtam', action='store_true')
    parser.add_argument('--refinement', action='store_true')
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
