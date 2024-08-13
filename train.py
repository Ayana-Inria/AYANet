import os
import csv
import torch
from models.AYANet import AYANet
import numpy as np
import datasets
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger, Timer
# import misc.utils as utils


class criterion_CEloss(nn.Module):
    def __init__(self,weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)
    def forward(self,output,target):
        return self.loss(F.log_softmax(output, dim=1), target)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-7

    def forward(self, logits, true):
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes, device=logits.device)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension() + 1))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_loss

class DiceLoss2(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss2, self).__init__()
        self.eps = 1e-6

    def forward(self, input, target):
        
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        B, H, W = target.size()
        C = input.size()[1]
        one_hot = torch.zeros(B, C, H, W, device=input.device)
        # create the labels one hot tensor
        target_one_hot = one_hot.scatter(1, target.unsqueeze(1), 1.0)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)

class BceDiceLoss(nn.Module):
    def __init__(self):
        super(BceDiceLoss, self).__init__()
        self.bce = criterion_CEloss()
        self.dice = DiceLoss()

    def forward(self, output, mask1, mask2, target):
        return self.bce(output, target) + self.dice(output, target) + 0.5*(self.bce(mask1, target) + self.dice(mask1, target)) + 0.5*(self.bce(mask2, target) + self.dice(mask2, target))


class DynamicComboLoss(nn.Module):
    def __init__(self):
        super(DynamicComboLoss, self).__init__()
        self.nllloss = criterion_CEloss()
        self.dice = DiceLoss()

    def forward(self, input, target, dice_weight):
        total_loss = dice_weight*self.dice(input, target) + (1-dice_weight)*self.nllloss(input, target)
        return total_loss

class CosOneCycle:
    def __init__(self, optimizer, max_lr, epochs, min_lr=None, up_rate=0.3):  # max=0.0035, min=0.00035
        self.optimizer = optimizer
        self.max_lr = max_lr
        if min_lr is None:
            self.min_lr = max_lr / 10
        else:
            self.min_lr = min_lr
        self.final_lr = self.min_lr / 50
        self.new_lr = self.min_lr

        self.step_i = 0
        self.epochs = epochs
        self.up_rate = up_rate
        assert up_rate < 0.5, "up_rate should be smaller than 0.5"

    def step(self):
        self.step_i += 1
        if self.step_i < (self.epochs*self.up_rate):
            self.new_lr = 0.5 * (self.max_lr - self.min_lr) * (np.cos((self.step_i/(self.epochs*self.up_rate) + 1) * np.pi) + 1) + self.min_lr
        else:
            self.new_lr = 0.5 * (self.max_lr - self.final_lr) * (np.cos(((self.step_i - self.epochs * self.up_rate) / (self.epochs * (1 - self.up_rate))) * np.pi) + 1) + self.final_lr

        if len(self.optimizer.state_dict()['param_groups']) == 1:
            self.optimizer.param_groups[0]["lr"] = self.new_lr
        elif len(self.optimizer.state_dict()['param_groups']) == 2:  # for finetune
            self.optimizer.param_groups[0]["lr"] = self.new_lr / 10
            self.optimizer.param_groups[1]["lr"] = self.new_lr
        else:
            raise Exception('Error. You need to add a new "elif". ')


class CDTrain():
    def __init__(self, arguments):
        self.args = arguments

        print('Encoder:' + self.args.encoder_arch)

        # if self.args.drtam:
        #     folder_name = 'DR-TANet'
        #     print('Dynamic Receptive Temporal Attention Network (DR-TANet)')
        # else:
        #     folder_name = 'TANet_k={}'.format(self.args.local_kernel_size)
        #     print('Temporal Attention Network (TANet)')

        # folder_name += ('_' + self.args.encoder_arch)

        # if self.args.refinement:
        #     folder_name += '_ref'
        #     print('Adding refinement...')

        self.dataset_train_loader = DataLoader(datasets.bcd(pjoin(self.args.datadir), self.args.split),
                                          num_workers=self.args.num_workers, batch_size=self.args.batch_size,
                                          shuffle=True)
        self.dataset_test_loader = DataLoader(datasets.bcd(pjoin(self.args.datadir), self.args.val_split),
                                          num_workers=self.args.num_workers, batch_size=self.args.batch_size,
                                          shuffle=True)

        self.device = torch.device("cuda:%s" % self.args.gpu_ids[0] if torch.cuda.is_available() and len(self.args.gpu_ids)>0
                                   else "cpu")

        self.net_G = AYANet(self.args.encoder_arch, self.args.decoder_arch)

        print(self.net_G)
        self.net_G = self._init_net(self.net_G, init_type='normal', init_gain=0.02, gpu_ids=self.args.gpu_ids)
        
        if self.args.optimizer == 'adam':
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),lr=self.args.lr,betas=(0.9,0.999))
        elif self.args.optimizer == 'adamw':
            self.optimizer_G = torch.optim.AdamW(self.net_G.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=0.01)

        self.batch_id = 0
        self.epoch_id = 0

        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(self.args.max_epochs + 1)
            return lr_l
        # lambda_lr = lambda epoch:(float)(self.args.max_epochs*len(self.dataset_train_loader)-self.epoch_id)/(float)(self.args.max_epochs*len(self.dataset_train_loader))
        
        self.exp_lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G,lr_lambda=lambda_rule)
        # self.exp_lr_scheduler_G = CosOneCycle(self.optimizer_G, max_lr=self.args.lr, epochs=self.args.max_epochs)

        # self.loss_f = BceDiceLoss()
        self.loss_f = criterion_CEloss()
        # self.loss_f = DynamicComboLoss()

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(self.args.checkpointdir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(self.args.__dict__)

        # visualize tensorboard
        self.visual_writer = SummaryWriter(self.args.tbdir)
        
        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0

        self.G_pred = None
        self.G_mask1 = None
        self.G_mask2 = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        
        # self.vis_dir = self.args.vis_dir

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.args.checkpointdir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.args.checkpointdir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.args.checkpointdir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.args.checkpointdir, 'train_acc.npy'))


        
    def _init_net(self, net, init_type='normal', init_gain=0.02, gpu_ids=0):
        """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

        Return an initialized network.
        """
        # if self.args.multi_gpu:
        #     net = nn.DataParallel(net).to(self.device)
        # else:
        #     net = net.to(self.device)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net = net.to(self.device)
            if len(gpu_ids) > 1:
                net = nn.DataParallel(net).to(self.device)  # multi-GPUs

        self._init_weights(net, init_type, init_gain=init_gain)
        return net

    def _init_weights(self, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.args.checkpointdir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.args.checkpointdir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        # elif self.args.pretrain is not None:
        #     print("Initializing backbone weights from: " + self.args.pretrain)
        #     self.net_G.load_state_dict(torch.load(self.args.pretrain), strict=False)
        #     self.net_G.to(self.device)
        #     self.net_G.eval()
        else:
            print('training from scratch...')

    def _clear_cache(self):
        self.running_metric.clear()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataset_train_loader)
        if self.is_training is False:
            m = len(self.dataset_test_loader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.args.max_epochs-1, self.batch_id, m,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)

        # running_acc contains class prec and recall?
        self.visual_writer.add_scalar('Loss-Epoch', self.G_loss.item(), self.epoch_id)
        self.visual_writer.add_scalar('LR-Epoch', self.optimizer_G.param_groups[0]['lr'], self.epoch_id)


        # if np.mod(self.batch_id, 500) == 1:
        #     vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        #     vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

        #     vis_pred = utils.make_numpy_grid(self._visualize_pred())

        #     vis_gt = utils.make_numpy_grid(self.batch['L'])
        #     vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        #     vis = np.clip(vis, a_min=0.0, a_max=1.0)
        #     file_name = os.path.join(
        #         self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
        #                       str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
        #     plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.args.max_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.args.checkpointdir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.args.checkpointdir, 'val_acc.npy'), self.VAL_ACC)

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.args.checkpointdir, ckpt_name))

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            if int(self.epoch_id) > (int(self.args.max_epochs-1)/2) :
                self.best_val_acc = self.epoch_acc
                self.best_epoch_id = self.epoch_id
                ckpt_name = 'epoch_' + str(self.epoch_id) + '_acc_' + str(self.epoch_acc) + '.pt'
                self._save_checkpoint(ckpt_name=ckpt_name)
                self._save_checkpoint(ckpt_name='best_ckpt.pt')
                self.logger.write('*' * 10 + 'Best model updated!\n')
                self.logger.write('\n')

    def _forward(self, inputs, label):
        data = {}
        data['I'] = inputs
        data['L'] = label
        self.batch = data
        # self.batch['I'] = inputs
        # self.batch['L'] = label
        
        inputs_img = self.batch['I'].to(self.device)
        # label_img = self.batch['L'].to(self.device)

        self.G_pred = self.net_G(inputs_img)
        # self.G_pred, self.G_mask1, self.G_mask2 = self.net_G(inputs_img)

    def train(self):
        '''
        self.batch['I'] = concat of self.batch['A'] and self.batch['B']
        skipped visualization
        skipped timer
        '''
        # Train from scratch or continue training from the last checkpoint
        self._load_checkpoint()

        param_num = sum(p.numel() for p in self.net_G.parameters())
        print('Number of parameters : ', param_num)

        # Train for x epochs
        for self.epoch_id in range(self.epoch_to_start, self.args.max_epochs):
            self._clear_cache()
            self.is_training = True
            # Set model to training mode
            self.net_G.train()

            # Iterate over data.
            # self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            
            for self.batch_id ,(inputs_train, label_train) in enumerate(tqdm(self.dataset_train_loader)):
                # Feed forward
                self._forward(inputs_train, label_train)

                # Update weights
                self.optimizer_G.zero_grad()
                # dice_weight = 0.5 * self.epoch_id / self.args.max_epochs
                # self.G_loss = self.loss_f(self.G_pred, self.batch['L'].to(self.device)[:,0], dice_weight)
                self.G_loss = self.loss_f(self.G_pred, self.batch['L'].to(self.device)[:,0])
                # self.G_loss = self.loss_f(self.G_pred, self.G_mask1, self.G_mask2, self.batch['L'].to(self.device)[:,0])
                self.G_loss.backward()
                self.optimizer_G.step()

                # Update metrics and other stuff (visualization, etc.)
                self._collect_running_batch_states()
                # self.visual_writer(self.net_G, inputs_train)


            self._collect_epoch_states()
            self._update_training_acc_curve()
            # Update lr scheduler
            self.exp_lr_scheduler_G.step()

            # Evaluation per epoch
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            for batch_id, (inputs_val, mask_val) in enumerate(tqdm(self.dataset_test_loader)):
                with torch.no_grad():
                    self._forward(inputs_val, mask_val)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            # Update checkpoints
            self._update_val_acc_curve()
            self._update_checkpoints()


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Arguments for training...")
    parser.add_argument('--dataset', type=str, default='bcd', required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default='val')
    parser.add_argument('--gpu_ids', default=0)
    parser.add_argument('--checkpointroot', type=str, required=True)
    parser.add_argument('--visroot', type=str, required=True)
    parser.add_argument('--tbroot', type=str, required=True)
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--multi-gpu',action='store_true',help='training with multi-gpus')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch-save', type=int, default=20)
    parser.add_argument('--step-test', type=int, default=200)
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--encoder-arch', type=str, required=True)
    parser.add_argument('--decoder-arch', type=str, required=True)
    parser.add_argument('--project_name', type=str)

    args = parser.parse_args()
    args.checkpointdir = pjoin(args.checkpointroot, args.project_name)
    os.makedirs(args.checkpointdir, exist_ok=True)

    args.visdir = pjoin(args.visroot, args.project_name)
    os.makedirs(args.visdir, exist_ok=True)

    args.tbdir = pjoin(args.tbroot, args.project_name)
    os.makedirs(args.visdir, exist_ok=True)

    if args.dataset == 'bcd':
        # train = train_bcd(parser.parse_args())
        # train.Init()
        train = CDTrain(args)
        train.train()
    else:
        print('Error: Cannot identify the dataset...(dataset: bcd or pcd or vl_cmu_cd)')
        exit(-1)








