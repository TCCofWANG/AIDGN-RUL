import sys
sys.path.append("..")
import os
import numpy as np
import shutil
import torch
from torch import optim
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yaml
from N_CMAPSS_Related_PINN.N_CMAPSS_load_data import get_n_cmapss_data_, N_CMAPSSData_index, N_CMAPSSData_index_PINN, get_n_cmapss_data_PINN
from CMAPSS_Related_PINN.load_data_CMAPSS import get_cmapss_data_index, get_cmapss_data_index_PINN
from CMAPSS_Related_PINN.CMAPSS_Dataset import CMAPSSData_index, CMAPSSData_index_PINN

from torch.utils.data import DataLoader

from Model import *
from layers.relobralo import relobralo
from Experiment.Early_Stopping import EarlyStopping
from Experiment.learining_rate_adjust import adjust_learning_rate_class

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

from tool.Write_csv import *
import datetime

from tqdm import tqdm
from Model.Dual_Mixer import DualMLPMixer


torch.autograd.set_detect_anomaly(True)


class Exp_PINN(object):
    def __init__(self, args):
        self.args = args

        self.device = self._acquire_device()

        self._get_path()

        # load dataset
        self.train_data, self.train_loader, self.vali_data, \
        self.vali_loader, self.test_data, self.test_loader, self.input_feature = self._get_data()

        # build the Model:
        self.model = self._get_model()

        # What optimisers and loss functions can be used by the model
        self.optimizer_dict = {"Adam": optim.Adam}

    # choose device
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            print('Use GPU: cuda: ' + str(os.environ["CUDA_VISIBLE_DEVICES"]))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    # ------------------- function to build model -------------------------------------
    def _get_model(self):

        # if self.args.model_name == 'PINN':
        self.r = (np.random.uniform(size=int(1e8)) < 0.9999).astype(int).astype(np.float64)
        self.a = [1, 0, 0.999]
        self.l0 = [1, 1]
        self.l1 = [1, 1]
        self.lamb = [1, 1]
        self.coef = 100
        model = PINN(self.args, hidden_dim=3, derivatives_order=2)

        print("Parameter :", np.sum([para.numel() for para in model.parameters()]))


        return model.double().to(self.device)

    # --------------------------- select optimizer ------------------------------
    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError

        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # ------------------------ get Dataloader -------------------------------------

    #  funnction of load CMPASS Dataset
    def _get_data(self):

        args = self.args
        # train and validation dataset
        if self.args.dataset_name == 'CMAPSS':
            X_train, y_train, index_train, X_vali, y_vali, index_vali, X_test, y_test, index_test, self.global_input = get_cmapss_data_index_PINN(data_path=args.data_path_CMAPSS,
                                                                                Data_id=args.Data_id_CMAPSS,
                                                                                sequence_length=args.input_length,
                                                                                MAXLIFE=args.MAXLIFE_CMAPSS,
                                                                                is_difference=False,
                                                                                normalization=args.normalization_CMAPSS,
                                                                                validation=args.validation)
            self.max_life = args.MAXLIFE_CMAPSS

        elif self.args.dataset_name == 'N_CMAPSS':
            X_train, index_train, y_train, X_vali, index_vali, y_vali, X_test, index_test, y_test, self.max_life  = get_n_cmapss_data_PINN(args=self.args,data_path=args.data_path,
                                                                                                 name=args.Data_id_N_CMAPSS)

        else:
            raise ValueError('without corresponding dataset')

        train_data_set = eval ( self.args.dataset_name + 'Data_index_PINN' ) ( X_train, index_train, y_train )
        vali_data_set = eval ( self.args.dataset_name + 'Data_index_PINN' ) ( X_vali, index_vali, y_vali )
        test_data_set = eval ( self.args.dataset_name + 'Data_index_PINN' ) ( X_test, index_test, y_test )

        input_fea = X_test.shape[-1]
        args.input_feature = input_fea

        train_data_loader = DataLoader(dataset=train_data_set,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       drop_last=True)

        vali_data_loader = DataLoader(dataset=vali_data_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=True)

        test_data_loader = DataLoader(dataset=test_data_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      drop_last=False)

        return train_data_set, train_data_loader, vali_data_set, vali_data_loader, test_data_set, test_data_loader, input_fea


    def save_hparam(self,per_epoch_time):
        # args: args from argparse return
        value2save = {k: v for k, v in vars(self.args).items() if not k.startswith('__') and not k.endswith('__')}
        # 删除影响使用的参数，以便加载
        del_key = ['train', 'resume', 'save_path', 'resume_path', 'batch_size', 'train_epochs', 'learning_rate']
        for key in del_key:
            del value2save[key]

        with open(os.path.join(self.save_path, 'hparam.yaml'), 'a+') as f:
            f.write(yaml.dump(value2save))
        with open(os.path.join(self.save_path, 'hparam.yaml'), 'a+') as f:
            f.write(yaml.dump(per_epoch_time))

    def _get_path(self):
        if not os.path.exists('./logs/'):
            os.makedirs('./logs/')

        exp_id = self.args.save_path
        # save address
        # if self.args.dataset_name == 'CMAPSS':
        #     dataset = self.args.Data_id_CMAPSS
        #     test_dataset = self.args.Data_id_CMAPSS_test
        # elif self.args.dataset_name == 'N_CMAPSS':
        #     data = self.args.Data_id_N_CMAPSS
        # elif self.args.dataset_name == 'XJTU':
        #     data = self.args.Data_id_XJTU
        # else:
        #     data = self.args.dataset_name

        if self.args.dataset_name == 'CMAPSS' :
            dataset = self.args.Data_id_CMAPSS
            test_dataset = self.args.Data_id_CMAPSS_test
        elif self.args.dataset_name == 'N_CMAPSS' :
            dataset = self.args.Data_id_N_CMAPSS
            test_dataset = self.args.Data_id_N_CMAPSS
        else :
            dataset = self.args.dataset_name + self.args.Data_id_XJTU
            test_dataset = self.args.dataset_name + self.args.Data_id_XJTU

        self.path = './logs/' + self.args.dataset_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.model_path = self.path + '/' + self.args.model_name
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if exp_id is not None and exp_id != 'None' and exp_id != 'none':
            self.save_path = self.model_path + '/' + exp_id
            if self.args.train:
                if os.path.exists(self.save_path):
                    shutil.rmtree(self.save_path)
                os.makedirs(self.save_path)

        else:
            # 如为None则自动加一
            path_list = os.listdir(self.model_path)
            if path_list == []:
                self.save_path = self.model_path + '/exp0'

            else:
                path_list = [int(idx[3:]) for idx in path_list]
                self.save_path = self.model_path + '/exp' + str(max(path_list) + 1)

            os.makedirs(self.save_path)
            self.args.save_path = self.save_path

    def _load_checkpoint(self):
        self.checkpoint_dir = self.model_path + '/' + self.args.resume_path + '/best_checkpoint.pth'
        if os.path.exists(self.checkpoint_dir):
            check_point = torch.load(self.checkpoint_dir)
            self.model.load_state_dict(check_point)
        else:
            raise print('checkpoint is not exists')

        # yaml_dir = self.model_path + '/' + self.args.resume_path + '/hparam.yaml'
        # with open(yaml_dir, 'r', encoding='utf-8') as f:
        #     yml = yaml.load(f.read(), Loader=yaml.FullLoader)
        #     key2load = [k for k in yml.keys()]
        #     for k in key2load:
        #         # 重赋值
        #         exec('self.args.' + k + '=yml[k]')

    def start(self):
        if self.args.train:
            # how many step of train and validation:
            train_steps = len(self.train_loader)
            vali_steps = len(self.vali_loader)
            print("train_steps: ", train_steps)
            print("validaion_steps: ", vali_steps)

            # initial early stopping
            early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)

            # initial learning rate
            learning_rate_adapter = adjust_learning_rate_class(self.args, True)
            # choose optimizer
            self.model_optim = self._select_optimizer()

        if self.args.loss_type == 'MSE':
            self.loss_criterion = nn.MSELoss()  # 使用均方误差损失
        elif self.args.loss_type == 'MAE':
            self.loss_criterion = nn.L1Loss()  # 使用平均绝对误差损失
        elif self.args.loss_type == 'QUAN':
            self.loss_criterion = QuantileLoss(quantile=0.3)  # 让模型保守预测
        else:
            raise ValueError("Unsupported loss type. Use 'MSE' or 'MAE'.")

        if self.args.resume:
            print('load checkpoint')
            self._load_checkpoint()
        else:
            print('random init')

        per_epoch_time = {} #存储每一个训练epoch时间
        if self.args.train:
            # training process
            print("start training")
            for epoch in range(self.args.train_epochs):

                # training process
                train_loss, epoch_time = self.training(epoch)
                per_epoch_time[f"epoch_{epoch}"] = epoch_time
                # validation process:
                vali_loss = self.validation(self.vali_loader, self.loss_criterion)

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}. it takse {4:.7f} seconds".format(
                        epoch + 1, train_steps, train_loss, vali_loss, epoch_time))

                # At the end of each epoch, Determine if we need to stop and adjust the learning rate
                early_stopping(vali_loss, self.model, self.save_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                learning_rate_adapter(self.model_optim, vali_loss)

            # 读取最优的参数
            check_point = torch.load(self.save_path + '/' + 'best_checkpoint.pth')
            self.model.load_state_dict(check_point)

        #计算平均每轮训练时间
        values = per_epoch_time.values()  # 提取所有值
        train_average_time = sum(values) / len(values) if values else 0

        self.save_hparam(per_epoch_time)

        # test:
        # 训练参数数目
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Paramaters 数量为:{0}'.format(total_params))

        average_enc_loss, average_enc_overall_loss, overall_score, test_average_time = self.test(self.test_loader)
        print(f"{self.args.dataset_name}: RMSE test performace of enc is: ", average_enc_loss, " of enc overall is: ",
              average_enc_overall_loss, 'socre of'
                                        'enc', overall_score)

        log_path = './logs/new_experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'batch_size', 'best_last_RMSE', 'best_overall_RMSE', 'score', 'windowsize',
                           'd_model', 'dropout', 'kernal_size', 'kernal_stride', 'train', 'total_params','train_average_time','test_average_time',
                           'savepath', 'resume',
                           'resumepath', 'info',
                           ]]
            write_csv(log_path, table_head, 'w+')

        time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间

        resume_dir = self.checkpoint_dir if self.args.resume else None

        if self.args.dataset_name == 'CMAPSS':
            dataset = self.args.Data_id_CMAPSS
        elif self.args.dataset_name == 'N_CMAPSS':
            dataset = self.args.Data_id_N_CMAPSS
        elif self.args.dataset_name == 'XJTU':
            dataset = self.args.Data_id_XJTU
        else:
            dataset = self.args.dataset_name

        a_log = [{'dataset': dataset, 'model': self.args.model_name, 'time': time_now,
                  'LR': self.args.learning_rate,
                  'batch_size': self.args.batch_size,
                  'best_last_RMSE': average_enc_loss, 'best_overall_RMSE': average_enc_overall_loss,
                  'score': overall_score, 'windowsize': self.args.input_length, 'd_model': self.args.d_model,
                  'dropout': self.args.dropout,
                  'kernal_size': self.args.kernel_size,
                  'kernal_stride': self.args.stride,
                  'train': self.args.train,
                  'total_params': total_params,
                  'train_average_time': train_average_time,
                  'test_average_time': test_average_time,
                  'savepath': self.save_path,
                  'resume': self.args.resume,
                  'resumepath': resume_dir,
                  'info': self.args.info,
                  }]
        write_csv_dict(log_path, a_log, 'a+')

    def training(self,epoch):
        start_time = time()

        iter_count = 0
        train_loss = []

        self.model.train()
        for i, (batch_x, idx_x, batch_y) in enumerate(tqdm(self.train_loader)):
            iter_count += 1
            self.model_optim.zero_grad()

            batch_x = batch_x.double().to(self.device)  # [B,window_size,D]
            batch_y = batch_y.double().to(self.device)  # [B,1]



            if self.args.dataset_name == 'XJTU':
                outputs, h, f = self.model(batch_x, mode='train', idx=idx_x)
                loss1 = torch.sqrt(self.loss_criterion(outputs, batch_y))
                loss2 = torch.sqrt(self.loss_criterion(f, torch.zeros(f.shape, dtype=torch.float64).cuda()))
                self.lamb = relobralo(loss_u=loss1, loss_f=self.coef * loss2, alpha=self.a[0], l0=self.l0,
                                      l1=self.l1, lam=self.lamb, T=0.1, rho=self.r[0])
                loss = self.lamb[0] * loss1 + self.lamb[1] * self.coef * loss2
                if len(self.a) > 1:
                    self.a = self.a[1:]
                self.r = self.r[1:]
                losses = [loss1, self.coef * loss2]
                if epoch == 0 and i == 0:
                    self.l0 = losses
                self.l1 = losses
            else:
                batch_y_norm = batch_y / self.max_life
                outputs, h, f = self.model(batch_x, mode='train', idx=idx_x)  # [B ,1]
                loss1 = torch.sqrt(self.loss_criterion(outputs, batch_y_norm))
                loss2 = torch.sqrt(self.loss_criterion(f, torch.zeros(f.shape,dtype=torch.float64).cuda()))
                self.lamb = relobralo(loss_u=loss1, loss_f=self.coef * loss2, alpha=self.a[0], l0=self.l0,
                                                l1=self.l1, lam=self.lamb, T=0.1, rho=self.r[0])
                loss = self.lamb[0] * loss1 + self.lamb[1] * self.coef * loss2
                if len(self.a) > 1:
                    self.a = self.a[1:]
                self.r = self.r[1:]
                losses = [loss1, self.coef * loss2]
                if epoch == 0 and i == 0:
                    self.l0 = losses
                self.l1 = losses
            train_loss.append(loss.item())
            loss.backward()
            self.model_optim.step()
            # ------------------------------------------------

        end_time = time()
        epoch_time = end_time - start_time
        train_loss = np.average(train_loss)  # avgerage loss

        return train_loss, epoch_time

    # ---------------------------------- validation function -----------------------------------------
    def validation(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for i, (batch_x, idx_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.device)
            batch_y = batch_y.double().to(self.device)


            if self.args.dataset_name == 'XJTU':
                outputs, h, f = self.model(batch_x, mode='val', idx=idx_x)
                loss = self.loss_criterion(outputs, batch_y)
            else:
                batch_y_norm = batch_y / self.max_life
                outputs, h, f = self.model(batch_x, mode='val', idx=idx_x)  # [B ,1]
                loss = self.loss_criterion(outputs, batch_y_norm)

            total_loss.append(loss.item())

        average_vali_loss = np.average(total_loss)

        self.model.train()
        return average_vali_loss

    # ----------------------------------- test function ------------------------------------------
    def test(self, test_loader):
        self.model.eval()
        enc_pred = []
        gt = []
        start_time = time()
        for i, (batch_x, idx_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.double().double().to(self.device)
            batch_y = batch_y.double().double().to(self.device)

            if self.args.is_minmax:
                outputs, h, f = self.model(batch_x, mode='test', idx=idx_x)
                outputs = outputs * self.max_life

            else:
                outputs, h, f = self.model(batch_x, mode='test', idx=idx_x)  # outputs[B,window_size]

            batch_y = batch_y.detach().cpu().numpy()
            enc = outputs.detach().cpu().numpy()

            if self.args.dataset_name == 'XJTU' and self.args.is_minmax:
                enc = enc / self.max_life
                batch_y = batch_y / self.max_life

            gt.append(batch_y)
            enc_pred.append(enc)
        end_time = time()
        test_average_time = end_time - start_time
        gt = np.concatenate(gt).reshape(-1, 1)
        enc_pred = np.concatenate(enc_pred).reshape(-1, 1)

        x = np.arange(0, gt.shape[0])

        plt.scatter(x, enc_pred, label='Enc Pred', color='blue')
        plt.scatter(x, gt, label='GT', color='red')

        # 添加标签和标题
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Enc Pred and GT')
        plt.legend()

        fig_path = self.save_path + '/visual_result.png'
        plt.savefig(fig_path)

        # 关闭图像，释放内存
        plt.close()

        if self.args.save_test:
            result_path = self.save_path + '/result.npz'
            np.savez(result_path, test_preds=enc_pred, test_trues=gt, )

        # 算的就是RMSE
        average_enc_loss = np.sqrt(mean_squared_error(enc_pred, gt))
        average_enc_overall_loss = np.sqrt(mean_squared_error(enc_pred, gt))
        # 计算score
        overall_score = self.score_compute(enc_pred, gt)

        return average_enc_loss, average_enc_overall_loss, overall_score,test_average_time

    def score_compute(self, pred, gt):
        # pred [B] gt[B]
        B = pred.shape
        score = 0

        if self.args.dataset_name == 'XJTU':  # 按照这种算法越大越好
            nonzero_idx = np.where(gt != 0.0)[0]
            gt = gt[nonzero_idx]
            gt = gt.reshape(-1, 1)

            pred = pred[nonzero_idx]
            pred = pred.reshape(-1, 1)

            score_list = np.where(pred - gt < 0, np.exp((gt - pred) * 100 * np.log(0.5) / (gt * 20)),
                                  np.exp(-(gt - pred) * 100 * np.log(0.5) / (gt * 5)))
        else:  # 越小越好
            score_list = np.where(pred - gt < 0, np.exp(-(pred - gt) / 13) - 1, np.exp((pred - gt) / 10) - 1)
        # 这里有的paper求均值，有的求和。实验里面先全都求和计算score
        if self.args.dataset_name == 'CMAPSS':
            score = np.sum(score_list)
        else:
            score = np.mean(score_list)
        return score

class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.1):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        errors = target - pred
        loss = torch.max(
            (self.quantile - 1) * errors,
            self.quantile * errors
        )
        return torch.mean(loss)


