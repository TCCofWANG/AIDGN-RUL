import random
import sys

from Model.CDSG import CDSG
from Model.SDAGCN import SDAGCN

sys.path.append ( ".." )
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
from N_CMAPSS_Related.N_CMAPSS_load_data import get_n_cmapss_data_, N_CMAPSSData
from CMAPSS_Related.load_data_CMAPSS import get_cmapss_data_
from CMAPSS_Related.CMAPSS_Dataset import CMAPSSData
from torch.utils.data import DataLoader

from Model import *

from Experiment.Early_Stopping import EarlyStopping
from Experiment.learining_rate_adjust import adjust_learning_rate_class
from Experiment.HTS_Loss_Function import Weighted_MSE_Loss, MSE_Smoothness_Loss

from tool.Write_csv import *
import datetime

from tqdm import tqdm

"""
This file only used for CMAPSS Datase
"""


class Exp ( object ) :
    def __init__(self, args) :
        self.args = args
        self.device = self._acquire_device ()

        self._get_path()

        # load CMAPSS dataset
        self.train_data, self.train_loader, self.vali_data, self.vali_loader, self.test_data, self.test_loader, self.input_feature = self._get_data ()

        # build the Inception-Attention Model:
        self.model = self._get_model()

        # What optimisers and loss functions can be used by the model
        self.optimizer_dict = {"Adam" : optim.Adam}
        self.criterion_dict = {"MSE" : nn.MSELoss, "CrossEntropy" : nn.CrossEntropyLoss,
                               "WeightMSE" : Weighted_MSE_Loss, "smooth_mse" : MSE_Smoothness_Loss}

    # choose device
    def _acquire_device(self) :
        if self.args.use_gpu :
            device = torch.device ( 'cuda' )
            print ( 'Use GPU: cuda: ' + str ( os.environ["CUDA_VISIBLE_DEVICES"] ) )
        else :
            device = torch.device ( 'cpu' )
            print ( 'Use CPU' )
        return device

    # ------------------- function to build model -------------------------------------
    def _get_model(self) :
        if self.args.model_name == 'Transformer' :
            model = Transformer ( self.args, input_feature = self.input_feature )

        elif self.args.model_name == 'Autoformer' :
            model = Autoformer ( self.args, input_feature = self.input_feature )

        elif self.args.model_name == 'PatchTST' :
            model = PatchTST ( self.args, input_feature = self.input_feature )

        elif self.args.model_name == 'Dual_Mixer':
            model = DualMLPMixer(self.args, self.input_feature)

        elif self.args.model_name == 'CNN' :
            model = CNN ( self.args, input_feature = self.input_feature )

        elif self.args.model_name == 'Transformer_domain' :
            model = Transformer_domain ( self.args, input_feature = self.input_feature )

        elif self.args.model_name == 'FCSTGNN' :
            self.args.k = 1
            self.args.conv_kernel = 2
            self.args.moving_window = [2, 2]
            self.args.stride = [1, 2]
            self.args.pool_choice = 'mean'
            self.args.decay = 0.7

            self.args.time_denpen_len = 10
            self.args.lstmout_dim = 10

            self.args.conv_time_CNN = 10
            self.args.conv_out = 7
            self.args.num_windows = (self.args.input_length // self.args.patch_size - 1) + (
                    self.args.input_length // self.args.patch_size // 2)
            self.args.hidden_dim = 8
            self.args.window_sample = self.args.input_length  # 1,3 :30   2,4:50
            self.args.lstmhidden_dim = 8
            self.args.num_sensor = self.input_feature

            if self.args.Data_id_CMAPSS == "FD001" :
                self.args.patch_size = 5
                self.args.time_denpen_len = 6
                self.args.conv_out = 7
                self.args.num_windows = 8
                self.args.lstmout_dim = 32
                self.args.hidden_dim = 8
                self.args.window_sample = 30
                self.args.conv_time_CNN = 6
                self.args.lstmhidden_dim = 8
                # self.args.time_denpen_len = 6
                # self.args.lstmout_dim = 32
                # self.args.conv_time_CNN = 6

            if self.args.Data_id_CMAPSS == "FD002" :
                self.args.data_sub = 2
                self.args.patch_size = 5
                self.args.time_denpen_len = 10
                self.args.conv_out = 7
                self.args.num_windows = 14
                self.args.lstmout_dim = 12
                self.args.hidden_dim = 8
                self.args.window_sample = 50
                self.args.conv_time_CNN = 10
                self.args.lstmhidden_dim = 8
                # self.args.patch_size = 5
                # self.args.time_denpen_len = 10
                # self.args.lstmout_dim = 12
                # self.args.conv_time_CNN = 10

            if self.args.Data_id_CMAPSS == "FD003" :
                self.args.data_sub = 3
                self.args.patch_size = 2
                self.args.time_denpen_len = 25
                self.args.conv_out = 4
                self.args.num_windows = 36
                self.args.lstmout_dim = 6
                self.args.hidden_dim = 24
                self.args.window_sample = 50
                self.args.conv_time_CNN = 25
                self.args.lstmhidden_dim = 8
                # self.args.patch_size = 5
                # self.args.time_denpen_len = 6
                # self.args.lstmout_dim = 32
                # self.args.conv_time_CNN = 6

            if self.args.Data_id_CMAPSS == "FD004" :
                self.args.data_sub = 4
                self.args.patch_size = 5
                self.args.time_denpen_len = 10
                self.args.conv_out = 7
                self.args.num_windows = 14
                self.args.lstmout_dim = 6
                self.args.hidden_dim = 8
                self.args.window_sample = 50
                self.args.conv_time_CNN = 10
                self.args.lstmhidden_dim = 8
                # self.args.time_denpen_len = 10
                # self.args.lstmout_dim = 6
                # self.args.conv_time_CNN = 10


            model = FC_STGNN_RUL ( self.args.patch_size, self.args.conv_out, self.args.lstmhidden_dim,
                                   self.args.lstmout_dim, self.args.conv_kernel, self.args.hidden_dim,
                                   self.args.conv_time_CNN, self.args.num_sensor, self.args.num_windows,
                                   self.args.moving_window, self.args.stride, self.args.decay, self.args.pool_choice,
                                   1 )

        elif self.args.model_name == 'LeNet' :
            model = LeNet ( self.args.input_length, self.input_feature )

        elif self.args.model_name == 'LSTM' :
            model = LSTM ( self.input_feature )

        elif self.args.model_name == 'AGCNN' :
            if self.args.dataset_name == 'CMAPSS' :
                if self.args.Data_id_CMAPSS == "FD001" :
                    self.args.input_length = 30
                    print(f'{self.args.Data_id_CMAPSS}的input_len默认值是30, 若希望可变请删掉该assert')

                elif self.args.Data_id_CMAPSS == "FD002" :
                    self.args.input_length = 20
                    print(f'{self.args.Data_id_CMAPSS}的input_len默认值是20, 若希望可变请删掉该assert')

                elif self.args.Data_id_CMAPSS == "FD003" :
                    self.args.input_length = 30
                    print(f'{self.args.Data_id_CMAPSS}的input_len默认值是30, 若希望可变请删掉该assert')

                elif self.args.Data_id_CMAPSS == "FD004" :
                    self.args.input_length = 18
                    print(f'{self.args.Data_id_CMAPSS}的input_len默认值是18, 若希望可变请删掉该assert')

            model = AGCNN ( input_len = self.args.input_length, num_features = self.input_feature, m = self.args.m,
                            rnn_hidden_size = self.args.rnn_hidden_size, dropout_rate = self.args.dropout_rate, bidirectional = self.args.bidirectional,
                            fcn_hidden_size = self.args.fcn_hidden_size )

        elif 'AIGCN' == self.args.model_name or 'AIGCN_best' == self.args.model_name:
            self.args.feature_num = self.input_feature
            self.args.outlayer = 'Linear'
            self.args.nlayer = self.args.nlayer
            self.args.fc_dropout = self.args.dropout
            self.args.hop = self.args.hop
            self.args.sequence_len = self.args.input_length  # sequence length (set explicitly)
            self.args.hidden_dim = self.args.d_model
            self.args.feature_fc_layer_dim = self.args.feature_num * 4
            self.args.fc_layer_dim = self.args.hidden_dim * 4
            self.args.kernal_size = self.args.AATE_dim
            self.args.num_sensor = self.input_feature
            model = AIGCN ( self.args )

        elif 'DPDG' == self.args.model_name or 'DPDG_best' == self.args.model_name:
            self.args.feature_num = self.input_feature
            self.args.outlayer = 'Linear'
            self.args.nlayer = self.args.nlayer
            self.args.fc_dropout = self.args.dropout
            self.args.hop = self.args.hop
            self.args.sequence_len = self.args.input_length  # sequence length (set explicitly)
            self.args.hidden_dim = self.args.d_model
            self.args.feature_fc_layer_dim = self.args.feature_num * 4
            self.args.fc_layer_dim = self.args.hidden_dim * 4
            self.args.kernal_size = self.args.AATE_dim
            self.args.num_sensor = self.input_feature
            model = DPDG ( self.args )

        elif self.args.model_name == 'CDSG':
            model = CDSG(self.args, input_feature=self.input_feature)

        elif self.args.model_name == 'SDAGCN':
            model = SDAGCN(self.args, input_feature=self.input_feature)

        print ( "Parameter :", np.sum ( [para.numel () for para in model.parameters ()] ) )

        return model.to ( self.device )

    # --------------------------- select optimizer ------------------------------
    def _select_optimizer(self) :
        if self.args.optimizer not in self.optimizer_dict.keys () :
            raise NotImplementedError

        model_optim = self.optimizer_dict[self.args.optimizer] ( self.model.parameters (),
                                                                 lr = self.args.learning_rate )
        return model_optim

    # ---------------------------- select criterion --------------------------------
    def _select_criterion(self) :
        if self.args.criterion not in self.criterion_dict.keys () :
            raise NotImplementedError

        criterion = self.criterion_dict[self.args.criterion] ()
        return criterion

    # ------------------------ get Dataloader -------------------------------------

    #  funnction of load CMPASS Dataset
    def _get_data(self) :
        args = self.args
        # train and validation dataset
        if self.args.dataset_name == 'CMAPSS' :
            X_train, y_train, index_train, X_vali, y_vali, index_vali, X_test, y_test, index_test, self.global_input = get_cmapss_data_ (
                data_path = args.data_path_CMAPSS, Data_id = args.Data_id_CMAPSS, test_Data_id =args.Data_id_CMAPSS_test, sequence_length = args.input_length,
                MAXLIFE = args.MAXLIFE_CMAPSS, is_difference = args.is_diff, validation = args.validation )
            self.max_life = args.MAXLIFE_CMAPSS

        elif self.args.dataset_name == 'N_CMAPSS' :
            # X_train, I_train, y_train, X_vali, I_vali, y_vali, X_test_valid_data, Y_test_valid_label, max_life.tolist()
            X_train, index_train, y_train, X_vali, index_vali, y_vali, X_test, index_test, y_test, self.max_life = get_n_cmapss_data_ (
                args = self.args,
                name = args.Data_id_N_CMAPSS )  # X_train, y_train, X_vali, y_vali, X_test, y_test, self.max_life = get_n_cmapss_data_(args=self.args, name=args.Data_id_N_CMAPSS)

        elif self.args.dataset_name == 'XJTU' :
            X_train, y_train, X_vali, y_vali, X_test, y_test, self.max_life = get_xjtu_data_ (
                pre_process_type = "Vibration", root_dir = './XJTU/XJTU-SY_Bearing_Datasets/35Hz12kN',
                train_bearing_data_set = ["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
                test_bearing_data_set = ["Bearing1_1"], STFT_window_len = 256, STFT_overlap_num = 32,
                window_length = args.input_length,
                validation_rate = 0.1 )  # class Transpose(nn.Module):  #     def __init__(self):  #         super(Transpose, self).__init__()  #     def forward(self, x):  #         return x.transpose(-1, -2)  #  # self.input_linear = nn.Sequential(nn.Linear(32768, 240, dtype=torch.double),  #                                   nn.ReLU(),  #                                   Transpose(),  #                                   nn.BatchNorm1d(240, dtype=torch.double),  #                                   Transpose()  #                                   ).to(self.device)
        else :
            raise ValueError ( 'without corresponding dataset' )

        train_data_set = eval ( self.args.dataset_name + 'Data' ) ( X_train, index_train, y_train )
        vali_data_set = eval ( self.args.dataset_name + 'Data' ) ( X_vali, index_vali, y_vali )
        test_data_set = eval ( self.args.dataset_name + 'Data' ) ( X_test, index_test, y_test )

        input_fea = X_test.shape[-1]
        # if self.args.dataset_name == 'XJTU':
        #     input_fea = 240

        train_data_loader = DataLoader ( dataset = train_data_set, batch_size = args.batch_size, shuffle = True,
                                         num_workers = 0, drop_last = True )

        vali_data_loader = DataLoader ( dataset = vali_data_set, batch_size = args.batch_size, shuffle = False,
                                        num_workers = 0, drop_last = True )

        test_data_loader = DataLoader ( dataset = test_data_set, batch_size = args.batch_size, shuffle = False,
                                        num_workers = 0, drop_last = False )

        return train_data_set, train_data_loader, vali_data_set, vali_data_loader, test_data_set, test_data_loader, input_fea

        #  funnction of load CMPASS Dataset

    def save_hparam(self) :
        # args: args from argparse return
        value2save = {k : v for k, v in vars ( self.args ).items () if
                      not k.startswith ( '__' ) and not k.endswith ( '__' )}
        # 删除影响使用的参数，以便加载
        del_key = ['train', 'resume', 'save_path', 'resume_path', 'batch_size', 'train_epochs', 'learning_rate']
        for key in del_key :
            del value2save[key]

        with open ( os.path.join ( self.save_path, 'hparam.yaml' ), 'a+' ) as f :
            f.write ( yaml.dump ( value2save ) )

    def _get_path(self) :
        if not os.path.exists ( './logs/' ) :
            os.makedirs ( './logs/' )

        exp_id = self.args.save_path
        # save address
        if self.args.dataset_name == 'CMAPSS' :
            data = self.args.Data_id_CMAPSS
        elif self.args.dataset_name == 'N_CMAPSS' :
            data = self.args.Data_id_N_CMAPSS
        else :
            data = self.args.dataset_name
        self.path = './logs/' + data
        if not os.path.exists ( self.path ) :
            os.makedirs ( self.path )

        self.model_path = self.path + '/' + self.args.model_name
        if not os.path.exists ( self.model_path ) :
            os.makedirs ( self.model_path )

        if exp_id is not None and exp_id != 'None' and exp_id != 'none' :
            self.save_path = self.model_path + '/' + exp_id
            if self.args.train :
                if os.path.exists ( self.save_path ) :
                    shutil.rmtree ( self.save_path )
                os.makedirs ( self.save_path )

        else :
            # 如为None则自动加一
            path_list = os.listdir ( self.model_path )
            if path_list == [] :
                self.save_path = self.model_path + '/exp0'

            else :
                path_list = [int ( idx[3 :] ) for idx in path_list]
                self.save_path = self.model_path + '/exp' + str ( max ( path_list ) + 1 )

            os.makedirs ( self.save_path )

    def _load_checkpoint(self) :
        self.checkpoint_dir = self.model_path + '/' + self.args.resume_path + '/best_checkpoint.pth'
        if os.path.exists ( self.checkpoint_dir ) :
            check_point = torch.load ( self.checkpoint_dir )
            self.model.load_state_dict ( check_point )
        else :
            raise print ( 'checkpoint is not exists' )

        yaml_dir = self.model_path + '/' + self.args.resume_path + '/hparam.yaml'
        with open ( yaml_dir, 'r', encoding = 'utf-8' ) as f :
            yml = yaml.load ( f.read (), Loader = yaml.FullLoader )
            key2load = [k for k in yml.keys ()]
            for k in key2load :
                # 重赋值
                exec ( 'self.args.' + k + '=yml[k]' )
    def add_sensor_noise(self,signal, noise_level=0.05, cutoff_ratio=0.5):
        """
        Adds realistic high-frequency vibration noise to multivariate sensor signals (GPU-compatible).

        Args:
            signal: torch.Tensor, shape (B, T, D)
            noise_level: float, noise std as a fraction of signal std (e.g., 0.05)
            cutoff_ratio: float in (0, 1), high-pass cutoff relative to Nyquist freq (e.g., 0.5 keeps >50% high-freq components)

        Returns:
            noisy_signal: torch.Tensor, shape (B, T, D)
        """
        B, T, D = signal.shape

        # Estimate std per feature (avoid broadcasting issues)
        signal_std = signal.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)

        # Generate Gaussian noise on GPU
        noise = torch.randn_like(signal) * self.noise_std ( signal_std , noise_level )

        # Apply FFT-based high-pass filter
        noise_freq = torch.fft.rfft(noise, dim=1)
        freqs = torch.linspace(0, 1, noise_freq.shape[1], device=signal.device)

        # Create high-pass mask (1 for high-freq, 0 for low-freq)
        mask = (freqs >= cutoff_ratio).float().view(1, -1, 1)
        noise_freq_filtered = noise_freq * mask

        # Inverse FFT to return to time domain
        noise_high = torch.fft.irfft(noise_freq_filtered, n=T, dim=1)

        # Add high-frequency noise to the signal
        noisy_signal = signal + noise_high
        return noisy_signal


    def noise_std(self, signal_std, noise_level):
        """Helper to compute noise std tensor."""
        return noise_level * signal_std

    def start(self) :
        epoch_times = []
        if self.args.train :
            # how many step of train and validation:
            train_steps = len ( self.train_loader )
            vali_steps = len ( self.vali_loader )
            print ( "train_steps: ", train_steps )
            print ( "validaion_steps: ", vali_steps )

            # initial early stopping
            early_stopping = EarlyStopping ( patience = self.args.early_stop_patience, verbose = True )

            # initial learning rate
            learning_rate_adapter = adjust_learning_rate_class ( self.args, True )
            # choose optimizer
            self.model_optim = self._select_optimizer ()

        if self.args.loss_type == 'MSE':
            self.loss_criterion = nn.MSELoss()  # 使用均方误差损失
        elif self.args.loss_type == 'MAE':
            self.loss_criterion = nn.L1Loss()  # 使用平均绝对误差损失
        elif self.args.loss_type == 'QUAN':
            self.loss_criterion = QuantileLoss(quantile=0.3)  # 让模型保守预测

        if self.args.resume :
            print ( 'load checkpoint' )
            self._load_checkpoint ()
        else :
            print ( 'random init' )

        if self.args.train :
            self.save_hparam ()
            # training process
            print ( "start training without manual seed" )
            # seed = self.args.seed
            # torch.manual_seed ( seed )
            # torch.cuda.manual_seed ( seed )
            # torch.cuda.manual_seed_all ( seed )
            # random.seed ( seed )
            # np.random.seed ( seed )

            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

            for epoch in range ( self.args.train_epochs ) :
                # training process
                train_loss, epoch_time = self.training ()
                epoch_times.append(epoch_time)
                # validation process:
                vali_loss = self.validation ( self.vali_loader, self.loss_criterion )

                print ( "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}. it takse {4:.7f} seconds".format (epoch + 1, train_steps, train_loss, vali_loss, epoch_time ) )

                # plt.figure(figsize=(8,5))
                # plt.plot(train_losses, label="Train Loss")
                # plt.plot(val_losses, label="Val Loss")
                # plt.xlabel("Epochs")
                # plt.ylabel("Loss")
                # plt.title("Training vs Validation Loss")
                # plt.legend()
                # plt.grid(True)
                # plt.show()


                # At the end of each epoch, Determine if we need to stop and adjust the learning rate
                early_stopping ( vali_loss, self.model, self.save_path )
                if early_stopping.early_stop :
                    print ( "Early stopping" )
                    break
                learning_rate_adapter ( self.model_optim, vali_loss )

            # 读取最优的参数
            check_point = torch.load ( self.save_path + '/' + 'best_checkpoint.pth' )
            self.model.load_state_dict ( check_point )
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if len(epoch_times) > 0 else 0
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        # test:
        torch.cuda.synchronize()  # Ensure all CUDA ops are complete before timing
        start_time = time()
        average_enc_loss, average_enc_overall_loss, overall_score = self.test ( self.test_loader )
        torch.cuda.synchronize()  # Again wait for all CUDA ops to finish
        end_time = time()
        inference_time = end_time - start_time

        print ( f"{self.args.dataset_name}- test performace using {self.args.model_name} model: RMSE : ", average_enc_loss, 'socre: ', overall_score )
        if self.args.dataset_name == 'CMAPSS' :
            dataset = self.args.Data_id_CMAPSS
            test_dataset = self.args.Data_id_CMAPSS_test
        elif self.args.dataset_name == 'N_CMAPSS' :
            dataset = self.args.Data_id_N_CMAPSS
            test_dataset = self.args.Data_id_N_CMAPSS
        else :
            dataset = self.args.dataset_name + self.args.Data_id_XJTU
            test_dataset = self.args.dataset_name + self.args.Data_id_XJTU

        log_path = './logs/'+dataset+'experimental_logs.csv'
        if not os.path.exists ( log_path ) :
            table_head = [
                ['train_dataset','test_dataset', 'model', 'time','inference_time', 'ablation_mode', 'LR', 'batch_size', 'best_last_RMSE', 'score',
                 'windowsize', 'd_model', 'd_ff', 'dropout', 'kernal_size/b_dim', 'kernal_stride', 'train', 'savepath', 'resume',
                 'resumepath', 'info', 'hope', 'nlayer', 'use_dynamic_enc']]
            write_csv ( log_path, table_head, 'w+' )

        time_now = datetime.datetime.now ().strftime ( '%Y%m%d-%H%M%S' )  # 获取当前系统时间

        resume_dir = self.checkpoint_dir if self.args.resume else None

        if self.args.train :
            a_log = [
            {'train_dataset': dataset,'test_dataset' : test_dataset, 'model' : self.args.model_name, 'time' : time_now, 'inference_time':avg_epoch_time, 'ablation_mode':self.args.ablation_mode, 'LR' : self.args.learning_rate,
             'batch_size' : self.args.batch_size, 'best_last_RMSE' : average_enc_loss, 'score' : overall_score,
             'windowsize' : self.args.input_length, 'd_model' : self.args.d_model, 'd_ff' : self.args.d_ff, 'dropout' : self.args.dropout,
             'kernal_size/b_dim' : self.args.AATE_dim if self.args.model_name in ["AIGCN"] else self.args.kernel_size, 'kernal_stride' : self.args.stride, 'train' : self.args.train,
             'savepath' : self.save_path, 'resume' : self.args.resume, 'resumepath' : resume_dir,
             'info' : self.args.info, 'hope': self.args.hop, 'nlayer': self.args.nlayer}]
            write_csv_dict ( log_path, a_log, 'a+' )

    def training(self) :
        start_time = time ()

        iter_count = 0
        train_loss = []

        self.model.train ()
        for i, (batch_x, idxs_x, batch_y) in enumerate ( tqdm ( self.train_loader ) ) :
            iter_count += 1
            self.model_optim.zero_grad ()

            batch_x = batch_x.to ( self.device )  # [B,window_size,D]
            # if self.args.add_noise :
            #     batch_x= self.add_sensor_noise(batch_x, self.args.noise_level_train )
            # if self.args.dataset_name == 'XJTU':
            #     batch_x = self.input_linear(batch_x)
            batch_y = batch_y.to ( self.device )  # [B,1]

            if self.args.is_minmax :

                # 只预测窗口内的最后一个rul
                batch_y_norm = batch_y / self.max_life
                _, outputs = self.model ( batch_x, idxs_x )
                loss = self.loss_criterion ( outputs, batch_y_norm )

            else :

                _, outputs = self.model ( batch_x, idxs_x )
                loss = self.loss_criterion ( outputs, batch_y )

            train_loss.append ( loss.item () )
            loss.backward ()
            self.model_optim.step ()  # ------------------------------------------------

        end_time = time ()
        epoch_time = end_time - start_time
        train_loss = np.average ( train_loss )  # avgerage loss

        return train_loss, epoch_time

    # ---------------------------------- validation function -----------------------------------------
    def validation(self, vali_loader, criterion) :
        self.model.eval ()
        val_losses = []

        for i, (batch_x, idx_x, batch_y) in enumerate ( vali_loader ) :
            batch_x = batch_x.to ( self.device )
            # if self.args.add_noise :
            #     batch_x= self.add_sensor_noise(batch_x, self.args.noise_level_train )
            # if self.args.dataset_name == 'XJTU':
            #     batch_x = self.input_linear(batch_x)
            batch_y = batch_y.to ( self.device )

            if self.args.is_minmax :
                batch_y_norm = batch_y / self.max_life
                _, outputs = self.model ( batch_x, idx_x )
                loss = self.loss_criterion ( outputs, batch_y_norm )

            else :
                _, outputs = self.model ( batch_x, idx_x )
                loss = self.loss_criterion ( outputs, batch_y )

            val_losses.append ( loss.item () )

        average_vali_loss = np.average ( val_losses )

        self.model.train ()
        return average_vali_loss

    # ----------------------------------- test function ------------------------------------------
    def test(self, test_loader) :
        self.model.eval ()
        enc_pred = []
        gt = []
        for i, (batch_x, idx_x, batch_y) in enumerate ( test_loader ) :
            batch_x = batch_x.to ( self.device )
            if self.args.add_noise :
                batch_x= self.add_sensor_noise(batch_x, self.args.noise_level_test)
            # if self.args.dataset_name == 'XJTU':
            #     batch_x = self.input_linear(batch_x)
            batch_y = batch_y.to ( self.device )
            idx_x = idx_x.to ( self.device )

            if self.args.is_minmax :
                _, outputs = self.model ( batch_x, idx_x )
                outputs = outputs * self.max_life

            else :
                _, outputs = self.model ( batch_x, idx_x )  # outputs[B,window_size]

            batch_y = batch_y.detach ().cpu ().numpy ()
            enc = outputs.detach ().cpu ().numpy ()

            if self.args.dataset_name == 'XJTU' and self.args.is_minmax :
                enc = enc / self.max_life
                batch_y = batch_y / self.max_life

            if not np.isfinite(enc).all():
                print(f"Invalid output detected in batch {i}, replacing with zeros")
                enc = np.nan_to_num(enc, nan=0.0, posinf=200.0, neginf=0.0) #np.nan_to_num(enc)
            gt.append ( batch_y )
            enc_pred.append ( enc )


        gt = np.concatenate ( gt ).reshape ( -1, 1 )
        enc_pred = np.concatenate ( enc_pred ).reshape ( -1, 1 )

        if self.args.save_test :
            result_path = self.save_path + '/result.npz'
            np.savez ( result_path, test_preds = enc_pred, test_trues = gt, )

        # 算的就是RMSE
        # Check for invalid values
        # if np.isnan(enc_pred).any() or np.isnan(gt).any():
        #     print("NaN detected in enc_pred or gt")
        #     enc_pred = np.nan_to_num(enc_pred, nan=200.0, posinf=0.0, neginf=0.0)
        # if np.isinf(enc_pred).any() or np.isinf(gt).any():
        #     print("Inf detected in enc_pred or gt")
        #     gt = np.nan_to_num(gt, nan=0.0, posinf=125.0, neginf=0.0)
        average_enc_loss = np.sqrt ( mean_squared_error ( enc_pred, gt ) )
        average_enc_overall_loss = np.sqrt ( mean_squared_error ( enc_pred, gt ) )
        # 计算score
        overall_score = self.score_compute ( enc_pred, gt )

        return average_enc_loss, average_enc_overall_loss, overall_score

    def score_compute(self, pred, gt) :
        # pred [B] gt[B]
        B = pred.shape
        score = 0
        if self.args.dataset_name == 'XJTU' :
            score_list = np.where ( pred - gt < 0, np.exp ( (gt - pred) * np.log ( 0.5 ) / (gt * 20) ),
                                    np.exp ( -(gt - pred) * np.log ( 0.5 ) / (gt * 5) ) )
        else :
            score_list = np.where ( pred - gt < 0, np.exp ( -(pred - gt) / 13 ) - 1, np.exp ( (pred - gt) / 10 ) - 1 )
        # 这里有的paper求均值，有的求和。实验里面先全都求和计算score
        if self.args.dataset_name == 'CMAPSS' :
            score = np.sum ( score_list )
        else :
            score = np.mean ( score_list )
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
