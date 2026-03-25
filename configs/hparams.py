## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.

def get_configs(dataset , dataset_id) :
    hparams_class = get_hparams_class ( dataset )
    hparams = hparams_class ( dataset_id )
    return hparams


def get_hparams_class(dataset_name) :
    """Return the algorithm class with the given name."""
    if dataset_name not in globals ( ) :
        raise NotImplementedError ( "Dataset not found: {}".format ( dataset_name ) )
    return globals ( )[dataset_name]


class CMAPSS ( ) :
    def __init__(self , dataset_id) :
        super ( CMAPSS , self ).__init__ ( )

        if dataset_id == 'FD001' :
            self.train_params = {
                'LeNet': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'LSTM': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Transformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Autoformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'PatchTST': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                'PINN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'CDSG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-3},
                'SDAGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-3},
                'Dual_Mixer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-3},
                'Transformer_domain': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'AIGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                'AGCNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'FCSTGNN': {'num_epochs': 41, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'AIGCN_best': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-3},
                'DPDG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-3}
            }
            self.alg_hparams = {
                'LeNet':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'LSTM':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Transformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Autoformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'PatchTST':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'PINN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'CDSG':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'SDAGCN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 32},
                'Dual_Mixer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Transformer_domain':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'AIGCN': {'input_length': 50, 'AATE_dim': 10, 'd_model': 64},

                'AGCNN': {'input_length': 30, 'm': 15, 'rnn_hidden_size': [18, 20], 'dropout_rate': 0.2, 'bidirectional': True, 'fcn_hidden_size': [20, 10]},
                'FCSTGNN':  {'input_length': 30,'time_denpen_len': 6, 'lstmout_dim': 32, 'conv_time_CNN': 10}, #hyper-pram are updated in Experiment.py with it's oringal values
                'AIGCN_best': {'input_length': 30, 'AATE_dim': 10, 'd_model': 64},
                'DPDG': {'input_length':50, 'AATE_dim':10, 'use_dynamic_encoder': True, 'kappa':0.5, 'alpha': 0.15, 'M': 1, 'd_model':64}
            }


        elif dataset_id == 'FD002' :
            self.train_params = {
                'LeNet': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'LSTM': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Transformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Autoformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'PatchTST': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                'PINN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'CDSG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'SDAGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Dual_Mixer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-2},
                'Transformer_domain': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'AIGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                'AGCNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'FCSTGNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'AIGCN_best': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'DPDG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3}
            }
            self.alg_hparams = {
                'LeNet':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'LSTM':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Transformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Autoformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'PatchTST':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'PINN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'CDSG':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'SDAGCN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Dual_Mixer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Transformer_domain':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'AIGCN': {'input_length': 50, 'AATE_dim': 10, 'd_model': 64},

                'AGCNN': {'input_length': 20, 'm': 15, 'rnn_hidden_size': [18, 20], 'dropout_rate': 0.2, 'bidirectional': True, 'fcn_hidden_size': [20, 10]},
                'FCSTGNN':  {'time_denpen_len': 10, 'lstmout_dim': 12, 'conv_time_CNN': 10},
                'AIGCN_best': {'input_length': 50, 'AATE_dim': 10, 'd_model': 128},
                'DPDG': {'input_length':50, 'AATE_dim':10, 'use_dynamic_encoder': True, 'kappa':0.15, 'alpha': 0.1, 'M': 1, 'd_model':64}
            }
        elif dataset_id == 'FD003':
            self.train_params = {
                'LeNet': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'LSTM': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Transformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Autoformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'PatchTST': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                'PINN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'CDSG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-4},
                'SDAGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Dual_Mixer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Transformer_domain': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'AIGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                'AGCNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-4, 'seed':42},
                'FCSTGNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'seed':0},
                'AIGCN_best': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'DPDG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3}
            }
            self.alg_hparams = {
                'LeNet':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'LSTM':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Transformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Autoformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'PatchTST':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'PINN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'CDSG':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'SDAGCN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Dual_Mixer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Transformer_domain':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'AIGCN': {'input_length': 50, 'AATE_dim': 10, 'd_model': 64},

                'AGCNN': {'input_length': 30, 'm': 15, 'rnn_hidden_size': [18, 20], 'dropout_rate': 0.2, 'bidirectional': True, 'fcn_hidden_size': [20, 10]},
                'FCSTGNN':  {'time_denpen_len': 6, 'lstmout_dim': 32, 'conv_time_CNN': 6},
                'AIGCN_best': {'input_length': 60, 'AATE_dim': 10, 'd_model': 64},
                'DPDG': {'input_length':50, 'AATE_dim':10, 'use_dynamic_encoder': True, 'kappa':0.1, 'alpha': 0.05, 'M': 2, 'd_model':64}
            }
        elif dataset_id == 'FD004' :
            self.train_params = {
                'LeNet': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'LSTM': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Transformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Autoformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'PatchTST': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                'PINN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'CDSG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'SDAGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Dual_Mixer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'Transformer_domain': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'AIGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                'AGCNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'FCSTGNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'AIGCN_best': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3}
            }
            self.alg_hparams = {
                'LeNet':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'LSTM':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Transformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Autoformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'PatchTST':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'PINN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'CDSG':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'SDAGCN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Dual_Mixer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'Transformer_domain':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 64},
                'AIGCN': {'input_length': 50, 'AATE_dim': 10, 'd_model': 64},

                'AGCNN': {'input_length': 18, 'm': 15, 'rnn_hidden_size': [18, 20], 'dropout_rate': 0.2, 'bidirectional': True, 'fcn_hidden_size': [20, 10]},
                'FCSTGNN':  {'time_denpen_len': 10, 'lstmout_dim': 6, 'conv_time_CNN': 10},
                'AIGCN_best': {'input_length': 50, 'AATE_dim': 10, 'd_model': 128}
            }
        else :
            raise ValueError ( 'No input dataset id for CMAPSS' )


class N_CMAPSS ( ) :
    def __init__(self , dataset_id=None) :
        super ( N_CMAPSS , self ).__init__ ( )
        if dataset_id == 'DS01' :
            self.train_params = {
                    'LeNet': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'LSTM': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Transformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Autoformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'PatchTST': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                    'PINN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                    'AGCNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'FCSTGNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'CDSG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'SDAGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Dual_Mixer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Transformer_domain': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'AIGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-3},
            }

            self.alg_hparams = {
                    'LeNet':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'LSTM':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Transformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Autoformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'PatchTST':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},

                    'PINN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},

                    'AGCNN': {'input_length': 18, 'm': 15, 'rnn_hidden_size': [18, 20], 'dropout_rate': 0.2, 'bidirectional': True, 'fcn_hidden_size': [20, 10]},
                    'CDSG':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'SDAGCN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Dual_Mixer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Transformer_domain':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'FCSTGNN':  {'time_denpen_len': 10, 'lstmout_dim': 6, 'conv_time_CNN': 10},
                    'AIGCN': {'input_length': 50, 'AATE_dim': 16, 'd_model': 128}
            }
        if dataset_id == 'DS02' :
            self.train_params = {
                    'LeNet': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'LSTM': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Transformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Autoformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-3},
                    'PatchTST': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                    'PINN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                    'AGCNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'FCSTGNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'CDSG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'SDAGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Dual_Mixer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-3},
                    'Transformer_domain': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'AIGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 5e-4},
            }

            self.alg_hparams = {
                    'LeNet':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'LSTM':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Transformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Autoformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'PatchTST':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},

                    'PINN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},

                    'AGCNN': {'input_length': 18, 'm': 15, 'rnn_hidden_size': [18, 20], 'dropout_rate': 0.2, 'bidirectional': True, 'fcn_hidden_size': [20, 10]},
                    'CDSG':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'SDAGCN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Dual_Mixer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Transformer_domain':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'FCSTGNN':  {'time_denpen_len': 10, 'lstmout_dim': 6, 'conv_time_CNN': 10},
                    'AIGCN': {'input_length': 50, 'AATE_dim': 16, 'd_model': 128}
            }
        else:
            self.train_params = {
                    'LeNet': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'LSTM': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Transformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Autoformer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'PatchTST': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                    'PINN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

                    'AGCNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'FCSTGNN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'CDSG': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'SDAGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Dual_Mixer': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'Transformer_domain': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                    'AIGCN': {'num_epochs': 300, 'batch_size': 128, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
            }

            self.alg_hparams = {
                    'LeNet':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'LSTM':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Transformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Autoformer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'PatchTST':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},

                    'PINN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},

                    'AGCNN': {'input_length': 18, 'm': 15, 'rnn_hidden_size': [18, 20], 'dropout_rate': 0.2, 'bidirectional': True, 'fcn_hidden_size': [20, 10]},
                    'CDSG':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'SDAGCN':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Dual_Mixer':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'Transformer_domain':  {'input_length': 50, 'dropout': 0.1, 'validation': 0.1,  'd_model': 128},
                    'FCSTGNN':  {'time_denpen_len': 10, 'lstmout_dim': 6, 'conv_time_CNN': 10},
                    'AIGCN': {'input_length': 50, 'AATE_dim': 16, 'd_model': 128}
            }
