import ml_collections
import torch

def get_default_configs():
    config = ml_collections.ConfigDict()

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.noise_removal = False # 在最后加一步denoising 步骤，在NCSNV2中证明有效
    sampling.snr=0.16 # ratio in annealed Langivan sampling
    sampling.n_steps_each = 1
    sampling.probability_flow = False


    config.training = ml_collections.ConfigDict()


    config.modeling = modeling = ml_collections.ConfigDict()

    # modeling.sigma_min = 0.01
    # modeling.sigma_max = 50
    modeling.residual_layers = 8
    modeling.residual_channels = 8
    modeling.dilation_cycle_length = 2
    modeling.scaling = True


    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.input_size = 1484
    config.learning_rate = 1e-3
    config.num_layers = 2
    config.num_cells = 40
    config.num_parallel_samples = 100
    config.dropout_rate = 0.1
    config.conditioning_length = 100
    config.num_batches_per_epoch = 100

    return config