from configs.elec.elec_config import get_default_configs



def get_elec_subvpsde_config():
    config = get_default_configs()

    config.weight_decay = None
    config.reduce_mean = True
    config.likelihood_weighting = False
    config.batch_size = 64
    config.epochs = 50

    modeling = config.modeling
    modeling.num_scales = 100
    modeling.beta_min = 0.1
    modeling.beta_max = 15
    modeling.md_type = 'subvpsde'

    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    training = config.training
    training.continuous = True # subvpsde应该是没有离散形式

    config.train = False
    config.save = True
    config.path = './model/elec_subvpsde.pkl'

    return config