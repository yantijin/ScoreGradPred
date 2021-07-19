from configs.elec.elec_config import get_default_configs



def get_elec_ddpm_discrete_config():
    config = get_default_configs()

    config.weight_decay = None
    config.reduce_mean = True
    config.likelihood_weighting = False
    config.batch_size = 64
    config.epochs = 20

    modeling = config.modeling
    modeling.num_scales = 100
    modeling.beta_min = 0.01
    modeling.beta_max = 10
    modeling.md_type = 'vpsde'

    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'ancestral_sampling'
    sampling.corrector = 'none'

    training = config.training
    training.continuous = False
    training.seed = 123

    config.train = True
    config.save = True
    config.path = './model/ddpm_d.pkl'

    return config