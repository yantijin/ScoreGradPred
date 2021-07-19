from configs.elec.elec_config import get_default_configs



def get_elec_smld_discrete_config():
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
    modeling.md_type = 'vesde'

    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    training = config.training
    training.continuous = False

    config.train = True
    config.save = True
    config.path = './model/elec_smld_d.pkl'

    return config