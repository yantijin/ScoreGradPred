from configs.elec.elec_config import get_default_configs



def get_elec_smld_cont_config():
    config = get_default_configs()

    config.weight_decay = 0.999
    config.reduce_mean = True
    config.likelihood_weighting = False
    config.batch_size = 64
    config.epochs = 30

    modeling = config.modeling
    modeling.num_scales = 200
    modeling.beta_min = 0.01
    modeling.beta_max = 10
    modeling.md_type = 'vesde'

    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    training = config.training
    training.continuous = True
    training.seed = 123

    config.train = True
    config.save = True
    config.path = './model/elec_smld_c.pkl'

    return config