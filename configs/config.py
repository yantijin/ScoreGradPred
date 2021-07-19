from .elec import *

def get_configs(dataset, name):
    elec_config_dict = {
        'ddpm_d': get_elec_ddpm_discrete_config(),
        'ddpm_c': get_elec_ddpm_cont_config(),
        'smld_d': get_elec_smld_discrete_config(),
        'smld_c': get_elec_smld_cont_config(),
        'subvpsde': get_elec_subvpsde_config()
    }

    config_dict = {
        'elec': elec_config_dict,
    }

    return config_dict[dataset][name]