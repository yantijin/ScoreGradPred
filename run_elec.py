import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from configs.config import get_configs
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from score_sde.score_sde_estimator import ScoreGradEstimator
# from pts import Trainer
from score_sde.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from utils import *
from score_sde.util import seed_torch


def plot(target, forecast, prediction_length, prediction_intervals=(50.0, 90.0), color='g', fname=None):
    label_prefix = ""
    rows = 4
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
    axx = axs.ravel()
    seq_len, target_dim = target.shape

    ps = [50.0] + [
        50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
    ]

    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3

    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        target[-2 * prediction_length:][dim].plot(ax=ax)

        ps_data = [forecast.quantile(p / 100.0)[:, dim] for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2

        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=forecast.index)
        p50_series.plot(color=color, ls="-", label=f"{label_prefix}median", ax=ax)

        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)
            ax.fill_between(
                forecast.index,
                ps_data[i],
                ps_data[-i - 1],
                facecolor=color,
                alpha=alpha,
                interpolate=True,
            )
            # Hack to create labels for the error intervals.
            # Doesn't actually plot anything, because we only pass a single data point
            pd.Series(data=p50_data[:1], index=forecast.index[:1]).plot(
                color=color,
                alpha=alpha,
                linewidth=10,
                label=f"{label_prefix}{100 - ptile * 2}%",
                ax=ax,
            )

    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    axx[0].legend(legend, loc="upper left")

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Available datasets: {list(dataset_recipes.keys())}")

dataset = get_dataset("electricity_nips", regenerate=False)
# print(dataset.metadata)


train_grouper = MultivariateGrouper(max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)),
                                   max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))
args = parse_args()
config = get_configs(dataset=args.data, name=args.name)
seed_torch(config.training.seed)

dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)

estimator = ScoreGradEstimator(
    input_size=config.input_size,
    freq=dataset.metadata.freq,
    prediction_length=dataset.metadata.prediction_length,
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    context_length=dataset.metadata.prediction_length,
    num_layers=config.num_layers,
    num_cells=config.num_cells,
    cell_type='GRU',
    num_parallel_samples=config.num_parallel_samples,
    dropout_rate=config.dropout_rate,
    conditioning_length=config.conditioning_length,
    diff_steps=config.modeling.num_scales,
    beta_min=config.modeling.beta_min,
    beta_end=config.modeling.beta_max,
    residual_layers=config.modeling.residual_layers,
    residual_channels=config.modeling.residual_channels,
    dilation_cycle_length=config.modeling.dilation_cycle_length,
    scaling=config.modeling.scaling,
    md_type=config.modeling.md_type,
    continuous=config.training.continuous,
    reduce_mean=config.reduce_mean,
    likelihood_weighting=config.likelihood_weighting,
    config=config,
    trainer=Trainer(
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches_per_epoch,
        learning_rate=config.learning_rate,
        decay=config.weight_decay,
        device=config.device,
        wandb_mode='disabled',
        config=config)
)

if config.train:
    train_output = estimator.train_model(dataset_train, num_workers=0)
    predictor = train_output.predictor

else:
    assert config.path is not None
    trainnet = estimator.create_training_network(config.device)
    trainnet.load_state_dict(torch.load(config.path))
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, trainnet, config.device)



forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                                 predictor=predictor,
                                                 num_samples=100)


forecasts = list(forecast_it)
targets = list(ts_it)

plot(
    target=targets[0],
    forecast=forecasts[0],
    prediction_length=dataset.metadata.prediction_length,
)



evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})

agg_metric, item_metrics = evaluator(targets, forecasts, num_series=len(dataset_test))

print("CRPS:", agg_metric["mean_wQuantileLoss"])
print("ND:", agg_metric["ND"])
print("NRMSE:", agg_metric["NRMSE"])
print("")
print("CRPS-Sum:", agg_metric["m_sum_mean_wQuantileLoss"])
print("ND-Sum:", agg_metric["m_sum_ND"])
print("NRMSE-Sum:", agg_metric["m_sum_NRMSE"])
metrics = {
    'CRPS': agg_metric["mean_wQuantileLoss"],
    "ND": agg_metric["ND"],
    "NRMSE": agg_metric["NRMSE"],
    "CRPS-Sum:": agg_metric["m_sum_mean_wQuantileLoss"],
    "ND-Sum:": agg_metric["m_sum_ND"],
    "NRMSE-Sum:": agg_metric["m_sum_NRMSE"],
}
if config.save and config.train:
    torch.save(train_output.trained_net.state_dict(), config.path[:-4] + str(metrics['CRPS-Sum:']) + config.path[-4:])
write_to_file(args, config, metrics, args.path)