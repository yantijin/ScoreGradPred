import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, x_shape):
    b = x_shape[0]
    return a.reshape(b, *((1,) * (len(x_shape) - 1)))


class LISDE(nn.Module):
    def __init__(self, net, sigma=1e-6, eps=1e-3):
        super().__init__()
        self.sigma_min = sigma
        self.net = net
        self.eps = eps

    def f_drift_diffusion(self, t):
        drift_coeff = - (1 - self.sigma_min) / (1 - (1 - self.sigma_min) * t)
        diffusion = torch.sqrt(2 * t / (1 - (1 - self.sigma_min) * t))
        return drift_coeff, diffusion

    def marginal_prob(self, x0, t):
        coeff = 1 - (1 - self.sigma_min) * t
        std = t
        mean = extract(coeff, x0.shape) * x0
        return mean, extract(std, x0.shape)

    def get_denoise_par(self, t, *args, **kwargs):
        res = (t,)
        if args:
            res += args
        if kwargs:
            res += tuple(kwargs.values())
        return res

    def get_score(self, xt, t, *args, **kwargs):
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        eps_theta = self.net(xt, *denoise_par)
        _, std = self.marginal_prob(torch.zeros_like(xt), t)
        return - eps_theta / std

    def rev_drift_diffusion(self, xt, t, *args, **kwargs):
        score = self.get_score(xt, t, *args, **kwargs)
        drift_coeff, diffusion = self.f_drift_diffusion(t)
        drift_coeff = extract(drift_coeff, xt.shape)
        diffusion = extract(diffusion, xt.shape)

        rev_drift = drift_coeff * xt - diffusion ** 2 *score
        return rev_drift, diffusion

    def euler_step(self, x, cur_time, delta_t, *args, **kwargs):
        drift, diffusion = self.rev_drift_diffusion(x, cur_time, *args, **kwargs)
        return x - extract(delta_t, x.shape) * drift + diffusion * extract(torch.sqrt(delta_t), x.shape) * torch.randn_like(x)


    def forward(self, x0, *args, **kwargs):
        noise = torch.randn_like(x0)
        t = torch.rand((x0.shape[0],),).to(x0.device)

        denoise_par = self.get_denoise_par(t, *args, **kwargs)

        mean, std = self.marginal_prob(x0, t)
        xt = mean + std * noise

        eps_theta = self.net(xt, *denoise_par)

        loss = F.mse_loss(eps_theta, noise)
        return loss

    def euler_sampling(self, x, num_steps, denoise=True, clamp=True, *args, **kwargs):
        bs = x.shape[0]
        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs, ), device=x.device)

        for i, time_step in enumerate(time_steps):
            s = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                s = cur_time

            x = self.euler_step(x, cur_time, s, *args, **kwargs)
            if clamp:
                x.clamp_(-1., 1.)

            cur_time = cur_time - s

        return x

    def pred_x0(self, xt, t, *args, **kwargs):
        # print(*args, **kwargs)
        denoise_par = self.get_denoise_par(t.squeeze(), *args, **kwargs)
        eps_theta = self.net(xt, *denoise_par)
        x0 = (xt - t * eps_theta) / (1 - (1 - self.sigma_min) * t)
        return x0, eps_theta

    def rev_step(self, xt, t, delta_t, *args, **kwargs):

        t = extract(t, xt.shape)
        delta_t = extract(delta_t, xt.shape)
        pred_x0, eps_theta = self.pred_x0(xt, t, *args, **kwargs)
        rev_mean = xt + delta_t * (1 - self.sigma_min) * pred_x0 - \
                   (t ** 2 - (t - delta_t) ** 2) / t * eps_theta
        rev_std = torch.sqrt((t-delta_t)**2 * (t**2 - (t-delta_t)**2) / t**2)
        noise = torch.randn_like(xt)
        return rev_mean + noise * rev_std

    def rev_sampling(self, x, num_steps, denoise=True, clamp=True, clamp_val=2., *args, **kwargs):
        bs = x.shape[0]
        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs,), device=x.device)

        for i, time_step in enumerate(time_steps):
            s = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                s = cur_time

            x = self.rev_step(x, cur_time, s, *args, **kwargs)
            if clamp:
                x.clamp_(-clamp_val, clamp_val) # adjust

            cur_time = cur_time - s

        return x


    @torch.no_grad()
    def sample(self, shape, num_steps, device, denoise=True, clamp=True, clamp_val=2., *args, **kwargs):
        x = torch.randn(shape).to(device)
        x = self.rev_sampling(x, num_steps, denoise=denoise, clamp=clamp, clamp_val=clamp_val, *args, **kwargs)
        return x