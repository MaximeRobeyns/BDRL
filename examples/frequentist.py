#!/usr/bin/env python3
import sys

import torch as t
import bayesfunc as bf
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from torch.distributions import Gamma

from bdrl.models import Ensemble, BDR_dist

dtype=t.float64
device="cpu"
in_features = 1
out_features = 1
train_batch = 40
batches = 3
data_size = batches * train_batch

t.manual_seed(42)

def generate_data():
    noise = Gamma(2, 0.1)
    X = t.rand(data_size, in_features) * 4 - 2
    x_1 = X[:int(data_size/2), :]
    x_2 = X[int(data_size/2):, :]

    y = t.cat((
        x_1**3 + 8 + 1   * noise.sample((int(data_size/2), in_features)),
        x_2**3 - 8 - 0.1 * noise.sample((int(data_size/2), in_features))
    ))
    # y = t.cat((
    #     x_1**3 + 8 + 0.1 * noise.sample((int(data_size/2), in_features)),
    #     x_2**3 - 8 - 0.1 * noise.sample((int(data_size/2), in_features))
    # ))

    scale = y.std()
    y = y/scale

    xys = t.cat((X,y),1)
    xys = xys[t.randperm(xys.size()[0])]
    X = xys[:,0].unsqueeze(1).to(device=device, dtype=dtype)
    y = xys[:,1].unsqueeze(1).to(device=device, dtype=dtype)
    return X, y, scale

def plot_data():
    X, y, scale = generate_data()
    plt.scatter(X.detach().cpu(), y.detach().cpu())
    plt.show()

def train_net(X, y):
    samples = 20
    net = Ensemble(
        in_features=X.shape[1],
        N=11,
        E=samples,
        layer_sizes=(50,),
        dtype=X.dtype,
        device=device,
        f_postproc='sort'
    )
    net.train(X, y, epochs=100, batch_size=32)

    return net

def save_params(net, x_loc):
    with t.no_grad():
        # num points on graph
        num_pts = 100
        # samples from posterior
        samples = 50

        xs = t.tensor([x_loc]).unsqueeze(1)
        xs = xs.expand(samples, -1, -1).to(dtype=dtype, device=device)

        theta = net.f(xs)
        t.save(theta, "theta.pt")

def plot_cdf(net, X, y):
    with t.no_grad():
        # num points on graph
        num_pts = 100
        # samples from posterior
        samples = 50

        x_loc = X.mean(0)
        xs = t.tensor([x_loc]).unsqueeze(1).to(dtype=dtype, device=device)
        theta = net.f(xs)
        dist = BDR_dist(*theta)

        # -5 to 5 is arbitrary... Seems to work well with the given data.
        ys = t.linspace(y.min(), y.max(), num_pts).to(dtype=dtype, device=device)

        cdfs = dist.cdf(ys, avg=True).detach().cpu()
        c_mean = cdfs.mean(0).squeeze()
        c_std  = cdfs.std(0).squeeze()

        fig, (ax1) = plt.subplots(1, 1, figsize=(10,6))

        ax1.plot(ys.detach().cpu(), c_mean)
        ax1.fill_between(ys.detach().cpu(), (c_mean+c_std), (c_mean-c_std), alpha=0.2)
        ax1.set_title(f"Distribution at x={x_loc}")
        ax1.set_ylabel("Density")
        ax1.set_xlabel("Y value")
        plt.show()

def plot_icdf(net, x_loc):
    with t.no_grad():
        num_pts = 100
        samples = 70
        xs = t.tensor([x_loc, x_loc]).unsqueeze(1).unsqueeze(0).to(dtype=dtype, device=device)
        theta = net.f(xs)

        dist = BDR_dist(*theta, fp=True)

        p_vals = t.linspace(0.005, 0.995, num_pts).to(dtype=dtype, device=device)

        icdfs = dist.icdf(p_vals, avg=True).detach().cpu()[0]

        ic_mean = icdfs.mean(0).squeeze()
        ic_std  = icdfs.std(0).squeeze()

        fig, (ax1) = plt.subplots(1, 1, figsize=(10,6))

        ax1.plot(p_vals, ic_mean)
        ax1.fill_between(p_vals, (ic_mean+ic_std), (ic_mean-ic_std), alpha=0.2)
        ax1.set_title(f"Distribution at x={x_loc}")
        ax1.set_ylabel("Return")
        ax1.set_xlabel("P")
        plt.show()

def sample_test(net, x_loc, true_ys):
    with t.no_grad():
        num_pts = 300
        samples = 100

        xs = t.tensor([x_loc]).unsqueeze(1).to(dtype=dtype, device=device)

        ys = t.linspace(-2, 5, num_pts).to(dtype=dtype, device=device)

        theta = net.f(xs)
        dist = BDR_dist(*theta, fp=True)

        ss = dist.log_prob(ys).exp().detach().cpu()
        ss_mean = ss.mean(0)
        ss_std  = ss.std(0)

        mean = dist.mean().detach().cpu()
        print("mean shape: ", mean.shape)

        fig, (ax1) = plt.subplots(1, 1, figsize=(10,6))

        (f, alpha, beta) = theta
        f_avg = f.mean(0)

        modes = dist.mode_at(0, avg=True).detach().cpu()
        ax1.vlines(modes, 0, 1, color='r', linewidths=2)

        modes = dist.modes()[0].detach().cpu()
        ax1.vlines(modes, 0, 1, color='r', linewidths=0.5, alpha=0.2)

        ax1.vlines(mean, 0, 1, color='b', linewidths=2)

        samples = dist.sample(1000, avg=True).detach().cpu()
        ax1.hist(samples, bins=100, density=True, color='grey', alpha=0.2)

        true_ys = true_ys.detach().cpu()
        ax1.scatter(true_ys, t.zeros_like(true_ys))

        ax1.plot(ys.detach().cpu(), ss_mean[0])
        ax1.fill_between(ys.detach().cpu(), (ss_mean+ss_std)[0], (ss_mean-ss_std)[0], alpha=0.2)
        ax1.set_title(f"Distribution at x={x_loc}")
        ax1.set_ylabel("Density")
        ax1.set_xlabel("Y value")
        plt.show()


def main():
    X, y, scale = generate_data()
    # plot_data()
    net = train_net(X, y)
    # save_params(net, X.mean(0))
    plot_cdf(net, X.mean(0), y)
    # plot_icdf(net, X, y)
    sample_test(net, X.mean(0), y)

if __name__ == '__main__':
    main()
