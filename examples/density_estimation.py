#!/usr/bin/env python3

import torch as t
import bayesfunc as bf
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from torch.distributions import Gamma

from bdrl.models import BDR, BDR_dist

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
        x_1**3 + 8 + 1 * noise.sample((int(data_size/2), in_features)),
        x_2**3 - 8 - 0.1 * noise.sample((int(data_size/2), in_features))
    ))
    y = t.cat((
        x_1**3 + 8 + 0.1 * noise.sample((int(data_size/2), in_features)),
        x_2**3 - 8 - 0.1 * noise.sample((int(data_size/2), in_features))
    ))

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
    net = BDR(in_features=X.shape[1],
              inducing_batch=40,
              N=7,
              layer_sizes=(40,),
              dtype=X.dtype,
              f_postproc='sort'
              )
    net.to(dtype=dtype, device=device)

    opt = t.optim.Adam(net.parameters(), lr=0.05)

    for _ in tqdm(range(100)):
        for batch in range(batches):
            l = batch * train_batch
            u = l     + train_batch
            batch_X = X[l:u].expand(samples, -1, -1)
            # no need to batch_y expand manually;
            # log_prob automatically broadcasts across the samples.
            batch_y = y[l:u]
            opt.zero_grad()
            theta, logpq, _ = bf.propagate(net, batch_X)
            ll = BDR_dist(*theta).log_prob(batch_y).sum(-1).mean(-1)
            assert ll.shape == (samples,)
            assert logpq.shape == (samples,)
            elbo = ll + logpq/data_size
            (-elbo.mean()).backward()
            opt.step()
    return net

def save_params(net, x_loc):
    with t.no_grad():
        # num points on graph
        num_pts = 100
        # samples from posterior
        samples = 50

        xs = t.tensor([x_loc]).unsqueeze(1)
        xs = xs.expand(samples, -1, -1).to(dtype=dtype, device=device)

        theta, _, _ = bf.propagate(net, xs)
        t.save(theta, "theta.pt")

def plot_cdf(net, x_loc):
    with t.no_grad():
        # num points on graph
        num_pts = 100
        # samples from posterior
        samples = 50

        xs = t.tensor([x_loc]).unsqueeze(1)
        xs = xs.expand(samples, -1, -1).to(dtype=dtype, device=device)

        theta, _, _ = bf.propagate(net, xs)

        dist = BDR_dist(*theta)

        # -5 to 5 is arbitrary... Seems to work well with the given data.
        ys = t.linspace(-2, 5, num_pts).to(dtype=dtype, device=device)

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
        xs = t.tensor([x_loc, x_loc]).unsqueeze(1)
        xs = xs.expand(samples, -1, -1).to(dtype=dtype, device=device)

        theta, _, _ = bf.propagate(net, xs)

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
        num_pts = 100
        samples = 100

        # We are only evaluating this at 1 x location
        # Think of this as a single (state, action) pair
        xs = t.tensor([x_loc]).unsqueeze(1)
        # Must expand to predict <samples> different parameter values for each x
        # location in the batch
        xs = xs.expand(samples, -1, -1).to(dtype=dtype, device=device)

        # In order to plot this, we will evaluate the density at the single x
        # location num_pts time:
        # ys = t.linspace(true_ys.min(), true_ys.max(), num_pts).to(dtype=dtype, device=device)
        ys = t.linspace(-5, 5, num_pts).to(dtype=dtype, device=device)

        # Generate the <samples> parameter predictions at this single x or
        # (s,a) location
        theta, _, _ = bf.propagate(net, xs)
        dist = BDR_dist(*theta, fp=True)

        # Evaluate the density of each of the <num_pts> plotting points
        ss = dist.log_prob(ys).exp().detach().cpu()
        # ss has shape [samples, batch, num_pts]
        ss_mean = ss.mean(0)
        ss_std  = ss.std(0)

        mean = dist.mean()
        print("mean shape: ", mean.shape)

        fig, (ax1) = plt.subplots(1, 1, figsize=(10,6))

        (f, alpha, beta) = theta
        # f_avg is [1], because we only predicted params at 1 x location (x_loc)
        f_avg = f.mean(0)
        # ax1.vlines(f_avg, 0, 1)

        # dist is a batch of distributions, with only 1 batch!
        modes = dist.mode_at(0, avg=True)
        ax1.vlines(modes, 0, 1, color='r', linewidths=2)

        # plot modes samples
        modes = dist.modes()[0]
        ax1.vlines(modes, 0, 1, color='r', linewidths=0.5, alpha=0.2)

        ax1.vlines(mean, 0, 1, color='b', linewidths=2)

        samples = dist.sample(1000)
        ax1.hist(samples, bins=100, density=True, color='grey', alpha=0.2)

        ax1.scatter(true_ys, t.zeros_like(true_ys))

        ax1.plot(ys.detach().cpu(), ss_mean[0])
        ax1.fill_between(ys.detach().cpu(), (ss_mean+ss_std)[0], (ss_mean-ss_std)[0], alpha=0.2)
        ax1.set_title(f"Distribution at x={x_loc}")
        ax1.set_ylabel("Density")
        ax1.set_xlabel("Y value")
        plt.show()


def main():
    X, y, scale = generate_data()
    plot_data()
    net = train_net(X, y)
    # save_params(net, X.mean(0))
    # plot_cdf(net, X.mean(0))
    # plot_icdf(net, X.mean(0))
    sample_test(net, X.mean(0), y)

if __name__ == '__main__':
    main()
