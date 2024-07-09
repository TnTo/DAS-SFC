# %%
from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.random_forest import RandomForestSampler

import numpy

from SFC import SFC, v, print_bs, print_fof, print_v

import warnings

numpy.seterr(all="warn")

# %%
T = 100

bounds = [
    [0.2, 0.0, 1.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.05, 0.05, 0.5, 0.01],
    [1.0, 0.5, 10.0, 1.0, 10.0, 0.5, 0.2, 0.1, 0.1, 0.5, 0.5, 2.0, 0.5],
]

bounds_step = [0.01] * 13

# Best parameters found: [0.99 0.02 2.6  0.21 5.56 0.44 0.03 0.08 0.02 0.09 0.12 0.62 0.43]


batch_size = 8
halton_sampler = HaltonSampler(batch_size=batch_size)
random_forest_sampler = RandomForestSampler(batch_size=batch_size)
best_batch_sampler = BestBatchSampler(batch_size=batch_size)

loss = MethodOfMomentsLoss()
calibration_seed = 8686

target = numpy.zeros((11, T - 20))
target[:, :] = numpy.array(
    [
        [
            0.6,  # APC Income
            0.11,  # APC Wealth
            0.01,  # growth rate
            0.01,  # inflation rate
            0.35,  # profit share
            0.25,  # Public debt interest rate
            1.4,  # Debt to GDP ratio
            0.25,  # Loans' interest rate
            0.42,  # Tax to GDP ratio
            0.1,  # unemployment rate
            0.5,  # Gvt spending to GDP ratio
        ]
    ]
).T


# %%
def model(params, n, seed):
    par, s, i, bs, fof = SFC(
        T=T,
        ay=params[0],
        av=params[1],
        k=params[2],
        phi=params[3],
        aW=params[4],
        tauW=params[5],
        rF=params[6],
        a4=params[7],
        a5=params[8],
        muC=params[9],
        muK=params[10],
        b=params[11],
        rQ=params[12],
    )

    res = numpy.zeros((11, T - 20))
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            res[0, :] = s[v.C, 21:] / (s[v.W, 21:] + s[v.M, 21:])
            res[1, :] = s[v.C, 21:] / (s[v.D, 21:] + s[v.S, 21:])
            res[2, :] = s[v.g, 21:]
            res[3, :] = s[v.psi, 21:]
            res[4, :] = s[v.rS, 21:] * s[v.S, 21:] / s[v.Y, 21:]
            res[5, :] = s[v.rB, 21:]
            res[6, :] = s[v.BG, 21:] / s[v.Y, 21:]
            res[7, :] = s[v.rL, 21:]
            res[8, :] = s[v.T, 21:] / s[v.Y, 21:]
            res[9, :] = s[v.omega, 21:]
            res[10, :] = (s[v.G, 21:] + s[v.M, 21:]) / s[v.Y, 21:]
        except Warning:
            res = numpy.ones((11, T - 20)) * 10

    return res


# %%
try:
    cal = Calibrator.restore_from_checkpoint("calibration", model=model)
    print("Loaded")
except FileNotFoundError:
    cal = Calibrator(
        samplers=[halton_sampler, random_forest_sampler, best_batch_sampler],
        real_data=target,
        model=model,
        parameters_bounds=bounds,
        parameters_precision=bounds_step,
        ensemble_size=1,
        loss_function=loss,
        random_state=calibration_seed,
        saving_folder="calibration",
    )
    print("Initialized")

# %%
params, losses = cal.calibrate(n_batches=1)

# %%
print(f"Best parameters found: {params[0]}")
par, s, i, bs, fof = SFC(
    T=T,
    ay=params[0][0],
    av=params[0][1],
    k=params[0][2],
    phi=params[0][3],
    aW=params[0][4],
    tauW=params[0][5],
    rF=params[0][6],
    a4=params[0][7],
    a5=params[0][8],
    muC=params[0][9],
    muK=params[0][10],
    b=params[0][11],
    rQ=params[0][12],
)
print_bs(bs, -1)
print_fof(fof, -1)
print_v(s, -1)

# %%
