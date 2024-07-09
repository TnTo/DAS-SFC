from enum import IntEnum
import numpy
from tabulate import tabulate
import matplotlib.pyplot as plt

### Enums
p = IntEnum(
    "Parameters",
    [
        "rhoC",
        "gamma",
        "uT",
        "ay",
        "av",
        "delta",
        "k",
        "phi",
        "omegaT",
        "aW",
        "tauW",
        "tauC",
        "tauS",
        "rF",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "GammaT",
        "psiT",
        "muC",
        "muK",
        "b",
        "rQ",
    ],
    start=0,
)
v = IntEnum(
    "Variables",
    [
        "DH",
        "DFC",
        "DFK",
        "D",
        "S",
        "LFC",
        "LFK",
        "L",
        "BB",
        "BG",
        "KFC",
        "KFK",
        "C",
        "G",
        "I",
        "W",
        "WFC",
        "WFK",
        "M",
        "T",
        "PiFC",
        "PiFK",
        "Pi",
        "rS",
        "rL",
        "rB",
        "pC",
        "pK",
        "beta",
        "omega",
        "nC",
        "nK",
        "nQ",
        "nCT",
        "nKT",
        "Y",
        "psi",
        "Gamma",
        "CT",
        "GT",
        "IT",
        "DKT",
        "DK",
        "g",
        "u",
    ],
    start=0,
)


def SFC(
    T=100,
    N_ITER=1000,
    rhoC=2.0,
    gamma=0.1,
    uT=0.8,
    ay=0.6,
    av=0.2,
    delta=0.03,
    k=2.0,
    phi=0.3,
    aW=2.0,
    tauW=0.35,
    tauC=0.2,
    tauS=0.2,
    rF=0.05,
    a1=0.5,
    a2=0.25,
    a3=0.25,
    a4=0.01,
    a5=0.01,
    GammaT=0.08,
    psiT=0.02,
    muC=0.2,
    muK=0.2,
    b=1.0,
    rQ=0.1,
):

    par = [0] * len(p)
    par[p.rhoC] = rhoC
    par[p.gamma] = gamma
    par[p.uT] = uT
    par[p.ay] = ay
    par[p.av] = av
    par[p.delta] = delta
    par[p.k] = k
    par[p.phi] = phi
    par[p.aW] = aW
    par[p.tauW] = tauW
    par[p.tauC] = tauC
    par[p.tauS] = tauS
    par[p.rF] = rF
    par[p.a1] = a1
    par[p.a2] = a2
    par[p.a3] = a3
    par[p.a4] = a4
    par[p.a5] = a5
    par[p.GammaT] = GammaT
    par[p.psiT] = psiT
    par[p.muC] = muC
    par[p.muK] = muK
    par[p.b] = b
    par[p.rQ] = rQ

    s = numpy.zeros((len(v), T + 1))
    i = numpy.zeros((len(v), N_ITER + 1))

    s[v.DH, 0] = 1.0
    s[v.D, 0] = 1.0
    s[v.KFC, 0] = 0.5
    s[v.KFK, 0] = 0.5
    s[v.pC, 0] = 1.0
    s[v.beta, 0] = 1.00001

    for t in range(1, T + 1):
        i[:, 0] = s[:, t - 1]
        for j in range(1, N_ITER + 1):

            ##########
            # LAGGED #
            ##########

            ### T
            i[v.T, j] = (
                par[p.tauW] * i[v.W, j - 1]
                + par[p.tauC] * i[v.C, j - 1]
                + par[p.tauS] * i[v.rS, j - 1] * i[v.S, j - 1]
            )

            ### GT
            i[v.GT, j] = max(
                0,
                par[p.delta] * i[v.Y, j - 1]
                + i[v.T, j]
                - i[v.M, j - 1]
                - i[v.rB, j - 1] * i[v.BG, j - 1],
            )

            ### Pi
            i[v.PiFC, j] = max(0, par[p.rF] * (i[v.DFC, j - 1] - i[v.LFC, j - 1]))
            i[v.PiFK, j] = max(0, par[p.rF] * (i[v.DFK, j - 1] - i[v.LFK, j - 1]))
            i[v.Pi, j] = i[v.PiFC, j] + i[v.PiFK, j]

            ### omega
            i[v.omega, j] = 1 - (i[v.nC, j - 1] + i[v.nK, j - 1] + i[v.nQ, j - 1])

            ### nQ
            if i[v.W, j - 1] == 0 or i[v.omega, j] == 1.0:
                i[v.nQ, j] == 0.0,
            else:
                i[v.nQ, j] = min(
                    1.0 - (i[v.nC, j - 1] + i[v.nK, j - 1]),
                    max(
                        0,
                        par[p.rQ]
                        * (i[v.pK, j - 1] * i[v.I, j - 1] - i[v.WFK, j - 1])
                        * (1 - i[v.omega, j])
                        / i[v.W, j - 1],
                    ),
                )

            ##########
            # SYNCED #
            ##########

            ### Gamma
            if i[v.L, 0] == 0:
                i[v.Gamma, j] = par[p.GammaT]
            else:
                i[v.Gamma, j] = (i[v.BB, 0] + i[v.L, 0] - i[v.D, 0] - i[v.S, 0]) / i[
                    v.L, 0
                ]

            ### beta
            i[v.beta, j] = i[v.beta, 0] * (1 + i[v.nQ, j]) ** par[p.b]

            ### W
            if i[v.omega, j] == 1:
                i[v.W, j] = 0
            else:
                if i[v.W, 0] == 0:
                    i[v.W, j] = 1 - i[v.omega, j]
                else:
                    i[v.W, j] = (
                        i[v.W, 0]
                        / (1 - i[v.omega, 0])
                        * (1 - i[v.omega, j])
                        * par[p.aW] ** (i[v.omega, 0] - i[v.omega, j])
                    )

            ### M
            if i[v.omega, 0] == 1:
                i[v.M, j] = i[v.M, 0] / i[v.omega, 0] * i[v.omega, j]
            else:
                i[v.M, j] = par[p.phi] * i[v.W, 0] / (1 - i[v.omega, 0]) * i[v.omega, j]

            ### CT
            i[v.CT, j] = max(
                0,
                par[p.ay] * (i[v.W, j] + i[v.M, j])
                + par[p.av] * (i[v.DH, 0] + i[v.S, 0]),
            )

            ### G
            i[v.G, j] = min(
                i[v.GT, j], min(i[v.KFC, 0], i[v.nC, j]) * i[v.beta, j] * par[p.k]
            )

            ### C
            i[v.C, j] = min(
                i[v.CT, j],
                min(i[v.KFC, 0], i[v.nC, j]) * i[v.beta, j] * par[p.k] - i[v.G, j],
            )

            ### S
            i[v.S, j] = max(0, i[v.DH, 0] + i[v.S, 0] - par[p.rhoC] * i[v.C, j])

            ### DK
            i[v.DKT, j] = max(
                0,
                (par[p.gamma] * min(i[v.KFK, 0], i[v.nK, 0]) + i[v.IT, 0])
                / (par[p.uT] * i[v.beta, j])
                - (1 - par[p.gamma]) * i[v.KFK, 0],
            )
            i[v.DK, j] = min(
                i[v.DKT, j], max(0, min(i[v.KFK, 0], i[v.nK, j])) * i[v.beta, j]
            )

            ### pC
            if i[v.W, j] == 0 or i[v.omega, j] == 1.0:
                i[v.pC, j] = max(0.01, i[v.pC, j])
            else:
                i[v.pC, j] = max(
                    0.01,
                    (1 + par[p.muC])
                    * i[v.W, j]
                    / ((1 - i[v.omega, j]) * i[v.beta, j] * par[p.k]),
                )

            ### psi
            i[v.psi, j] = i[v.pC, j] / i[v.pC, 0] - 1

            ### r
            i[v.rB, j] = max(
                0,
                min(
                    0.1,
                    i[v.psi, j]
                    + par[p.a1] * (i[v.psi, j] - par[p.psiT])
                    - par[p.a2] * (i[v.omega, j] - par[p.omegaT])
                    + par[p.a3] * (i[v.u, j] - par[p.uT]),
                ),
            )
            i[v.rS, j] = max(
                0, i[v.rB, j] + par[p.a4] * (max(0, i[v.Gamma, j]) - par[p.GammaT])
            )
            i[v.rL, j] = max(
                0, i[v.rB, j] - par[p.a5] * (max(0, i[v.Gamma, j]) - par[p.GammaT])
            )

            ### n*T
            i[v.nCT, j] = min(
                i[v.KFC, 0],
                max(
                    0,
                    (i[v.CT, j] + i[v.GT, j]) / (i[v.pC, j] * i[v.beta, j] * par[p.k]),
                ),
            )
            i[v.nKT, j] = min(
                i[v.KFK, 0],
                max(0, (i[v.DKT, j]) / i[v.beta, j]),
            )

            ### nC nK
            i[v.nC, j] = i[v.nCT, j] / max(1.0, i[v.nCT, j] + i[v.nKT, j])
            i[v.nK, j] = i[v.nKT, j] / max(1.0, i[v.nCT, j] + i[v.nKT, j])

            ### u
            i[v.u, j] = (i[v.nC, j] + i[v.nK, j]) / (i[v.KFC, 0] + i[v.KFK, 0])

            ### WF*
            if i[v.W, j] == 0 or i[v.nC, j] == 0 or i[v.omega, j] == 1:
                i[v.WFC, j] = 0
            else:
                i[v.WFC, j] = i[v.W, j] * i[v.nC, j] / (1 - i[v.omega, j])
            if i[v.W, j] == 0 or (i[v.nK, j] + i[v.nQ, j]) == 0 or i[v.omega, j] == 1:
                i[v.WFK, j] = 0
            else:
                i[v.WFK, j] = (
                    i[v.W, j] * (i[v.nK, j] + i[v.nQ, j]) / (1 - i[v.omega, j])
                )

            ### I
            i[v.IT, j] = max(
                0,
                (i[v.CT, j] + i[v.GT, j]) / (i[v.beta, j] * par[p.k] * i[v.pC, j])
                - (1 - par[p.gamma]) * i[v.KFC, 0],
            )
            i[v.I, j] = min(
                i[v.IT, j],
                max(
                    0,
                    (1 - par[p.gamma]) * i[v.KFK, 0]
                    + i[v.DK, j]
                    - (par[p.gamma] * min(i[v.KFK, 0], i[v.nK, 0]) + i[v.IT, 0])
                    / i[v.beta, j],
                ),
            )

            ### pK
            if i[v.WFK, j] == 0 or i[v.I, j] == 0:
                i[v.pK, j] = max(0.01, i[v.pK, j])
            else:
                i[v.pK, j] = max(0.01, (1 + par[p.muK]) * i[v.WFK, j] / i[v.I, j])

            ### K
            i[v.KFC, j] = (1 - par[p.gamma]) * i[v.KFC, 0] + i[v.I, j]
            i[v.KFK, j] = (1 - par[p.gamma]) * i[v.KFK, 0] + i[v.DK, j] - i[v.I, j]

            ### L
            i[v.LFC, j] = (
                max(
                    0,
                    i[v.WFC, j] - i[v.DFC, 0],
                    i[v.WFC, j]
                    + i[v.pK, j] * i[v.I, j]
                    - i[v.DFC, 0]
                    - (i[v.C, j] + i[v.G, j]),
                )
                + (1 - par[p.gamma]) * i[v.LFC, 0]
            )
            i[v.LFK, j] = (
                max(0, i[v.WFK, j] - i[v.DFK, 0]) + (1 - par[p.gamma]) * i[v.LFC, 0]
            )
            i[v.L, j] = i[v.LFC, j] + i[v.LFK, j]

            ### D
            i[v.DH, j] = (
                i[v.DH, 0]
                + i[v.S, 0]
                - i[v.S, j]
                - i[v.C, j]
                + i[v.W, j]
                + i[v.M, j]
                - i[v.T, j]
                + i[v.rS, j] * i[v.S, j]
            )
            i[v.DFC, j] = (
                i[v.DFC, 0]
                - i[v.LFC, 0]
                + i[v.LFC, j]
                + i[v.C, j]
                + i[v.G, j]
                - i[v.pK, j] * i[v.I, j]
                - i[v.WFC, j]
                - i[v.PiFC, j]
                - i[v.rL, j] * i[v.LFC, j]
            )
            i[v.DFK, j] = (
                i[v.DFK, 0]
                - i[v.LFK, 0]
                + i[v.LFK, j]
                + i[v.pK, j] * i[v.I, j]
                - i[v.WFK, j]
                - i[v.PiFK, j]
                - i[v.rL, j] * i[v.LFK, j]
            )
            i[v.D, j] = i[v.DH, j] + i[v.DFC, j] + i[v.DFK, j]

            ### B
            i[v.BB, j] = (
                i[v.L, 0]
                + i[v.BB, 0]
                - i[v.D, 0]
                - i[v.S, 0]
                - i[v.L, j]
                + i[v.D, j]
                + i[v.S, j]
                + i[v.Pi, j]
                - i[v.rS, j] * i[v.S, j]
                + i[v.rL, j] * i[v.L, j]
            ) / (1 - i[v.rB, j])
            i[v.BG, j] = (i[v.BG, 0] + i[v.G, j] + i[v.M, j] - i[v.T, j]) / (
                1 - i[v.rB, j]
            )

            ### Y
            i[v.Y, j] = i[v.C, j] + i[v.G, j] + i[v.pK, j] * i[v.I, j]

            ### g
            if i[v.Y, 0] == 0:
                i[v.g, j] = 0
            else:
                i[v.g, j] = i[v.Y, j] / i[v.Y, 0] - 1

        s[:, t] = i[:, N_ITER]

    bs = numpy.zeros((6, 6, T))
    fof = numpy.zeros((17, 6, T))

    bs[0, 0, :] = s[v.DH, 1:]
    bs[0, 1, :] = s[v.DFC, 1:]
    bs[0, 2, :] = s[v.DFK, 1:]
    bs[0, 3, :] = -s[v.D, 1:]
    bs[1, 0, :] = s[v.S, 1:]
    bs[1, 3, :] = -s[v.S, 1:]
    bs[2, 1, :] = -s[v.LFC, 1:]
    bs[2, 2, :] = -s[v.LFK, 1:]
    bs[2, 3, :] = s[v.L, 1:]
    bs[3, 3, :] = s[v.BB, 1:]
    bs[3, 4, :] = -s[v.BG, 1:]
    bs[4, 1, :] = s[v.pK, 1:] * s[v.KFC, 1:]
    bs[4, 2, :] = s[v.pK, 1:] * s[v.KFK, 1:]
    bs[:, 5, :] = bs.sum(axis=1)
    bs[5, :, :] = bs.sum(axis=0)

    fof[0, 0, :] = -s[v.pC, 1:] * s[v.C, 1:]
    fof[0, 1, :] = s[v.pC, 1:] * s[v.C, 1:]
    fof[1, 1, :] = s[v.pC, 1:] * s[v.G, 1:]
    fof[1, 4, :] = -s[v.pC, 1:] * s[v.G, 1:]
    fof[2, 2, :] = s[v.pK, 1:] * s[v.DK, 1:]
    fof[3, 1, :] = -s[v.pK, 1:] * s[v.I, 1:]
    fof[3, 2, :] = s[v.pK, 1:] * s[v.I, 1:]
    fof[4, 0, :] = s[v.W, 1:]
    fof[4, 1, :] = -s[v.WFC, 1:]
    fof[4, 2, :] = -s[v.WFK, 1:]
    fof[5, 0, :] = -s[v.T, 1:]
    fof[5, 1, :] = s[v.T, 1:]
    fof[6, 0, :] = s[v.M, 1:]
    fof[6, 1, :] = -s[v.M, 1:]
    fof[7, 1, :] = -s[v.PiFC, 1:]
    fof[7, 2, :] = -s[v.PiFK, 1:]
    fof[7, 3, :] = +s[v.Pi, 1:]
    fof[8, 0, :] = s[v.rS, 1:] * s[v.S, 1:]
    fof[8, 3, :] = -s[v.rS, 1:] * s[v.S, 1:]
    fof[9, 1, :] = -s[v.rL, 1:] * s[v.LFC, 1:]
    fof[9, 2, :] = -s[v.rL, 1:] * s[v.LFK, 1:]
    fof[9, 3, :] = s[v.rL, 1:] * s[v.L, 1:]
    fof[10, 3, :] = s[v.rB, 1:] * s[v.BB, 1:]
    fof[10, 4, :] = -s[v.rB, 1:] * s[v.BG, 1:]
    fof[11, 0, :] = -(s[v.DH, 1:] - s[v.DH, 0:-1])
    fof[11, 1, :] = -(s[v.DFC, 1:] - s[v.DFC, 0:-1])
    fof[11, 2, :] = -(s[v.DFK, 1:] - s[v.DFK, 0:-1])
    fof[11, 3, :] = s[v.D, 1:] - s[v.D, 0:-1]
    fof[12, 0, :] = -(s[v.S, 1:] - s[v.S, 0:-1])
    fof[12, 3, :] = s[v.S, 1:] - s[v.S, 0:-1]
    fof[13, 1, :] = s[v.LFC, 1:] - s[v.LFC, 0:-1]
    fof[13, 2, :] = s[v.LFK, 1:] - s[v.LFK, 0:-1]
    fof[13, 3, :] = -(s[v.L, 1:] - s[v.L, 0:-1])
    fof[14, 3, :] = -(s[v.BB, 1:] - s[v.BB, 0:-1])
    fof[14, 4, :] = s[v.BG, 1:] - s[v.BG, 0:-1]
    fof[15, 1, :] = -(s[v.pK, 1:] * s[v.KFC, 1:] - s[v.pK, 0:-1] * s[v.KFC, 0:-1])
    fof[15, 2, :] = -(s[v.pK, 1:] * s[v.KFK, 1:] - s[v.pK, 0:-1] * s[v.KFK, 0:-1])
    fof[:, 5, :] = fof.sum(axis=1)
    fof[16, :, :] = fof.sum(axis=0)

    return par, s, i, bs, fof


def print_bs(bs, t):
    print(
        tabulate(
            numpy.hstack(
                (numpy.array([["D", "S", "L", "B", "K", "V"]]).T, bs[:, :, t])
            ),
            floatfmt=".4f",
            headers=["", "H", "FC", "FK", "B", "G", ""],
        )
    )


def print_fof(fof, t):
    print(
        tabulate(
            numpy.hstack(
                (
                    numpy.array(
                        [
                            [
                                "C",
                                "G",
                                "DK+",
                                "I",
                                "W",
                                "T",
                                "M",
                                "Pi",
                                "rS",
                                "rL",
                                "rB",
                                "DD",
                                "DS",
                                "DL",
                                "DB",
                                "DK",
                                "",
                            ]
                        ]
                    ).T,
                    fof[:, :, t],
                )
            ),
            floatfmt=".4f",
            headers=["", "H", "FC", "FK", "B", "G", ""],
        )
    )


def print_v(s, t):
    print(
        tabulate(
            numpy.vstack((numpy.array([e.name for e in v]), s[:, t])).T,
            floatfmt=".4f",
        )
    )


def print_p(par, t):
    print(
        tabulate(
            numpy.vstack((numpy.array([e.name for e in p]), par[:, t])).T,
            floatfmt=".4f",
        )
    )
