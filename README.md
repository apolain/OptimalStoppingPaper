# OptimalStoppingPaper

Code repository for numerical experiments on Bermudan option pricing using classical and learning-based methods.

---

## Overview

This project implements and compares several approaches for Bermudan option pricing:

* LSMC (Longstaff–Schwartz least-squares Monte Carlo)
* DOS (Deep Optimal Stopping)
* PG (policy-gradient / REINFORCE)
* A2C (actor-critic variant of policy gradient)

The implementation is built around a reusable pricing library ("bermudan") and a set of experiment scripts ("experiments") used to reproduce the numerical study.

---

## Repository structure

OptimalStoppingPaper/
│
├── bermudan/
│   ├── diffusions/     # Simulatable market models (GBM, Heston)
│   ├── methods/        # Pricing methods: LSMC, DOS, PG, A2C
│   ├── networks/       # Feature construction and neural networks
│   ├── options/        # Bermudan option specification
│   ├── payoffs/        # Payoff definitions (Put, MaxCall)
│   └── utils/          # Logging, timing, seeds, stopping times
│
├── experiments/
│   └── full/           # Main experiment suite
│
├── Pipfile
└── Pipfile.lock

---

## Implemented models

Diffusions:

* GBM: exact log-normal simulation under the risk-neutral measure
* Heston: Euler–Maruyama scheme with full truncation

Payoffs:

* Put
* MaxCall

Pricing methods:

* LSMC: regression-based backward induction
* DOS: separate neural network per exercise date
* PG: policy-gradient with shared neural network
* A2C: actor-critic variant with learned value baseline

---

## Main abstraction

The central object is "BermudanOption", which bundles:

* the diffusion model,
* the payoff,
* the maturity and time grid,
* the Bermudan exercise dates,
* the numerical configuration (TorchConfig).

This unified interface allows all pricing methods to be applied consistently to the same problem.

---

## Installation

Python version: 3.10

Using Pipenv:
pipenv install
pipenv shell

Main dependencies:

* torch==2.1
* numpy<2
* pandas
* matplotlib

---

## Running experiments

Main experiment suite:
python experiments/full/main.py --device cpu

With GPU:
python experiments/full/main.py --device cuda

Quick run (reduced workload):
python experiments/full/main.py --device cpu --quick

---

## Experiment cases

The main experiment script includes:

* Case A: 1D Bermudan put under GBM
* Case B: Bermudan max-call under multi-asset GBM
* Case C: Bermudan put under Heston
* Scaling experiment: performance vs number of exercise dates

---

## Numerical implementation details

* Path simulation is performed in float32 for efficiency
* Downstream numerical computations use float64
* TorchConfig separates simulation precision from numerical precision
* Training uses freshly simulated Monte Carlo trajectories at each epoch

Policy-gradient methods:

* Shared neural network across exercise dates
* Time is included as an input feature
* Entropy regularisation prevents premature convergence

Actor-critic variant:

* Uses a learned value function as a state-dependent baseline
* Reduces gradient variance during training
* Critic is used only during training and discarded at evaluation

---

## Outputs

Experiments write logs to:
logs/experiment/

Outputs include:

* summary.csv (prices, standard errors, runtimes, metadata)
* optional intermediate logs for training diagnostics

---

## Reproducibility

The repository includes:

* deterministic seeding utilities
* centralized experiment logging
* separate validation scripts for benchmarks

---

## Status

This repository contains the code used for the numerical experiments of the associated research paper.

Additional documentation (exact replication steps, figures, and tables) can be added as the experimental protocol is finalised.
