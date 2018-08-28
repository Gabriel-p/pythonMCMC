List of Python-based MCMC packages in no particular order.


##### Table of Contents  
1. [PyMC3](#pymc3)
1. [PyStan](#pystan)
1. [PyJAGS](#pyjags)
1. [emcee](#emcee)
1. [ptemcee](#ptemcee)
1. [pgmpy](#pgmpy)
1. [pyhmc](#pyhmc)



## [PyMC3](http://docs.pymc.io/intro.html)

> PyMC3 is a probabilistic programming module for Python that allows users
to fit Bayesian models using a variety of numerical methods, most notably
Markov chain Monte Carlo (MCMC) and variational inference (VI). Its flexibility
and extensibility make it applicable to a large suite of problems. Along with
core model specification and fitting functionality, PyMC3 includes
functionality for summarizing output and for model diagnostics.

* Docs: http://docs.pymc.io/intro.html
* Repo: https://github.com/pymc-devs/pymc3
* Tutorials:
  1. [Bayesian Modelling in Python](https://github.com/markdregan/Bayesian-Modelling-in-Python)
  1. [Using PyMC3](http://people.duke.edu/~ccc14/sta-663-2017/19A_PyMC3.html)
  1. [Tutorial 5a: Parameter estimation with Markov chain Monte Carlo](http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2017/tutorials/t5a_mcmc.html)
* Recommended book: [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
* Recommended book: [Statistical Rethinking with Python and PyMC3](https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3)
* Examples: https://stats.stackexchange.com/questions/119879/how-to-interpret-autocorrelation-plot-in-mcmc (autocorrelation)
* Article: [Probabilistic Programming in Python using PyMC, Salvatier et al. (2015)](https://arxiv.org/abs/1507.08050)


## [PyStan](http://mc-stan.org/about/)

> PyStan provides an interface to Stan, a package for Bayesian inference using
the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.*

* Stan's homepage: http://mc-stan.org/about/
* PyStan's homepage: https://pystan.readthedocs.io/en/latest/index.html
* Examples: https://stats.stackexchange.com/questions/162857/managing-high-autocorrelation-in-mcmc (autocorrelation)


## [PyJAGS](https://pyjags.readthedocs.io/en/latest/)

> PyJAGS provides a Python interface to JAGS, a program for analysis of
Bayesian hierarchical models using Markov Chain Monte Carlo (MCMC) simulation.

* Repo: https://github.com/tmiasko/pyjags
* Blog article: https://martynplummer.wordpress.com/2016/01/11/pyjags/


## [emcee](http://dfm.io/emcee/current/)

> emcee is an MIT licensed pure-Python implementation of Goodman & Weare’s
[Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble
sampler](http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml)

* Docs: https://emcee.readthedocs.io/en/latest/
* Repo: https://github.com/dfm/emcee
* Article: [emcee: The MCMC Hammer, Foreman-Mackey et al.
(2012)](https://arxiv.org/abs/1202.3665)


## [ptemcee](http://ptemcee.readthedocs.io/en/latest/)

> ...is a fork of Daniel Foreman-Mackey's wonderful emcee to implement parallel
tempering more robustly. As far as possible, it is designed as a drop-in
replacement for emcee. If you're trying to characterise awkward, multi-modal
probability distributions, then ptemcee is your friend.

* Repo: https://github.com/willvousden/ptemcee
* Docs: http://ptemcee.readthedocs.io/en/latest/


## [pgmpy](https://github.com/pgmpy/pgmpy)

> pgmpy is a python library for working with Probabilistic Graphical Models.

* Repo: https://github.com/pgmpy/pgmpy
* Docs (and list of algorithms supported): http://pgmpy.org/
* Examples: https://github.com/pgmpy/pgmpy/tree/dev/examples
* Basic tutorial: https://github.com/pgmpy/pgmpy_notebook
* Article: [MCMC: Hamiltonian Monte Carlo and No-U-Turn
Sampler](http://khalibartan.github.io/MCMC-Hamiltonian-Monte-Carlo-and-No-U-Turn-Sampler/)


## [pyhmc](https://pythonhosted.org/pyhmc/index.html)

> Hamiltonian Monte Carlo or Hybrid Monte Carlo (HMC) is a Markov chain Monte
Carlo (MCMC) algorithm. Hamiltonian dynamics can be used to produce distant
proposals for the Metropolis algorithm, thereby avoiding the slow exploration
of the state space that results from the diffusive behaviour of simple
random-walk proposals. It does this by taking a series of steps informed by
first-order gradient information. This feature allows it to converge much more
quickly to high-dimensional target distributions compared to simpler methods
such as Metropolis, Gibbs sampling (and derivatives).

* Repo: https://github.com/rmcgibbo/pyhmc
* Docs: https://pythonhosted.org/pyhmc/index.html


## [Sampyl](http://matatat.org/sampyl/index.html)
## [NUTS](https://github.com/mfouesneau/NUTS)
## [XHMC](https://arxiv.org/abs/1601.00225)
## [nestle](http://kylebarbary.com/nestle/)
## [DNest4](https://github.com/eggplantbren/DNest4)
## [kombine](http://pages.uoregon.edu/bfarr/kombine/index.html)
## [bmcmc](http://bmcmc.readthedocs.io/en/latest/index.html)
## [MCcubed](http://pcubillos.github.io/MCcubed/)
## [Nested Sampling](http://js850.github.io/nested_sampling/)
## [hmc](https://github.com/bd-j/hmc)
## [hoppMCMC](https://github.com/kerguler/hoppMCMC)
## [PyDREAM](https://github.com/LoLab-VU/PyDREAM)
