**A list of Python-based MCMC packages**.
Also here's a nice list
of [MCMC algorithms](https://m-clark.github.io/docs/ld_mcmc/).


## abcpmc

> A Python Approximate Bayesian Computing (ABC) Population Monte Carlo (PMC)
implementation based on Sequential Monte Carlo (SMC) with Particle Filtering
techniques.
>
> Features:
> * Entirely implemented in Python and easy to extend
> * Follows Beaumont et al. 2009 PMC algorithm
> * Parallelized with muliprocessing or message passing interface (MPI)
> * Extendable with k-nearest neighbour (KNN) or optimal local covariance
matrix (OLCM) pertubation kernels (Fillipi et al. 2012)
> * Detailed examples in IPython notebooks

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/jakeret/abcpmc) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://abcpmc.readthedocs.org/) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1504.07245)

---


## astroABC

> astroABC is a Python implementation of an Approximate Bayesian Computation
Sequential Monte Carlo (ABC SMC) sampler for parameter estimation.
>
> Key features
> * Parallel sampling using MPI or multiprocessing
> * MPI communicator can be split so both the sampler, and simulation launched
by each particle, can run in parallel
> * A Sequential Monte Carlo sampler (see e.g. Toni et al. 2009, Beaumont et
al. 2009, Sisson & Fan 2010)
> * A method for iterative adapting tolerance levels using the qth quantile of
the distance for t iterations (Turner & Van Zandt (2012))
> * Scikit-learn covariance matrix estimation using Ledoit-Wolf shrinkage for
singular matrices
> * A module for specifying particle covariance using method proposed by Turner
& Van Zandt (2012), optimal covariance matrix for a multivariate normal
perturbation kernel, local covariance estimate using scikit-learn KDTree method
for nearest neighbours (Filippi et al 2013) and a weighted covariance 
(Beaumont et al 2009)
> * Restart files output frequently so an interrupted run can be resumed at any
iteration
> * Output and restart files are backed up every iteration
> * User defined distance metric and simulation methods
> * A class for specifying heterogeneous parameter priors
> * Methods for drawing from any non-standard prior PDF e.g using Planck/WMAP
chains
> * A module for specifying a constant, linear, log or exponential tolerance
level
> * Well-documented examples and sample scripts

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/EliseJ/astroABC) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://github.com/EliseJ/astroABC/wiki) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1608.07606)

---


## bmcmc

> bmcmc is a general purpose mcmc package which should be useful for Bayesian
data analysis. It uses an adaptive scheme for automatic tuning of proposal
distributions. It can also handle hierarchical Bayesian models via
Metropolis-Within-Gibbs scheme.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/sanjibs/bmcmc/) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://bmcmc.readthedocs.io) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1706.01629)

---


## CosmoABC

> cosmoabc is a package which enables parameter inference using an Approximate
Bayesian Computation (ABC) algorithm. The code was originally designed for
cosmological parameter inference from galaxy clusters number counts based on
Sunyaev-Zel’dovich measurements. In this context, the cosmological simulations
were performed using the NumCosmo library .

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/COINtoolbox/CosmoABC) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://cosmoabc.readthedocs.io) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1504.06129)

---


## dynesty

> A Dynamic Nested Sampling package for computing Bayesian posteriors and
evidences.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/joshspeagle/dynesty) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://dynesty.readthedocs.io/)

---


## ELFI

> ELFI is a statistical software package written in Python for likelihood-free
inference (LFI) such as Approximate Bayesian Computation (ABC). The term LFI
refers to a family of inference methods that replace the use of the likelihood
function with a data generating simulator function. ELFI features an easy to use
generative modeling syntax and supports parallelized inference out of the box.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/elfi-dev/elfi) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://elfi.readthedocs.io/) |
<img src="./img/art.png" width="20" height="20"> [Article](
http://www.jmlr.org/papers/v19/17-374.html)

---


## emcee

> emcee is an MIT licensed pure-Python implementation of Goodman & Weare’s
[Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble
sampler](http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml). It's designed for
Bayesian parameter estimation and it's really sweet!

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/dfm/emcee) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://emcee.readthedocs.io) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1202.3665)

---


## hmc

> A simple Hamiltonian MCMC sampler.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/bd-j/hmc)

---


## hoppMCMC

> An adaptive basin-hopping Markov-chain Monte Carlo algorithm for Bayesian
optimisation. Python implementation of the hoppMCMC algorithm aiming to identify
and sample from the high-probability regions of a posterior distribution. The
algorithm combines three strategies: (i) parallel MCMC, (ii) adaptive Gibbs
sampling and (iii) simulated annealing. Overall, hoppMCMC resembles the
basin-hopping algorithm implemented in the optimize module of scipy, but it is
developed for a wide range of modelling approaches including stochastic models
with or without time-delay.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/kerguler/hoppMCMC)

---


## kombine

> kombine is an ensemble sampler built for efficiently exploring multimodal
distributions. By using estimates of ensemble’s instantaneous distribution as a
proposal, it achieves very fast burnin, followed by sampling with very short
autocorrelation times.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/bfarr/kombine) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://pages.uoregon.edu/bfarr/kombine/index.html)

---


## MCcubed

> Powerful Bayesian-statistics tool that offers:
>
> * Levenberg-Marquardt least-squares optimization.
> * Markov-chain Monte Carlo (MCMC) posterior-distribution sampling following
the:
>   * Metropolis-Hastings algorithm with Gaussian proposal distribution,
>   * Differential-Evolution MCMC (DEMC), or
>   * DEMCzs (Snooker).

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/pcubillos/MCcubed) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://pcubillos.github.io/MCcubed/) |
<img src="./img/art.png" width="20" height="20"> [Article](
http://adsabs.harvard.edu/abs/2017AJ....153....3C)

---


## Nested Sampling

> Flexible and efficient Python implementation of the nested sampling algorithm.
This implementation is geared towards allowing statistical physicists to use
this method for thermodynamic analysis but is also being used by
astrophysicists.
>
> This implementation uses the language of statistical mechanics (partition
function, phase space, configurations, energy, density of states) rather than
the language of Bayesian sampling (likelihood, prior, evidence). This is simply
for convenience, the method is the same.
>
> The package goes beyond the bare implementation of the method providing:
>
>* built-in parallelisation on single computing node (max total number of cpu
threads on a single machine)
>* built-in Pyro4-based parallelisation by distributed computing, ideal to run
calculations on a cluster or across a network
>* ability to save and restart from checkpoint binary files, ideal for very
long calculations
>* scripts to compute heat capacities and perform error analysis
integration with the MCpele package to implement efficient Monte Carlo walkers.

* [Official site](http://www.inference.phy.cam.ac.uk/bayesys/)
* [Compared to annealing](http://www.mrao.cam.ac.uk/~steve/malta2009/images/nestposter.pdf)

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/js850/nested_sampling) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://js850.github.io/nested_sampling/) |
<img src="./img/art.png" width="20" height="20"> [Article](
http://projecteuclid.org/euclid.ba/1340370944)

---


## nestle

http://kylebarbary.com/nestle/

---


## NUTS

https://github.com/mfouesneau/NUTS

---


## pgmpy

> pgmpy is a python library for working with Probabilistic Graphical Models.

* Repo: https://github.com/pgmpy/pgmpy
* Docs (and list of algorithms supported): http://pgmpy.org/
* Examples: https://github.com/pgmpy/pgmpy/tree/dev/examples
* Basic tutorial: https://github.com/pgmpy/pgmpy_notebook
* Article: [MCMC: Hamiltonian Monte Carlo and No-U-Turn
Sampler](http://khalibartan.github.io/MCMC-Hamiltonian-Monte-Carlo-and-No-U-Turn-Sampler/)

---


## ptemcee

> ...is a fork of Daniel Foreman-Mackey's wonderful emcee to implement parallel
tempering more robustly. As far as possible, it is designed as a drop-in
replacement for emcee. If you're trying to characterise awkward, multi-modal
probability distributions, then ptemcee is your friend.

* Repo: https://github.com/willvousden/ptemcee
* Docs: http://ptemcee.readthedocs.io/en/latest/

---


## PTMCMCSampler

https://github.com/jellis18/PTMCMCSampler

---


## ptmpi

https://github.com/chris-n-self/ptmpi (blog entry
https://chrisdoesscience.wordpress.com/2016/07/17/parallelised-parallel-tempering-with-mpi/)

---


## pyabc

https://github.com/icb-dcm/pyabc

---


## PyDREAM

https://github.com/LoLab-VU/PyDREAM

---


## pyhmc

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

---


## PyJAGS

> PyJAGS provides a Python interface to JAGS, a program for analysis of
Bayesian hierarchical models using Markov Chain Monte Carlo (MCMC) simulation.

* Repo: https://github.com/tmiasko/pyjags
* Docs: https://pyjags.readthedocs.io/en/latest/
* Blog article: https://martynplummer.wordpress.com/2016/01/11/pyjags/

---


## PyMC3

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

---


## pymcmc

https://github.com/gmcgoldr/pymcmc

---


## py-mcmc

https://pypi.org/project/py-mcmc/

---


## PyMultiNest

https://github.com/JohannesBuchner/PyMultiNest

---


## PyStan

> PyStan provides an interface to Stan, a package for Bayesian inference using
the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.*

* Stan's homepage: http://mc-stan.org/about/
* PyStan's homepage: https://pystan.readthedocs.io/en/latest/index.html
* Examples: https://stats.stackexchange.com/questions/162857/managing-high-autocorrelation-in-mcmc (autocorrelation)

---


## Sampyl

> Sampyl is a Python library implementing Markov Chain Monte Carlo (MCMC)
samplers in Python. It’s designed for use in Bayesian parameter estimation
and provides a collection of distribution log-likelihoods for use in
constructing models.

* Repo: https://github.com/mcleonard/sampyl/
* Docs: http://matatat.org/sampyl/index.html
* Example: http://matatat.org/ab-testing-with-sampyl.html

---


## simpleabc

https://github.com/rcmorehead/simpleabc

---


## SPOTPY

https://github.com/thouska/spotpy

---



## UltraNest

https://github.com/JohannesBuchner/UltraNest

---


## XHMC

https://arxiv.org/abs/1601.00225
