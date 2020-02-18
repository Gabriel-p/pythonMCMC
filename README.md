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

> Package which enables parameter inference using an Approximate
Bayesian Computation (ABC) algorithm. The code was originally designed for
cosmological parameter inference from galaxy clusters number counts based on
Sunyaev-Zel’dovich measurements. In this context, the cosmological simulations
were performed using the NumCosmo library.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/COINtoolbox/CosmoABC) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://cosmoabc.readthedocs.io) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1504.06129)

---


## CPNest

> Parallel nested sampling in python.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/johnveitch/cpnest) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://johnveitch.github.io/cpnest/)

---


## dynesty

> A Dynamic Nested Sampling package for computing Bayesian posteriors and
evidences.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/joshspeagle/dynesty) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://dynesty.readthedocs.io/)

---


## Edward2

> Edward2 is a probabilistic programming language in TensorFlow and Python. It
extends the TensorFlow ecosystem so that one can declare models as probabilistic
programs and manipulate a model's computation for flexible training, latent
variable inference, and predictions.

* Original project: [Edward](http://edwardlib.org/) ([Tran et al.
(2016)](https://arxiv.org/abs/1610.09787))

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2)

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


## MC3

> Multi-Core Markov-Chain Monte Carlo (MC3) is a powerful Bayesian-statistics
tool that offers:
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


## Nestle

> Pure Python, MIT-licensed implementation of nested sampling algorithms.
> Nested Sampling is a computational approach for integrating posterior
probability in order to compare models in Bayesian statistics. It is similar to
Markov Chain Monte Carlo (MCMC) in that it generates samples that can be used to
estimate the posterior probability distribution. Unlike MCMC, the nature of the
sampling also allows one to calculate the integral of the distribution. It also
happens to be a pretty good method for robustly finding global maxima.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/kbarbary/nestle) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://kylebarbary.com/nestle/)

---


## NUTS

> No-U-Turn Sampler (NUTS) for python
> This package implements the No-U-Turn Sampler (NUTS) algorithm 6 from the NUTS
paper (Hoffman & Gelman, 2011).

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/mfouesneau/NUTS) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://js850.github.io/nested_sampling/) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1111.4246)

---


## pgmpy

> Python library for working with Probabilistic Graphical Models.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/pgmpy/pgmpy) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://pgmpy.org/)

---


## ptemcee

> Fork of Daniel Foreman-Mackey's emcee to implement parallel
tempering more robustly. As far as possible, it is designed as a drop-in
replacement for emcee. If you're trying to characterise awkward, multi-modal
probability distributions, then ptemcee is your friend.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/willvousden/ptemcee) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://ptemcee.readthedocs.io) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1501.05823)

---


## PTMCMCSampler

> MPI enabled Parallel Tempering MCMC code written in Python.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/jellis18/PTMCMCSampler) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://jellis18.github.io/PTMCMCSampler/)

---


## ptmpi

> Python class that coordinates an MPI implementation of parallel tempering.
> Supports a fully parallelised implementation of parallel tempering using
mpi4py (message passing interface for python). Each replica runs as a separate
parallel process and they communicate via an mpi4py object. To minimise message
passing the replicas stay in place and only the temperatures are exchanged
between the processes. It is this exchange of temperatures that ptmpi handles.

* [Blog entry](https://chrisdoesscience.wordpress.com/2016/07/17/parallelised-parallel-tempering-with-mpi/)

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/chris-n-self/ptmpi) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://jellis18.github.io/PTMCMCSampler/)

---


## pyabc

> pyABC is a framework for distributed, likelihood-free inference. That means,
if you have a model and some data and want to know the posterior distribution
over the model parameters, i.e. you want to know with which probability which
parameters explain the observed data, then pyABC might be for you.
>
> All you need is some way to numerically draw samples from the model, given the
model parameters. pyABC “inverts” the model for you and tells you which
parameters were well matching and which ones not. You do not need to
analytically calculate the likelihood function.
>
> pyABC runs efficiently on multi-core machines and distributed cluster setups.
It is easy to use and flexibly extensible.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/icb-dcm/pyabc) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://pyabc.readthedocs.io/en/latest/index.html) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://www.biorxiv.org/content/early/2017/07/17/162552)

---


## PyDREAM

> A Python implementation of the MT-DREAM(ZS) algorithm.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/LoLab-VU/PyDREAM) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://pydream.readthedocs.io/en/latest/) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2011WR010608)

---


## pyhmc

> This package is a straight-forward port of the functions `hmc2.m` and
`hmc2_opt.m` from the MCMCstuff matlab toolbox written by Aki Vehtari. The code
is originally based on the functions hmc.m from the netlab toolbox written by
Ian T Nabney. The portion of algorithm involving "windows" is derived from the C code for this function included in the Software for Flexible Bayesian Modeling written by Radford Neal.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/rmcgibbo/pyhmc) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://pythonhosted.org/pyhmc/) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1206.1901v1)

---


## PyJAGS

> PyJAGS provides a Python interface to JAGS, a program for analysis of
Bayesian hierarchical models using Markov Chain Monte Carlo (MCMC) simulation.

* [Blog article](https://martynplummer.wordpress.com/2016/01/11/pyjags/)

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/tmiasko/pyjags) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://pyjags.readthedocs.io)

---


## PyMC3

> PyMC3 is a probabilistic programming module for Python that allows users
to fit Bayesian models using a variety of numerical methods, most notably
Markov chain Monte Carlo (MCMC) and variational inference (VI). Its flexibility
and extensibility make it applicable to a large suite of problems. Along with
core model specification and fitting functionality, PyMC3 includes
functionality for summarizing output and for model diagnostics.

* Tutorials:
  1. [Bayesian Modelling in Python](https://github.com/markdregan/
Bayesian-Modelling-in-Python)
  1. [Using PyMC3](http://people.duke.edu/~ccc14/sta-663-2017/19A_PyMC3.html)
  1. [Tutorial 5a: Parameter estimation with Markov chain Monte Carlo](http://
bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2017/tutorials/
t5a_mcmc.html)
* Books:
  1. [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/
Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
  1. [Statistical Rethinking with Python and PyMC3](https://github.com/
aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3)

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/pymc-devs/pymc3) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://docs.pymc.io/intro.html) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://arxiv.org/abs/1507.08050)

---


## PyMCMC

> Simple implementation of the Metropolis-Hastings algorithm for Markov Chain
Monte Carlo sampling of multidimensional spaces.
> The implementation is minimalistic. All that is required is a funtion which
accepts an iterable of parameter values, and returns the positive log likelihood
at that point.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/gmcgoldr/pymcmc)

---


## py-mcmc

> A simple MCMC framework for training Gaussian processes adding functionality
to GPy.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/PredictiveScienceLab/py-mcmc)

---


## PyMultiNest

> MultiNest is a program and a sampling technique. As a Bayesian inference
technique, it allows parameter estimation and model selection. Recently,
MultiNest added Importance Nested Sampling which is now also supported.
> The efficient Monte Carlo algorithm for sampling the parameter space is based
on nested sampling and the idea of disjoint multi-dimensional ellipse sampling.
> For the scientific community, where Python is becoming the new lingua franca 
(luckily), I provide an interface to MultiNest.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/JohannesBuchner/PyMultiNest) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://johannesbuchner.github.io/PyMultiNest/) |
<img src="./img/art.png" width="20" height="20"> [Article](
http://www.aanda.org/articles/aa/abs/2014/04/aa22971-13/aa22971-13.html)

---


## pysmc

> pysmc is a Python package for sampling complicated probability densities using the celebrated Sequential Monte Carlo method.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/PredictiveScienceLab/pysmc) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://predictivesciencelab.github.io/pysmc/)

---


## PyStan

> PyStan provides an interface to [Stan](http://mc-stan.org/), a package for
Bayesian inference using the No-U-Turn sampler, a variant of Hamiltonian
Monte Carlo.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/stan-dev/pystan) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://pystan.readthedocs.io)

---


## Sampyl

> Sampyl is a Python library implementing Markov Chain Monte Carlo (MCMC)
samplers in Python. It’s designed for use in Bayesian parameter estimation
and provides a collection of distribution log-likelihoods for use in
constructing models.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/mcleonard/sampyl/) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://matatat.org/sampyl/index.html)

---


## simpleabc

> A Python package for Approximate Bayesian Computation.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/rcmorehead/simpleabc)

---


## SPOTPY

> A Statistical Parameter Optimization Tool for Python.
> SPOTPY is a Python framework that enables the use of Computational
optimization techniques for calibration, uncertainty and sensitivity analysis
techniques of almost every (environmental-) model. 

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/thouska/spotpy) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
http://fb09-pasig.umwelt.uni-giessen.de/spotpy/) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0145180)

---


## UltraNest

> Nested Sampling Development Framework & UltraNest
> A Pythonic implementation of the Nested Sampling integration algorithm for
Bayesian model comparison and parameter estimation. This package provides
multiple implementations for constrained drawing functions, and a test suite to
evaluate the correctness, accuracy and efficiency of various implementations.
> The source code consists of three components:
>
>* A Modular framework for nested sampling algorithms (nested_sampling) and
their development
>* Test framework to evaluate the performance and accuracy of algorithms 
(testsuite)
>* UltraNest - A fast C implementation of a mixed RadFriends/MCMC nested
sampling algorithm

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/JohannesBuchner/UltraNest) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://johannesbuchner.github.io/UltraNest/ultranest/) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://link.springer.com/article/10.1007/s11222-014-9512-y)

---


## Zeus

> zeus is a pure-Python implementation of the Ensemble Slice Sampling method.
> >
> * Fast & Robust Bayesian Inference,
> * No hand-tuning,
> * Excellent performance in terms of autocorrelation time and convergence rate,
> * Scale to multiple CPUs without any extra effort.

<img src="./img/github.png" width="20" height="20"> [Repo](
https://github.com/minaskar/zeus) | 
<img src="./img/docs.png" width="20" height="20"> [Docs](
https://zeus-mcmc.readthedocs.io/) |
<img src="./img/art.png" width="20" height="20"> [Article](
https://ui.adsabs.harvard.edu/abs/2020arXiv200206212K/abstract)
