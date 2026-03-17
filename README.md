score_identity.py
-----------------
Numerical verification of the exact score identity between product-limit
tail completion and inverse-probability-of-censoring weighting (IPCW)
under right censoring.
 
Reference
---------
...............

Description
-----------
For a parametric event-time model without covariates, in the absence of ties,
and when the largest observed time is an event, the completion-induced score
 
    U_comp(theta) = sum_i E[ phi(T_i^*; theta) | D_n ]
 
is exactly equal to the classical IPCW score
 
    U_IPCW(theta) = sum_i (delta_i / G_hat(Y_i^-)) phi(Y_i; theta).
 
This script:
  1. Implements both scores from scratch for the exponential model.
  2. Generates right-censored data and verifies the identity numerically.
  3. Produces three publication-quality figures:
     - Figure 1 (left):  U_comp and U_IPCW overlaid as functions of theta.
     - Figure 1 (right): Their difference Delta(theta) at machine precision.
     - Figure 2:         Histogram of |theta_comp - theta_IPCW| across
                         500 Monte Carlo replications.
 
Usage
-----
    python score_identity.py
 
Output
------
    fig1_scores.pdf / .png   -- Score curves and their difference (two panels)
    fig2_mc_roots.pdf / .png -- Monte Carlo root-difference histogram
 
Dependencies
------------
    numpy, scipy, matplotlib  (standard scientific Python stack)
 
Author: Ersin Yilmaz, Mugla Sitki Kocman University
