# References

Publications and software whose methods SeisForge implements or follows.

## Ambient noise

| reference | used in |
|---|---|
| Levshin, A. L., Pisarenko, V. F., & Pogrebinsky, G. A. (1972). On a frequency-time analysis of oscillations. *Annales de Géophysique*, 28, 211-218. | [AFTAN](ant/dispersion.md): frequency-time analysis |
| Levshin, A. L., & Ritzwoller, M. H. (2001). Automated detection, extraction, and measurement of regional surface waves. *Pure and Applied Geophysics*, 158, 1531-1545. | [AFTAN](ant/dispersion.md#5-phase-matched-filtering): phase-matched filter |
| Bensen, G. D., Ritzwoller, M. H., Barmin, M. P., Levshin, A. L., Lin, F., Moschetti, M. P., Shapiro, N. M., & Yang, Y. (2007). Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements. *Geophysical Journal International*, 169, 1239-1260. | [AFTAN](ant/dispersion.md#3-picking-a-continuous-dispersion-curve): SNR windows |
| Lin, F.-C., Ritzwoller, M. H., & Snieder, R. (2009). Eikonal tomography: surface wave tomography by phase front tracking across a regional broad-band seismic array. *Geophysical Journal International*, 177, 1091-1110. | [Eikonal tomography](ant/tomography.md) |
| Lin, F.-C., & Ritzwoller, M. H. (2011). Helmholtz surface wave tomography for isotropic and azimuthally anisotropic structure. *Geophysical Journal International*, 186, 1104-1120. | [Helmholtz correction](ant/tomography.md#2-the-helmholtz-correction) |

## Receiver functions

| reference | used in |
|---|---|
| Zhu, L., & Kanamori, H. (2000). Moho depth variation in southern California from teleseismic receiver functions. *Journal of Geophysical Research*, 105(B2), 2969-2980. | [H-k stacking](rf/hk.md#2-the-stack) |
| Ligorría, J. P., & Ammon, C. J. (1999). Iterative deconvolution and receiver-function estimation. *Bulletin of the Seismological Society of America*, 89, 1395-1400. | Gaussian parameter convention |
| Xu, M., & He, J. (2023). Seispy: Python module for batch calculation and postprocessing of receiver functions. *Seismological Research Letters*, 94. | H-k amplitude correction, RF conventions |
| Thomson, W. T. (1950). Transmission of elastic waves through a stratified solid medium. *Journal of Applied Physics*, 21, 89-93. | [Synthetic RF](rf/synthetic.md#2-from-the-surface-to-the-half-space): propagator |
| Haskell, N. A. (1962). Crustal reflection of plane P and SV waves. *Journal of Geophysical Research*, 67, 4751-4767. | Synthetic RF: propagator |
| Kennett, B. L. N. (1983). *Seismic Wave Propagation in Stratified Media*. Cambridge University Press. | Synthetic RF: motion-stress formulation |
| Aki, K., & Richards, P. G. (2002). *Quantitative Seismology* (2nd ed.), ch. 7. University Science Books. | Synthetic RF: layered-medium matrices |

## HVSR

| reference | used in |
|---|---|
| Sánchez-Sesma, F. J., et al. (2011). A theory for microtremor H/V spectral ratio: application for a layered medium. *Geophysical Journal International*, 186, 221-225. | [Diffuse-field H/V](hvsr/index.md) |
| García-Jerez, A., Piña-Flores, J., Sánchez-Sesma, F. J., Luzón, F., & Perton, M. (2016). A computer code for forward calculation and inversion of the H/V spectral ratio under the diffuse field assumption. *Computers & Geosciences*, 97, 67-78. | HV-DFA reference solver |

## Inversion

| reference | used in |
|---|---|
| Bodin, T., Sambridge, M., Tkalčić, H., Arroucau, P., Gallagher, K., & Rawlinson, N. (2012). Transdimensional inversion of receiver functions and surface wave dispersion. *Journal of Geophysical Research*, 117, B02301. | [sampled noise scales](inv/forward.md#3-the-likelihood) |
| Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.), Sec. 11.4. CRC Press. | split R-hat |
| Gelman, A., Roberts, G. O., & Gilks, W. R. (1996). Efficient Metropolis jumping rules. In J. M. Bernardo et al. (Eds.), *Bayesian Statistics 5*, 599-607. Oxford University Press. | adaptive proposal scaling $2.38^2/d$ |
| Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. *Bernoulli*, 7, 223-242. | [adaptive proposals](inv/sampler.md#3-adaptive-proposals) |
| Geyer, C. J. (1991). Markov chain Monte Carlo maximum likelihood. *Computing Science and Statistics: Proceedings of the 23rd Symposium on the Interface*, 156-163. | parallel tempering |
| Earl, D. J., & Deem, M. W. (2005). Parallel tempering: theory, applications, and new perspectives. *Physical Chemistry Chemical Physics*, 7, 3910-3916. | parallel tempering |
| Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16, 667-718. | effective sample size |
| Brocher, T. M. (2005). Empirical relations between elastic wavespeeds and density in the Earth's crust. *Bulletin of the Seismological Society of America*, 95, 2081-2092. | `brocher2005` Vp and density scaling |
| Gardner, G. H. F., Gardner, L. W., & Gregory, A. R. (1974). Formation velocity and density -- the diagnostic basics for stratigraphic traps. *Geophysics*, 39, 770-780. | `gardner1974` density scaling |

## Software

| software | used in |
|---|---|
| [disba](https://github.com/keurfonluu/disba) (K. Luu), a numba port of Herrmann's Computer Programs in Seismology surface-wave codes | dispersion and ellipticity forward models |
| [HV-DFA](https://github.com/agarcia-jerez/HV-DFA) | optional H/V reference solver |
| Fortran AFTAN and PyAFTAN | reference implementations for AFTAN ([AFTAN](ant/dispersion.md)) |
| PyGRT | independent check of the synthetic receiver functions |
