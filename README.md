# Pricing-Barrier-Options

Certain types of exotic options are cheaper than standard vanilla options, because a zero payoff
may occur before expiry. The pricing of path dependent exotic options such as barrier options is
not straight forward using analytical formulae. The objective of this project is to compare pricing
of the barrier options using standard Black Scholes model and Monte Carlo simulations. This
project compares the two variance reduction techniques such as Antithetic Variates and Control
Variates with crude Monte Carlo simulation in pricing the barrier options and the convergence
with the analytical solution as we increase the number of simulations. Also, we look at the impact
of barrier price on the different variance reduction methods. From the computational results, we
find that the antithetic variates method substantially improves the precision of Monte Carlo
estimates. Our results also show that using the ordinary put option as a control variate significantly
reduces standard errors.
