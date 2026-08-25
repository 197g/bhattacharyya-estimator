Compute a minimum Hellinger distance, or equivalently an upper bound of the
Bhattacharyya Coefficient, between an empirical sample and an analytical
continuous distribution. There's two main methods enabled by this which are
documented as follows.

## A p-value test

The library does not estimate the coefficient directly, for this see [Ding, R.,
& Mullhaupt, A. (2023)](https://doi.org/10.3390/e25040612).

Rather the library provides a rigorous p-value test where you first choose a
confidence level. The library then computes a bound that holds with the chosen
probability with one of two computational paths. A quick but rough estimate can
be given directly based on the Dvoretzky–Kiefer–Wolfowitz bounds of the
empirical data where the library is responsible for ensuring (slow) convergence
of this estimate with larger sample sizes. Alternatively, it also provides a
constraint-solver based bound within the function space of all CDFs that fit
inside the DKW bounds (runs in O(n³) worst case).

You can derive also probabilistic lower bounds of the total variational
distance and Kullback-Leibler divergence from this result using a number of
existing results in literature.

## An E-value test

The library can also flip this around. Instead of choosing a confidence level
you can give a maximum Hellinger distance of interest and the library then
computes a random value with expected value of at most `1` if the bound holds,
being evidence against the hypothesized value if the returned value is larger.

It does so by computing a large-as-possible total variational distance band
around the empirical distribution for which it can demonstrate, with either of
the above computations, to *conflict* with the maximum bound. (The library does
bisection search for the minimum width it can find). This band width is a
derived random variable. For the hypothesis to hold, the unknown true
distribution is at least that far away from the empirical data, which has a
bounded probability according to DKW. Thus we can derive an expected value
upper bound for the width of this band, and turn the computed value into a
random variable with expectation at most `1`.

# Mix of distributions

A corollary of the above allows dealing with an unknown mixed distribution
where a finite set of underlying basis distributions is known, i.e.
constraining the point on the coefficient simplex. You provide known analytical
Hellinger distances between pairs of such mixed distributions and their
respective simplex coordinates. Then, any significant minimum distance between
the sample and the base points also implies a linear constraint in the simplex.

Please note this is not particularly powerful (yet?) and the bounds are not
tight at all. Also you should be really careful with interpretation as this
evaluates multiple p-value tests and the implementation does not properly
correct the levels. (Please contribute this).

# Scope and non-goals

There is no implementation for computing the inverse bound, i.e. a non-trivial
maximum Hellinger distance. Indeed, without further assumptions on smoothness
this is not possible for empirical samples as the underlying PDF might always
be large set of yet-to-be-found dirac-delta points. Such a ragged PDF would
have a distance of `1` to any other not sharing its ideal components.

Also, all interfaces will at least indirectly deal with an empirical sample.
Computing analytical distances for families of distributions is left to other
packages, e.g. `statrs`.
