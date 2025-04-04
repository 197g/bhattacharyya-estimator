## How to upper bound H²

Recall the bound `H² <= TV` and then do yourself the favor of checking out this
absolutely beautiful result on bounding that:

<https://arxiv.org/pdf/2202.07198> in particular the derivation of result (5)
via TFL in Section 5 from Hao-Chung Chen. Pure beauty "… that just works".

> Sidenote: Although, after having seen the derivation of Donsker-Varadhan, the
> paper does put the stranger formulation of the inequality first. `DK(p||q) >=
> Ep[log g] - log Eq[g]` is conceptually much clearer and the replacement where
> `f := log u` is only for practicality of specific expected values; also see
> how the stronger bound is found by using Jensen on the *other* portion of the
> sum as in the weaker bound is Hoeffding's Lemma. Curios.

For a full derivation on Donsker-Varadhan, this blog has it and does all the
transformations: <https://burklight.github.io/pages/blog/gv-and-dv.html>.

> Sidenote: the change in measure would be clearer in my opinion if it were
> employed after seperating the integrals of `log u`, avoiding the weird change
> back. I'm unsure as to why that's being done? It would seem to me like that
> almost slight-of-hand like trick of extracting a DK term from any measured
> logarithm was the path to finding that Lemma in the first place.
