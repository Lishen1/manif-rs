# manif
## _A small library for Lie theory_

Unofficial partial port of [manif]
>**Warning**
>**WIP**
## TODO
- [ ] impl SO2
- [ ] impl SE2
- [ ] impl SE3
- [ ] move common tests to macroses
- [ ] move some impl into trait defaults
- [ ] add docs
### Available Operations

| Operation  |       | Code |
| :---       |   :---:   | :---: |
|       |   Base Operation   |  |
| Inverse | $\mathbf\Phi^{-1}$ | `X.inverse()` |
| Composition | $\mathbf{\mathcal{X}}\circ\mathbf{\mathcal{Y}}$ | `X * Y`<br/>`X.compose(Y)` |
| Hat | $\varphi^\wedge$ | `w.hat()` |
| Act on vector | $\mathbf{\mathcal{X}}\circ\mathbf v$ | `X.act(v)` |
| Retract to group element | $\exp(\mathbf\varphi^\wedge)$ | `w.exp()` |
| Lift to tangent space | $\log(\mathbf{\mathcal{X}})^\vee$ | `X.log()` |
| Manifold Adjoint | $\operatorname{Adj}(\mathbf{\mathcal{X}})$ | `X.adj()` |
| Tangent adjoint | $\operatorname{adj}(\mathbf{\varphi^\wedge})$ | `w.smallAdj()` |
|       |   Composed Operation   |  |
| Manifold right plus | $\mathbf{\mathcal{X}}\oplus\mathbf\varphi = \mathbf{\mathcal{X}}\circ\exp(\mathbf\varphi^\wedge)$ | `X + w`<br/>`X.plus(w)`<br/>`X.rplus(w)` |
| Manifold left plus | $\mathbf{\mathbf\varphi\oplus\mathcal{X}} = \exp(\mathbf\varphi^\wedge)\circ\mathbf{\mathcal{X}}$ | `w + X`<br/>`w.plus(X)`<br/>`w.lplus(X)` |
| Manifold right minus | $\mathbf{\mathcal{X}}\ominus\mathbf{\mathcal{Y}} = \log(\mathbf{\mathcal{Y}}^{-1}\circ\mathbf{\mathcal{X}})^\vee$ | `X - Y`<br/>`X.minus(Y)`<br/>`X.rminus(Y)` |
| Manifold left minus | $\mathbf{\mathcal{X}}\ominus\mathbf{\mathcal{Y}} =\log(\mathbf{\mathcal{X}}\circ\mathbf{\mathcal{Y}}^{-1})^\vee$ | `X.lminus(Y)` |
| Between | $\mathbf{\mathcal{X}}^{-1}\circ\mathbf{\mathcal{Y}}$ | `X.between(Y)` |
| Inner Product | $\langle\varphi,\tau\rangle$ | `w.inner(t)` |
| Norm | $\left\lVert\varphi\right\rVert$ | `w.weightedNorm()`<br/>`w.squaredWeightedNorm()` |

Above, $\mathbf{\mathcal{X}},\mathbf{\mathcal{Y}}$ represent group elements,
$\mathbf\varphi^\wedge,\tau^\wedge$ represent elements in the Lie algebra of the Lie group,
$\mathbf\varphi,\tau$ or `w,t` represent the same elements of the tangent space
but expressed in Cartesian coordinates in $\mathbb{R}^n$,
and $\mathbf{v}$ or `v` represents any element of $\mathbb{R}^n$.
### Jacobians

All operations come with their respective analytical Jacobian matrices.
Throughout **manif**, **Jacobians are differentiated with respect to a local perturbation on the tangent space**.
These Jacobians map tangent spaces, as described in [this paper][jsola18].

Currently, **manif** implements the **right Jacobian**, whose definition reads:

$\frac{\delta f(\mathbf{\mathcal{X}})}{\delta\mathbf{\mathcal{X}}}\triangleq\displaystyle{\lim_{\mathbf\varphi\to0}}\frac{f(\mathcal{X}\oplus\mathbf\varphi)\ominus f(\mathcal{X})}{\mathbf\varphi}\triangleq \displaystyle{\lim_{\mathbf\varphi\to0}}\frac{log(f(\mathcal{X})^{-1}f(\mathcal{X}\exp(\mathbf\varphi^\wedge)))^{\vee}}{\mathbf\varphi}$

The Jacobians of any of the aforementionned operations can then be evaluated:
## License
GNU GPLv3

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [manif]: <https://github.com/artivis/manif#a-small-header-only-library-for-lie-theory>
   [jsola18]: http://arxiv.org/abs/1812.01537

