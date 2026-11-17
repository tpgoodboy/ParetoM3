# ParetoM$^3$
The implementation of "[ParetoM$^3$: Learning on the Pareto Set under Preference Guidance via Min‑Max‑Min Optimization](https://arxiv.org/abs/2611.18252)", NeurIPS 2026. 

We provide the codes for experiments on the MultiMNIST dataset.

To run the code, execute the command 
"
python linscalar_train.py / epo_train.py / ferero_train.py / paretom3_train.py
"

For the penalty parameter $\rho$, while theoretical convergence requires a sufficiently large $\rho$, a moderate value yields satisfactory empirical performance. We tune $\rho$ using grid search: we first explore candidates $\{0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0\}$ to locate a coarse range and then conduct a finer-grained search.
