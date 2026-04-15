Problem: While the hw2d regression training (tpflow/apps/05_train_regression.py) somewhat works, the rollouts are often unstable. Maybe a parameter sweep fixes this, maybe this is a deeper problem.
Potential sub-problem: I found out in the raw-data there are some trajectories that are completely zero. Does this destroy anything in the scaling and then downstream it doesn't make sense anymore?

If I need to re-run the whole pipeline, let me know.
