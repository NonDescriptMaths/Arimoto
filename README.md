# An implimentation of the Blahut-Arimoto Algorithm
As stated this repo is my own implimentation of the Blahut Arimoto algorithm in both Python and C++, I interfaced C++ with python via the `pybind11` package which probably wasn't the most efficient, should have used a `.sl` file for the data. At some point I'm hoping to come back to this project to improve its interface.

Key functions/files of the project are:
- The Arimoto functions: which impliment it in both python3 and C++, along with functions which automate it for repeated use
- Generator functions: which simply generate priors (e.g dirichlet prior generators for distributions, or random channel priors) for use in experiments
- Convergence Functions: which are for simulating repeated Arimoto under a variety of conditions and observe the outcome, the most useful one would be `simulate_distribution_ternary` which plots the density of arimoto solutions in the 3 simplex.
-  under either random or fixed channel matrices
