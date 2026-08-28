**π-Stack Optimizer** is a high-performance computational framework for discovering energetically favorable stacking configurations in molecular systems. Leveraging semi-empirical quantum chemistry (xTB) coupled with multiple global optimization algorithms (PSO, GA, GWO, PSO+Nelder–Mead), this tool provides researchers with a flexible, efficient, and reproducible workflow for π-stacking studies. The system features parallel energy evaluations, automatic symmetry detection, and comprehensive logging capabilities, making it suitable for both exploratory research and production computational chemistry pipelines.

**Authors:** 
Arunima Ghosh, Susmita Barik, Roshan J Singh, & Sandeep K. Reddy

**How to activate the repository scripts in your shell**

Source the activation script to get `pi-stack-generator` and `pi-hyperopt` available in your shell session.

For complete documentation, refer to `doc/manual.pdf`.

From the repo root:

```bash
source ./activate_pi_stack.sh
```

This does three things:
- Adds the project root to your `PATH` so scripts can be executed directly.
- Adds the project root to `PYTHONPATH` so `import modules.*` resolves.
- Defines two shell functions:
  - `pi-stack-generator` -> runs `pi-stack-generator.py` with the same args.
  - `pi-hyperopt` -> runs `hyperparameter-opt/hyperopt.py` with the same args.

If you prefer executable scripts instead of functions, make the script files executable:

```bash
chmod +x pi-stack-generator.py
chmod +x hyperparameter-opt/hyperopt.py
```

For more details, please refer to the corresponding publication. If you use this code in your work, we kindly request that you cite the following paper:

**Citation**

Ghosh, A., Barik, S., Singh, R. J, & Reddy, S. K. (2026). *π-Stack Optimizer: A high-performance computational framework for discovering energetically favorable stacking configurations in molecular systems.*

**BibTeX**

@article{ghosh2026pistack,
  title        = {π-Stack Optimizer: A high-performance computational framework for discovering energetically favorable stacking configurations in molecular systems},
  author       = {Ghosh, Arunima and Barik, Susmita and Singh, Roshan J and Reddy, Sandeep K.},
  year         = {2026}
}


