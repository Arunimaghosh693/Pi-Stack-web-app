# How to activate the repository scripts in your shell

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
