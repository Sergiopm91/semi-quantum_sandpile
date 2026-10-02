from semisandpile.config import load_config
from semisandpile.io import save_result
from semisandpile.simulation import simulate

cfg = load_config("configs/classical.yaml")
result = simulate(cfg)
out = save_result(result)
print(result.summary())
print("Saved to", out)
