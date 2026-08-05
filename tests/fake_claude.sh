#!/bin/bash
# Offline stand-in for the claude CLI: consumes stdin, emits one valid recipe.
cat > /dev/null
cat <<'EOF'
Here is an improved recipe.

```python
import math

OPTIMIZER = {"sigma0": 0.4, "popsize": 12}


def schedule(gen):
    return {"episodes_per_eval": 6, "max_steps": 400}


def shaped_reward(obs, action, reward, terminated, truncated, step):
    return reward + 0.001 * math.cos(step * 0.01)
```
EOF
