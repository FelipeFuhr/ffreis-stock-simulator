---
paths:
  - "src/stock_simulator/types.py"
  - "src/stock_simulator/orders.py"
---

## Feature vector shape

- **Feature vector shape is 11 features.** This is enforced by `ml/ffreis-integration-hub`
  smoke tests. Changing the observation space breaks the RL agent without a compile error.
