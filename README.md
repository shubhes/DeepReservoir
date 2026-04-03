# DeepReservoir
Repository to create Reinforcement Learning (RL) based Artificial Intelligence (AI) agents for autonomous management of reservoir operations




# Packages Used

# Models

# Data
This project uses data from the U.S. Bureau of Reclamation (USBR). Data can be accessed through the following link:

[USBR- Reservoir Data](https://github.com/shubhes/DeepReservoir/blob/main/data/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv)

# Codes

# Results

# Deliverable Model

# Experiments Conducted

| Experiment ID | Purpose | Agent | Action Space |Train-Test Split | Reward Function | Episode Length| Number of Episodes|Scaling|
|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 1 | - | DQN | Discrete | N/A | Binary (±1) | 120 | 20000 | N/A |
| 2 | - | DQN | Discrete | N/A | Binary (±1) | 120 | 20000 | N/A |
| 3 | - | PPO | Continuous | 18000-3000 | Binary (±1) | 120 | - | Max Scaling |
| 4 | - | PPO | Continuous | 18000-3000 | Binary (±1) | 360 | 5000 | Max Scaling |
| 5 | - | PPO | Continuous | 18000-3000 | Binary (±1) | 3600 | 500 | Max Scaling |
| 6 | - | TD3 | Continuous | 18000-3000 | Binary (±1) | 120 | - | Max Scaling |
| 7 | - | TD3 | Continuous | 18000-3000 | Binary (±1) | 3600 | 500 | Max Scaling |
| 8 | - | PPO | Continuous | 18000-3000 | Binary (±1) | 3600 | 400 | Mean-Std Standardization |


# Usage
Researchers and practitioners in the field of reservoir operations can utilize this repository for various purposes, including:

*Developing and testing reservoir control algorithms

*Investigating reservoir behavior under different operational conditions

# Citation

@misc{shubhesD73:online,
author = {},
  title = {shubhes/DeepReservoir: Repository for Reinforcement Learning codes to simulate reservoir operations},
  url = "https://github.com/shubhes/DeepReservoir",
month = {},
year = {},
  note = "[Online; accessed 2025-02-26]"
}

# License

This repository is provided under the Creative Commons **Attribution-NonCommercial-ShareAlike 4.0 International License**. Please review the license terms before using the content of the repository for any purpose other than non-commercial research and educational activities.

For more information, contact **shubhsingh@lanl.gov**, **jschwenk@lanl.gov**.
