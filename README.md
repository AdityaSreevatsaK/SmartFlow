# <p align="center">SmartFlow: Reinforcement Learning and Agentic AI for Bike-Sharing Optimisation</p>

---

## Objective

SmartFlow introduces a novel solution for enhancing the efficiency of bike-sharing systems through the integration of
deep reinforcement learning and agentic AI. The RL model learns optimal policies to balance supply and demand across
stations, while the agentic components manage real-time communication with ground personnel, such as truck drivers. This
dual approach ensures that intelligent, data-driven decisions are translated into timely, real-world actions, offering a
scalable and adaptive strategy for modern urban transport networks.

---

## Overview

SmartFlow is a **hybrid framework** designed to optimise urban bike-sharing operations. It combines a deep reinforcement
learning (RL) model with an agentic AI layer to dynamically learn bike movement patterns, anticipate demand, and
recommend optimal bike redistribution strategies. The agentic layer autonomously communicates with truck drivers and
operational staff, enabling seamless, hands-free coordination and significantly reducing the need for manual
intervention.

#### This project aims to:

* Minimise bike idle time and station imbalances (overcrowding or emptiness).
* Improve overall bike availability and enhance user satisfaction.
* Lower operational costs associated with fleet management.
* Provide a scalable and adaptable solution for complex urban environments.

---

## Architecture

A high-level diagram illustrating the interaction between the RL core, the simulator, and the agentic AI layer.

<img src='docs/SmartFlow - Architecture.png'></img>

---

## Key Features

* **Deep Reinforcement Learning Core:**
  Learns optimal, state-aware redistribution policies from historical trip data and simulated demand, outperforming
  static and heuristic-based rebalancing methods.

* **Agentic AI Communication Layer:**
  Automates operational logistics by issuing clear, actionable tasks (e.g., truck routing, inventory checks) to ground
  staff via autonomous communication agents.

* **Simulator-in-the-Loop Training:**
  Utilises a digital twin of the urban environment to train, test, and validate policies safely, allowing for dynamic
  adaptation to real-time changes in demand and traffic before deployment.

* **Modular and Extensible:**
  Designed for flexible integration with public data feeds, real-time APIs, and diverse urban infrastructures, ensuring
  broad applicability.

---

## Motivation

Despite the global success of bike-sharing systems, maintaining an optimal distribution of bikes in real-time remains a
significant operational challenge. Traditional rebalancing methods are often static or manually scheduled, rendering
them unable to adapt to the dynamic fluctuations of user demand, traffic congestion, or city-wide events. This
inefficiency leads to underutilised assets, increased operational costs, and a diminished user experience.

SmartFlow directly addresses these limitations by enabling an adaptive, data-driven, and automated approach to fleet
management that aligns with modern smart city objectives.

---
### Technology Stack

The SmartFlow framework is built on a specific stack of modern libraries for data science, reinforcement learning, and agentic AI.

* **Core ML & Data Science:** Python, Pandas, NumPy, Scikit-learn
* **Deep Learning Backend:** PyTorch
* **Reinforcement Learning:** Stable-Baselines3 (DQN Algorithm), Gymnasium (for the environment API)
* **Agentic AI & NLP:** Hugging Face (Transformers, Accelerate, BitsAndBytes) with the `google/gemma-2b-it` model
* **Geospatial & Routing:** OSMnx, NetworkX
* **Visualisation & Mapping:** Matplotlib, Seaborn, Folium, TensorBoard

---

## Repository Structure

```
.
├── data/                 # Raw and processed datasets
│   ├── processed/
│   └── raw/
├── docs/                 # Project documentation, papers, and reports
├── images/               # Diagrams and visual assets for documentation
├── references/           # Supporting research papers and code snippets
│   ├── code-snippets/
│   └── papers/
├── results/              # Model outputs, logs, and performance visualisations
│   ├── models/
│   ├── other_metrics/
│   ├── plots/
│   ├── rewards/
│   └── simulation/
├── src/                  # Source code for different implementation approaches
│   ├── library_based/    # Implementations using existing RL/AI libraries
│   ├── snippets/         # Reusable code snippets for development
│   └── vanilla_python/   # Core logic implemented in standard Python
├── .env                  # Environment variables configuration
├── .gitignore            # Files and directories to be ignored by Git
├── environment.yml       # Conda environment dependencies
├── LICENSE               # Project software license
├── README.md             # This file
└── TODO.md               # Development roadmap and tasks
```

---

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AdityaSreevatsaK/SmartFlow.git
   cd SmartFlow
   ```

2. **Create and activate the Conda environment:**
   The `environment.yml` file contains all the necessary dependencies. Create the environment using this file.

   ```bash
   conda env create -f environment.yml
   ```

   This will create a new environment with the name specified inside the file (e.g., `smartflow_env`). Activate it:

   ```bash
   conda activate smartflow_env
   ```

3. **Prepare the data:**
   Place your raw trip data in the `data/raw/` directory. Raw data can be processed and cleaned for modelling using the
   dedicated [**SmartFlow-Prep pipeline**](https://github.com/AdityaSreevatsaK/SmartFlow-Prep).

4. **Run the scripts:**
   Execute the desired training or simulation scripts located within the `src/` subdirectories.

---

### Citation

If you use SmartFlow in your research, please cite it as follows. For in-text citations, please use the format (
Sreevatsa K et al., 2025).

**BibTeX Entry:**

```bibtex
@unpublished{sreevatsa2025smartflow,
  author    = {Sreevatsa K, Aditya and Raveendran, Arun Kumar and Mani, Jesrael K and Shigli, Prakash G and Rangadore, Rajkumar},
  title     = {{SmartFlow: Reinforcement Learning and Agentic AI for Bike-Sharing Optimisation}},
  note      = {Manuscript in preparation},
  year      = {2025},
  month     = {aug}
}
```

---

## Licensing

The source code of SmartFlow is licensed under the **MIT License**. You can find the full license text in the `LICENSE`
file.

The accompanying research paper is licensed under a
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**, as required by the conference publisher.

---

## Project Status

See `TODO.md` for our development roadmap, planned features, and bug tracking.

---
