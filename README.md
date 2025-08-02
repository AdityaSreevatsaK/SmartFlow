# <p align="centre">SmartFlow: Reinforcement Learning and Agentic AI for Bike Sharing Optimisation</p>

---

## Objective

SmartFlow is a novel solution for enhancing the efficiency of bike-sharing systems through the integration of deep
reinforcement learning and agentic AI. The RL model learns optimal actions to balance supply and demand across stations,
while agentic components handle real-time notifications to ground personnel, such as truck drivers. This dual approach
ensures both intelligent decision-making and timely execution, offering a scalable and adaptive strategy for modern
urban transport networks.

---

## Overview

SmartFlow is a **hybrid framework** designed to optimise urban bike-sharing operations. It combines deep reinforcement
learning (RL) with agentic AI components to dynamically learn bike movement patterns, anticipate operational demand, and
recommend optimal bike redistribution across city stations. The agentic layer autonomously communicates with truck
drivers and ground staff, enabling real-time, hands-free coordination and reducing the need for manual intervention.

#### This project aims to:

* Minimise bike idle time at stations
* Improve overall bike availability and user satisfaction
* Lower operational costs for fleet management
* Support scalable, adaptable solutions for large, complex urban transport networks

---

## Key Features

* **Deep Reinforcement Learning:**
  Learns station-level redistribution policies from real trip data and simulated demand, outperforming static and
  rule-based rebalancing methods.

* **Agentic AI Layer:**
  Automates operational notifications, assigning tasks (such as truck routing or inventory checks) to ground staff or
  autonomous agents.

* **Simulator-in-the-Loop:**
  Enables closed-loop feedback, allowing for dynamic adaptation to real-time changes in demand and traffic.

* **Flexible Integration:**
  Designed for compatibility with public data feeds, real-time APIs, and diverse urban infrastructures.

---

## Motivation

Despite the global adoption of bike-sharing systems, maintaining an optimal, real-time distribution of bikes remains a
major challenge. Traditional rebalancing methods are often static or manually scheduled, making them unable to adapt to
fluctuations in user demand or city events. This results in stations being overcrowded or empty, inefficient resource
allocation, increased operational costs, and ultimately, lower user satisfaction.

SmartFlow addresses these limitations by enabling adaptive, data-driven, and automated fleet management that aligns with
sustainability and smart city goals.

---

## Repository Structure

* `data/` – Raw and processed data
* `docs/` – Paper drafts, presentations, reports
* `images/` – Images for the report
* `references/` – Research papers, external PDFs
* `results/` – Output files, visualisations, model logs
* `src/` – Notebooks, scripts, modules

---

## Getting Started

1. **Clone the repository and install dependencies:**

   ```bash
   git clone https://github.com/yourusername/SmartFlow.git
   cd SmartFlow
   pip install -r requirements.txt
   ```

2. **Place your raw trip data** in `data/raw/`.
   Update `src/constants.py` if file paths change.

3. **Run training or simulation scripts** from the `src/` folder.

---

## Citation

If you use SmartFlow in your research or deployment, please cite:

> Aditya Sreevatsa K\*, Arun Kumar Raveendran, Jesrael K Mani, Prakash G Shigli, Rajkumar Rangadore.
> "SmartFlow: Reinforcement Learning and Agentic AI for Bike-Sharing Optimisation"

---

## Tasks

See [TODO.md](TODO.md) for planned features, bug tracking, and future work.

---
