# SmartFlow

A **research simulation** for bike-sharing redistribution. It combines a reinforcement-learning inventory policy, journey planning and an LLM-generated operations briefing.

The repository demonstrates **simulated planning and reporting**, not a deployed real-time bike-sharing service or automated communication with drivers.

## Implemented components

| Component | Source |
| --- | --- |
| Redistribution environment | [`src/library_based/environment.py`](src/library_based/environment.py) |
| Model training and loading | [`src/library_based/train.py`](src/library_based/train.py) |
| Journey routing and scheduling | [`src/library_based/routing.py`](src/library_based/routing.py) |
| Simulation and output generation | [`src/library_based/run_simulation.py`](src/library_based/run_simulation.py) |
| Human-readable dispatch briefing | [`src/library_based/agent.py`](src/library_based/agent.py) |

The reporting module uses a Hugging Face text-generation model to prepare a manager briefing and produces dispatch tables from the underlying journey data. If text generation fails, a deterministic report is used. It **does not send real notifications or control vehicles**.

## Repository contents

- [Simulation source](src/library_based/) and earlier [Python exploration](src/vanilla_python/).
- [Input datasets](data/) and [example results](results/) for local experiments.
- [Architecture illustration](docs/SmartFlow%20-%20Architecture.png) and [environment specification](environment.yml).

## Reproducing the experiment

Review [`environment.yml`](environment.yml), [`constants.py`](src/library_based/constants.py) and the [simulation entry point](src/library_based/run_simulation.py) before running. Execution depends on the included datasets, compatible scientific-Python and machine-learning packages, optional model downloads, and available compute resources.

**Offline CI** runs six mocked dispatch-report regression tests, plus Python syntax checks, across Windows/Linux and Python 3.10/3.11. It verifies route fields, deterministic fallback and that remote model code loading is disabled. **It does not test Gemma model compatibility, the full RL simulation or clean-machine reproduction.** The committed outputs are experiment artefacts, not independent validation of performance improvements.

## Limitations and safety

- No measured real-world reduction in bike shortages, dispatch time or operating costs is established here.
- The source does not implement a driver messaging service or production operations controls.
- Model-generated briefing text can be inaccurate and must be checked against the deterministic dispatch data.
- Local loading of third-party models and datasets requires appropriate licences and a trusted environment. Remote custom model-code execution is disabled by default in the reporting module.
- Referenced research papers and datasets retain their respective rights and terms.

Licensed under [MIT](LICENSE).
