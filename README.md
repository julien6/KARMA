# KARMA: Kubernetes Autoscaling with Resilient Multi-Agent system 

> ⚠️ **Warning: This project is a work in progress.**
>
> Some features may be incomplete or non-functional as development is currently ongoing. Use with caution and feel free to report any issues.


KARMA is a framework designed to simulate and manage microservice architectures using Kubernetes clusters. It supports Multi-Agent Reinforcement Learning (MARL) with modeling, training, and deployment pipelines. KARMA is built for scalability, adaptability, and robust simulation of cluster's problems like bottlenecks, crashes, and high workloads.


## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)


## **Overview**

KARMA provides:
- A **mock microservice generator** to simulate Kubernetes-based microservices.
- A **custom MARL training framework** integrated with [PettingZoo](https://www.pettingzoo.ml/) for simulating agent interactions.
- Tools for **modeling, training, analysis, and deployment** of multi-agent systems in a Kubernetes environment.
- Support for **organizational constraints** (using MOISE+ principles) to guide MARL training.

### Core Components
- **Analyzer**: Extracts roles, missions, and organizational insights from trained agents.
- **Modeler**: Models state transitions using a hybrid approach (exact transitions + MLP-based approximation).
- **Trainer**: Manages MARL training with MOISE+ roles and missions.
- **Transferer**: Deploys trained agent policies into live Kubernetes clusters.
- **Utils**: Utilities for generating mocks, manifests, and clusters.


## **Features**

- **Mock Microservices**:
  - Generate microservices using `Flask` for HTTP APIs.
  - Simulate resource consumption (CPU, RAM).
  - Supports customizable behavior (linear or non-linear resource usage).

- **Kubernetes Integration**:
  - Generates Kubernetes manifests (deployments, services).
  - Automates cluster creation using [Kind](https://kind.sigs.k8s.io/).
  - Monitors and applies policies to live clusters.

- **Reinforcement Learning**:
  - MARL training via [RLlib](https://docs.ray.io/en/latest/rllib.html) or [MARLlib](https://github.com/Replicable-MARL/MARLlib).
  - Integrated with PettingZoo environments.
  - Allows organizational constraints (MOISE+) to guide training.

- **Continuous Deployment**:
  - Real-time agent deployment in live clusters.
  - Handles crashes and updates policies dynamically.


## **Installation**

### Prerequisites
Ensure the following tools are installed:
- **Docker**: For containerized services.
- **kubectl**: To interact with Kubernetes clusters.
- **Kind**: Kubernetes in Docker for local clusters.
- **Helm**: Kubernetes package manager.
- **Miniconda**: For managing Python environments.

### Install KARMA Tools
Run the provided installation script:
```bash
bash install.sh
```
This installs Docker, kubectl, Kind, Helm, and Miniconda.

---

## **Usage**

### 1. Define a Topology
Create a `topology.json` file describing the microservices, connections, and resource requirements. Example:
```json
{
  "services": {
    "service1": { "cpu": 500, "ram": 256, "computation_throughput": 100 },
    "service2": { "cpu": 1000, "ram": 512, "computation_throughput": 50 }
  },
  "connections": [
    { "source": "service1", "destination": "service2" }
  ]
}
```

### 2. Generate and Deploy the Cluster
Use the `KarmaCluster` class to manage the cluster setup:
```python
from utils.cluster_util import KarmaCluster

kc = KarmaCluster("topology.json")
kc.check_prerequisites()
kc.generate_mocks()
kc.build_docker_images()
kc.generate_manifests()
kc.create_cluster()
kc.wait_for_pods_ready()
```

### 3. Train Agents
Leverage the `trainer` module to train agents in the mock environment. Example:
```python
from components.trainer.trainer import MARLTrainer

trainer = MARLTrainer("config.yaml", "mock_environment")
trainer.train_agents(episodes=100)
trainer.save_policies("policies/")
```
