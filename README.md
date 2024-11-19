# k8s-hpa

The implemented k8s-hpa is a custom [OpenAi Gym](https://gym.openai.com/) 
environment for the training of Reinforcement Learning (RL) agents for auto-scaling research 
in the Kubernetes (K8s) platform. 


## How does it work?

Three RL environments have been designed for actions, observations, reward function:

 - **Redis**: based on the [Redis Cluster](https://github.com/bitnami/charts/tree/master/bitnami/redis-cluster) application;
 - **Online Boutique**: based on the [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) application;
 - **Chained Services**: based on a K8s cluster comprises interdependent chained services where bottlenecks are to be minimized when input data is fluctuating.


Please check the [run.py](policies/run/run.py) file to understand how to run the framework. 

To run in the real cluster mode, you should add the token to your cluster [here](k8s_hpa/envs/deployment.py)

## Usage

To run the script, use the following command:

```sh
python run.py [OPTIONS]
```

## Command-Line Options

### Algorithm Selection

- `--alg`: Specifies the algorithm to use. Possible values are:
  - `ppo`: Proximal Policy Optimization
  - `recurrent_ppo`: Recurrent Proximal Policy Optimization
  - `a2c`: Advantage Actor-Critic

  **Default**: `ppo`

  ```sh
  python run.py --alg ppo
  ```

### Kubernetes Mode

- `--k8s`: Enables Kubernetes mode. When this flag is set, the script will interact with a real Kubernetes cluster.

  **Default**: `False`

  ```sh
  python run.py --k8s
  ```

### Use Case

- `--use_case`: Specifies the use case to run. Possible values are:
  - `redis`: Redis use case
  - `online_boutique`: Online Boutique use case
  - `chained_services`: Chained Services use case

  **Default**: `chained_services`

  ```sh
  python run.py --use_case online_boutique
  ```

### Reward Goal

- `--goal`: Specifies the reward goal. Possible values are:
  - `cost`: Optimize for cost
  - `latency`: Optimize for latency
  - `multi`: Optimize both cost and latency with Langragian relaxation

  **Default**: `latency`

  ```sh
  python run.py --goal cost
  ```

### Training Mode

- `--training`: Enables training mode. When this flag is set, the script will train the model.

  **Default**: `False`

  ```sh
  python run.py --training
  ```

### Testing Mode

- `--testing`: Enables testing mode. When this flag is set, the script will test the model.

  **Default**: `False`

  ```sh
  python run.py --testing
  ```

### Loading Mode

- `--loading`: Enables loading mode. When this flag is set, the script will load a pre-trained model and resume training.

  **Default**: `False`

  ```sh
  python run.py --loading
  ```

### Load Path

- `--load_path`: Specifies the path to the pre-trained model to load. This option is used in conjunction with the `--loading` flag.

  **Default**: `logs/model/test.zip`

  ```sh
  python run.py --loading --load_path logs/model/my_model.zip
  ```

### Test Path

- `--test_path`: Specifies the path to the model to test. This option is used in conjunction with the `--testing` flag.

  **Default**: `logs/model/test.zip`

  ```sh
  python run.py --testing --test_path logs/model/my_model.zip
  ```

### Steps for Saving

- `--steps`: Specifies the number of steps for saving checkpoints during training.

  **Default**: `500`

  ```sh
  python run.py --steps 1000
  ```

### Total Steps

- `--total_steps`: Specifies the total number of steps for training.

  **Default**: `5000`

  ```sh
  python run.py --total_steps 10000
  ```

## Example Commands

### Train a PPO Model for Redis Use Case

```sh
python run.py --alg ppo --use_case redis --goal cost --training --total_steps 10000
```

### Test a Pre-trained Model for Online Boutique Use Case

```sh
python run.py --use_case online_boutique --goal latency --testing --test_path logs/model/online_boutique_model.zip
```

### Resume Training a Pre-trained Model

```sh
python run.py --alg ppo --use_case redis --goal cost --loading --load_path logs/model/redis_model.zip --training --total_steps 5000
```

### Train a Recurrent PPO Model in Kubernetes Mode

```sh
python run.py --alg recurrent_ppo --use_case online_boutique --goal latency --k8s --training --total_steps 15000
```


## Contact

If you want to contribute, please contact:

Lead developer: [Julien Soulé](https://github.com/julien6)

For questions or support, please use GitHub's issue system.
