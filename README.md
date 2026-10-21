# rltools-mujoco-cpp-test
C++ prototype for integrating RLTools with Mujoco


## Building

Initialize the dependency-submodules:

```bash
make init
```

Configure and build the application (MuJoCo is built automatically):

```bash
make configure
```

On Windows, the build copies `mujoco.dll` beside `test.exe` so the executable can load it.
You can also configure and build with `make`:

```bash
make build
```

## Running

Train a policy, saving its weights to `policy.dat`:

```bash
.\build\Release\test.exe models\cartpole.xml train 100 policy.dat
```

Run/evaluate a previously trained policy:

```bash
.\build\Release\test.exe models\cartpole.xml run 5 policy.dat
```

When using the default five evaluation episodes, the policy path can be supplied
directly after `run`:

```bash
.\build\Release\test.exe models\cartpole.xml run policy.dat
```

The episode count and policy path are optional. Training defaults to 100 episodes;
evaluation defaults to 5. If no mode is specified, the executable trains a new policy.
Training uses RLTools' Twin Delayed DDPG (TD3) algorithm with the included MuJoCo
model. It observes cart position and velocity plus pole angle and angular velocity,
uses normalized cart-force actions, and randomizes the initial state near upright.
Episodes end when the pole falls, the cart leaves the track, or the 1000-step limit
is reached. Policies are saved as binary files compatible with this network's
architecture. Existing policies from older versions are not compatible; train a
new policy before evaluating them, and retrain after changing the layer dimensions.
The reported MuJoCo simulation time accumulates across episode resets; training and
evaluation also print wall-clock time, which includes RL inference and training work.