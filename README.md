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