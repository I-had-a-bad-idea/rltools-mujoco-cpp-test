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

From PowerShell on Windows:

```bash
.\build\Release\test.exe models\cartpole.xml
```

On Linux or macOS:

```bash
./build/test models/cartpole.xml
```