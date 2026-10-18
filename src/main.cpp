#include <mujoco/mujoco.h>

#include <stdio.h>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: test <model.xml>\n");
        return 1;
    }

    char error[1024] = {};

    mjModel* model = mj_loadXML(argv[1], nullptr, error, sizeof(error));
    if (!model) {
        std::fprintf(stderr, "MuJoCo error: %s\n", error);
        return 1;
    }

    mjData* data = mj_makeData(model);
    if (!data) {
        mj_deleteModel(model);
        return 1;
    }

    for (int i = 0; i < 1000; ++i) {
        mj_step(model, data);
    }

    std::printf("Simulation successful.\n");
    std::printf("Simulation time: %.3f s\n", data->time);

    mj_deleteData(data);
    mj_deleteModel(model);

    return 0;
}