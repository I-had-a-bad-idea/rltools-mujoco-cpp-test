#include "mujoco_env.hpp"

void MujocoEnv::load(const char* xml_path) {
    char error[1024] = {};
    mjModel* model = mj_loadXML(xml_path, nullptr, error, sizeof(error));
    if (!model) {
        throw std::runtime_error(error);
        return;
    }

    mjData* data = mj_makeData(model);

    slider_id = mj_name2id(model, mjOBJ_JOINT, "slider");
    hinge_id = mj_name2id(model, mjOBJ_JOINT, "hinge");

    if (!data || slider_id < 0 || hinge_id < 0) {
        throw std::runtime_error("Mujoco data or joints could not be loaded");
        mj_deleteModel(model);
        return;
    }
}

Observation MujocoEnv::observe() const {
    const int sx = model->jnt_qposadr[slider_id];
    const int sv = model->jnt_dofadr[slider_id];
    const int ha = model->jnt_qposadr[hinge_id];
    const int hv = model->jnt_dofadr[hinge_id];

    return {{
        static_cast<float>(data->qpos[sx]),
        static_cast<float>(data->qvel[sv]),
        static_cast<float>(data->qpos[ha]),
        static_cast<float>(data->qvel[hv])
    }};
}

void MujocoEnv::step(float action) {
    data->ctrl[0] = action;
    mj_step(model, data);
}

float MujocoEnv::reward(float cart_x, float pole_angle, float action) {
    return 1.0f
        - 1.0f * pole_angle * pole_angle
        - 0.1f * cart_x * cart_x
        - 0.001f * action * action;
}

void MujocoEnv::close() {
    std::printf("Simulation complete.\n");
    std::printf("Simulation time: %.3f s\n", data->time);
    if (data) mj_deleteData(data);
    if (model) mj_deleteModel(model);

    data = nullptr;
    model = nullptr;
}