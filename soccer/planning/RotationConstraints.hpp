#pragma once

#include <Configuration.hpp>

struct RotationConstraints {
public:
    RotationConstraints()
        : maxSpeed(*_max_rotation_speed),
          maxAccel(*_max_rotation_acceleration) {}
    double maxSpeed;
    double maxAccel;

    static void createConfiguration(Configuration* cfg);
    static ConfigDouble* _max_rotation_speed;
    static ConfigDouble* _max_rotation_acceleration;
};