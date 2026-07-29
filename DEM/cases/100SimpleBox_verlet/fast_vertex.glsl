#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in float aRadius;

uniform mat4 view;
uniform mat4 projection;
uniform float viewportHeight;

void main()
{
    vec4 viewPos = view * vec4(aPos,1.0);
    gl_Position = projection * viewPos;

    float viewZ = -viewPos.z;

    float projScale = projection[1][1];

    gl_PointSize =
        aRadius * projScale * viewportHeight / viewZ;
}
