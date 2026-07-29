#version 330 core
out vec4 FragColor;
uniform vec3 objectColor;

void main()
{
    // 点内部の座標 [-1,1]
    vec2 uv = gl_PointCoord * 2.0 - 1.0;

    float r2 = dot(uv, uv);

    // 円の外側を削除
    if(r2 > 1.0)
        discard;

    // 疑似球の法線を作る
    vec3 normal;
    normal.xy = uv;
    normal.z  = sqrt(1.0 - r2);

    // SphereModeに近いライト方向
    vec3 lightDir = normalize(vec3(3.0,3.0,3.0));

    float diffuse = max(dot(normal, lightDir),0.0);

    // 少し環境光を足す
    float ambient = 0.2;

    vec3 color = vec3(ambient + diffuse*0.8);

    FragColor = vec4(color*objectColor,1.0);
}
