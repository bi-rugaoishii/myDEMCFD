#version 330 core
//phong lighting
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform float opacity;

void main() {
    vec3 norm= normalize(Normal);
    vec3 lightColor = vec3(1.0);
    vec3 viewDir = normalize(viewPos -FragPos);

    //Ambient 
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * lightColor ;

    //Diffuse
    vec3 lightDir = viewDir;
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    //Specular 
    float specularStrength = 0.5;
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength*spec*lightColor;



    vec3 result = (ambient + diffuse + specular)*objectColor;
    FragColor = vec4(result,opacity);
}
