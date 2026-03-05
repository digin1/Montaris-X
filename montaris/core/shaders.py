"""Optional GLSL shaders for GPU-accelerated image adjustments.

These are provided for future OpenGL rendering integration.
Currently the application uses numpy-based adjustments (see adjustments.py).
"""

ADJUSTMENT_VERTEX_SHADER = """
#version 330
in vec2 position;
in vec2 texcoord;
out vec2 v_texcoord;
uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

ADJUSTMENT_FRAGMENT_SHADER = """
#version 330
in vec2 v_texcoord;
out vec4 fragColor;

uniform sampler2D image;
uniform float brightness;
uniform float contrast;
uniform float exposure;
uniform float gamma;

void main() {
    vec4 color = texture(image, v_texcoord);
    vec3 rgb = color.rgb;

    // Exposure
    rgb *= pow(2.0, exposure);

    // Contrast
    rgb = (rgb - 0.5) * contrast + 0.5;

    // Brightness
    rgb += brightness;

    // Gamma
    rgb = clamp(rgb, 0.0, 1.0);
    rgb = pow(rgb, vec3(1.0 / gamma));

    fragColor = vec4(rgb, color.a);
}
"""
