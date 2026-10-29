/* 点睛 · 修图  ——  WebGL2 单 pass 修图管线
 * 算法来源见 ALGORITHM_SPEC.md（与 Python / Metal 版同源）
 */
'use strict';

const $ = (s) => document.querySelector(s);

/* ---------------------------------------------------------------- shader */
const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main(){
  v_uv = vec2(a_pos.x * 0.5 + 0.5, a_pos.y * 0.5 + 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o;

uniform sampler2D u_img;
uniform sampler2D u_lut;
uniform sampler2D u_faceMask;   // 脸部皮肤蒙版（468 点画的，白=皮肤 黑=眼唇眉）
uniform sampler2D u_warp;       // 手动液化位移场（RG 编码 0..1 → -32..32 像素）
uniform vec2  u_size;        // 处理分辨率（像素）
uniform vec2  u_texel;       // 1/u_size

uniform float u_smooth;      // 0..1
uniform float u_whiten;      // 0..1
uniform float u_texture;     // 0..1 保留质感
uniform float u_lutAmount;   // 0..1
uniform int   u_hasLut;

uniform float u_exposure;    // -1..1
uniform float u_contrast;    // -1..1
uniform float u_saturation;  // -1..1
uniform float u_temp;        // -1..1
uniform float u_shadow;      // -1..1
uniform float u_highlight;   // -1..1

uniform int   u_hasFace;
uniform float u_slimFace;    // 0..1
uniform float u_eyes;        // 0..1
uniform float u_slimChin;    // 0..1 尖下巴
uniform float u_slimNose;    // 0..1 瘦鼻
uniform float u_smile;       // 0..1 微笑唇
uniform float u_lift;        // 0..1 提拉
uniform vec2  u_faceC;       // 像素坐标
uniform float u_faceW, u_faceH, u_zygoY, u_chinY;
uniform vec2  u_eyeL, u_eyeR;
uniform float u_eyeRad;
uniform vec2  u_mouthC;
uniform float u_mouthW, u_noseY;

uniform float u_unify;       // 0..1 匀肤
uniform float u_complexion;  // 0..1 气色
uniform float u_brightEye;   // 0..1 亮眼
uniform float u_lip;         // 0..1 唇彩
uniform float u_teeth;       // 0..1 牙齿美白
uniform float u_eyeWhite;    // 0..1 眼白
uniform float u_dehaze;      // 0..1 去雾
uniform float u_blur;        // 0..1 背景虚化
uniform float u_warpScale;   // 位移场像素→当前尺寸的换算（高清导出时 != 1）
uniform float u_softglow;    // 0..1 柔光
uniform float u_sharpen;     // 0..1 锐化

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

/* ---- 肤色蒙版：YCbCr 主判 + HSV 边界软化 ---- */
float skinMask(vec3 c){
  float Y  = dot(c, vec3(0.299, 0.587, 0.114));
  float Cb = dot(c, vec3(-0.168736, -0.331264, 0.5)) + 0.5;
  float Cr = dot(c, vec3(0.5, -0.418688, -0.081312)) + 0.5;
  // 77..133 / 255
  float cb = smoothstep(0.290, 0.318, Cb) * (1.0 - smoothstep(0.512, 0.540, Cb));
  // 133..180 / 255
  float cr = smoothstep(0.510, 0.538, Cr) * (1.0 - smoothstep(0.694, 0.722, Cr));
  float y  = smoothstep(0.150, 0.200, Y);
  float m = cb * cr * y;
  // 排除过暗（头发/眉毛）与高饱和（衣服）
  float sat = (max(max(c.r,c.g),c.b) - min(min(c.r,c.g),c.b)) / max(max(max(c.r,c.g),c.b), 1e-3);
  m *= 1.0 - smoothstep(0.55, 0.80, sat);
  return clamp(m, 0.0, 1.0);
}

/* ---- 表面模糊（surfaceBlur.frag 逐字复刻）---- */
vec3 surfaceBlur(vec2 uv, vec3 c, float th){
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  for (int y = -3; y <= 3; y++){
    for (int x = -3; x <= 3; x++){
      vec3 nb = texture(u_img, uv + vec2(float(x), float(y)) * u_texel).rgb;
      float d = dot(abs(nb - c), vec3(0.3333333));
      float w = clamp(1.0 - d / (2.5 * th), 0.0, 1.0);
      sum += nb * w;
      wsum += w;
    }
  }
  return sum / max(wsum, 1e-5);
}

/* ---- 3D LUT：8x8 格 × 64 采样，texel 中心 ---- */
vec3 lutSlice(vec2 rg, float sl){
  float gx = mod(sl, 8.0);
  float gy = floor(sl / 8.0 + 1e-4);
  vec2 uv = vec2((gx + (clamp(rg.r,0.0,1.0) * 63.0 + 0.5) / 64.0) / 8.0,
                 1.0 - (gy + (clamp(rg.g,0.0,1.0) * 63.0 + 0.5) / 64.0) / 8.0);
  return texture(u_lut, uv).rgb;
}
vec3 applyLut(vec3 c){
  float b  = clamp(c.b, 0.0, 1.0) * 63.0;
  float b0 = floor(b);
  float b1 = min(b0 + 1.0, 63.0);
  return mix(lutSlice(c.rg, b0), lutSlice(c.rg, b1), b - b0);
}

/* ---- 液化：像素空间反向映射 ---- */
vec2 eyeWarp(vec2 s, vec2 c, float r, float amt){
  vec2 d = s - c;
  float k = length(d) / max(r, 1.0);
  if (k < 1.0){
    float w = pow(1.0 - k * k, 2.0);
    return c + d * (1.0 - amt * 0.72 * w);
  }
  return s;
}

void main(){
  vec2 uv = v_uv;
  vec2 pix = vec2(uv.x * u_size.x, (1.0 - uv.y) * u_size.y);  // y 向下 = 图像坐标

  /* --- 1. 液化 --- */
  vec2 src = pix;
  // 手动液化：用户拖出来的位移场（RG 通道各存 -32..32 像素）
  vec2 manual = (texture(u_warp, v_uv).rg * 192.0 - 96.0) * u_warpScale;
  src += manual;
  if (u_hasFace > 0){
    if (u_eyes > 0.001){
      src = eyeWarp(src, u_eyeL, u_eyeRad, u_eyes);
      src = eyeWarp(src, u_eyeR, u_eyeRad, u_eyes);
    }
    if (u_slimFace > 0.001){
      float halfW = 0.55 * u_faceW;
      float top   = u_zygoY - 0.12 * u_faceH;
      float bot   = u_chinY + 0.08 * u_faceH;
      if (src.y > top && src.y < bot){
        float nx = (src.x - u_faceC.x) / halfW;
        if (abs(nx) < 1.0){
          float ty = clamp((src.y - top) / max(bot - top, 1.0), 0.0, 1.0);
          float gx = smoothstep(0.0, 0.85, abs(nx)) * (1.0 - smoothstep(0.92, 1.0, abs(nx)));
          float gy = smoothstep(0.0, 0.35, ty) * (1.0 - smoothstep(0.88, 1.0, ty));
          // 绝对位移：向中心压缩，边缘平滑收敛，避免撕裂背景
          src.x += sign(nx) * halfW * u_slimFace * 0.34 * gx * gy;
        }
      }
    }

    /* 尖下巴：下巴窄带内横向收 */
    if (u_slimChin > 0.001){
      float t2 = u_chinY - 0.30 * u_faceH;
      float b2 = u_chinY + 0.06 * u_faceH;
      if (src.y > t2 && src.y < b2){
        float ty = (src.y - t2) / max(b2 - t2, 1.0);
        float nx = (src.x - u_faceC.x) / (0.42 * u_faceW);
        float w = pow(clamp(ty, 0.0, 1.0), 1.4) * (1.0 - smoothstep(0.8, 1.5, abs(nx)));
        src.x += sign(nx) * 0.060 * u_faceW * u_slimChin * w * clamp(abs(nx) * 1.6, 0.0, 1.0);
      }
    }

    /* 瘦鼻：鼻梁周围横向轻收 */
    if (u_slimNose > 0.001){
      float t3 = u_noseY - 0.10 * u_faceH;
      float b3 = u_noseY + 0.12 * u_faceH;
      if (src.y > t3 && src.y < b3){
        float ty = (src.y - t3) / max(b3 - t3, 1.0);
        float nx = (src.x - u_faceC.x) / (0.14 * u_faceW);
        float w = sin(ty * 3.14159) * (1.0 - smoothstep(0.9, 1.8, abs(nx)));
        src.x += sign(nx) * 0.042 * u_faceW * u_slimNose * w;
      }
    }

    /* 微笑唇：嘴角向上提（只动嘴角小范围，避免整唇拉伸） */
    if (u_smile > 0.001){
      vec2 cL = u_mouthC - vec2(0.5 * u_mouthW, 0.0);
      vec2 cR = u_mouthC + vec2(0.5 * u_mouthW, 0.0);
      float r = max(u_mouthW * 0.40, 3.0);
      for (int i = 0; i < 2; i++){
        vec2 cc = (i == 0) ? cL : cR;
        float d = length(src - cc);
        if (d < r){
          float w = 1.0 - d / r;
          src.y -= 0.15 * r * u_smile * w * w;   // y 向下 → 减小即上提
        }
      }
    }

    /* 提拉：脸颊下半部整体上提 */
    if (u_lift > 0.001){
      float ty = clamp((src.y - u_zygoY) / max(u_chinY - u_zygoY, 1.0), 0.0, 1.0);
      float nx = (src.x - u_faceC.x) / (0.62 * u_faceW);
      float w = (1.0 - smoothstep(0.75, 1.25, abs(nx))) * sin(ty * 3.14159);
      src.y -= 0.034 * u_faceH * u_lift * w;
    }
  }
  vec2 suv = vec2(src.x / u_size.x, 1.0 - src.y / u_size.y);
  vec3 c = texture(u_img, clamp(suv, u_texel * 0.5, 1.0 - u_texel * 0.5)).rgb;

  /* --- 2. 磨皮 + 匀肤（共用一次表面模糊）--- */
  float sc = skinMask(c);
  float fm = (u_hasFace > 0) ? texture(u_faceMask, v_uv).r : 0.0;
  // 脸内（蒙版白）直接算皮肤，脸外仍靠色彩阈值 —— 解决浓妆/深肤色边界漏检
  float skin = clamp(fm + sc * 0.50 * (1.0 - fm), 0.0, 1.0);
  vec3 sm = c;
  if (u_smooth > 0.001 || u_unify > 0.001 || u_softglow > 0.001 || u_sharpen > 0.001){
    float th2 = 0.012 + 0.10 * (1.0 - max(u_smooth, 0.45));
    sm = surfaceBlur(clamp(suv, u_texel, 1.0 - u_texel), c, th2);
  }
  if (u_smooth > 0.001){
    float luma0 = dot(c, LUMA);
    float guard = 1.0 - 0.75 * clamp((luma0 - 0.72) / 0.28, 0.0, 1.0);
    float k = u_smooth * (0.35 + 0.65 * u_smooth) * skin * guard;
    // 高频细节按 texture 回加，避免塑料感
    vec3 detail = (c - sm) * (1.0 - u_texture * 0.85);
    c = mix(c, sm, k) + detail * k;
  }
  if (u_unify > 0.001){
    // 匀肤：保留当前亮度，替换成平滑后的色度（≈ Lab A/B 抹平）
    float l0 = dot(c, LUMA);
    vec3 chroma = sm - vec3(dot(sm, LUMA));
    c = mix(c, clamp(vec3(l0) + chroma, 0.0, 1.0), u_unify * 0.72 * skin);
  }

  /* --- 3. 美白（shadow → 降饱和 → 降色温 → 提亮高光）--- */
  if (u_whiten > 0.001){
    float a = u_whiten;
    vec3 w = c;
    float lum = dot(w, LUMA);
    w += a * 0.42 * (1.0 - lum) * (1.0 - lum) * (1.0 - w);      // 抬暗部
    float g = dot(w, LUMA);
    w = mix(vec3(g), w, 1.0 - 0.16 * a);                          // 降饱和
    w.r -= 0.10 * a; w.b += 0.03 * a;                             // 降色温
    w += a * 0.06 * lum;                                          // 提亮高光
    float mixAmt = skin * (0.85 + 0.15 * a);
    c = mix(c, clamp(w, 0.0, 1.0), mixAmt);
  }

  /* --- 3b. 气色（皮肤加红润）--- */
  if (u_complexion > 0.001){
    float a = u_complexion * skin;
    float l0 = dot(c, LUMA);
    float fade = 1.0 - smoothstep(0.58, 0.96, l0);
    c.r += 0.085 * a * fade;
    c.g += 0.018 * a * fade;
    c.b -= 0.004 * a * fade;
  }

  /* --- 3c. 亮眼（眼周提亮 + 局部加对比）--- */
  if (u_brightEye > 0.001 && u_hasFace > 0){
    float er = u_eyeRad * 1.55;
    float dmin = min(length(pix - u_eyeL), length(pix - u_eyeR));
    float m = 1.0 - smoothstep(er * 0.55, er, dmin);
    c += u_brightEye * 0.20 * m * (1.0 - c);
    c = mix(c, (c - 0.5) * (1.0 + u_brightEye * 0.50 * m) + 0.5, m);
  }

  /* --- 3d. 唇彩（唇部提饱和偏红）--- */
  if (u_lip > 0.001 && u_hasFace > 0){
    vec2 dd = (pix - u_mouthC) / vec2(max(u_mouthW * 0.62, 3.0), max(u_mouthW * 0.40, 3.0));
    float m = 1.0 - smoothstep(0.72, 1.05, length(dd));
    float redness = clamp((c.r - c.g) * 6.0, 0.0, 1.0);
    float mm = m * mix(0.30, 1.0, redness);
    float l0 = dot(c, LUMA);
    vec3 t = mix(vec3(l0), c, 1.0 + 0.55 * u_lip * mm);
    t.r += 0.055 * u_lip * mm;
    t.g -= 0.014 * u_lip * mm;
    c = mix(c, clamp(t, 0.0, 1.0), mm);
  }

  /* --- 3f. 牙齿美白（唇内 亮+低饱和 去黄提亮）--- */
  if (u_teeth > 0.001 && u_hasFace > 0){
    vec2 dt = (pix - u_mouthC) / vec2(max(u_mouthW * 0.44, 2.0), max(u_mouthW * 0.22, 2.0));
    float mt = 1.0 - smoothstep(0.68, 1.02, length(dt));
    float lt = dot(c, LUMA);
    float mxt = max(max(c.r, c.g), c.b);
    float mnt = min(min(c.r, c.g), c.b);
    float st = (mxt - mnt) / max(mxt, 1e-3);
    float tooth = mt * smoothstep(0.42, 0.70, lt) * (1.0 - smoothstep(0.12, 0.34, st));
    vec3 tw = c;
    tw.r -= 0.060 * u_teeth;          // 去黄
    tw.g -= 0.012 * u_teeth;
    tw += (1.0 - tw) * (0.20 * u_teeth);
    c = mix(c, clamp(tw, 0.0, 1.0), tooth);
  }

  /* --- 3g. 眼白提亮（眼球内 高亮+低饱和）--- */
  if (u_eyeWhite > 0.001 && u_hasFace > 0){
    float er = u_eyeRad * 1.10;
    float de = min(length(pix - u_eyeL), length(pix - u_eyeR));
    float me = 1.0 - smoothstep(er * 0.40, er, de);
    float le = dot(c, LUMA);
    float mxe = max(max(c.r, c.g), c.b);
    float mne = min(min(c.r, c.g), c.b);
    float se = (mxe - mne) / max(mxe, 1e-3);
    float wht = me * smoothstep(0.32, 0.60, le) * (1.0 - smoothstep(0.10, 0.28, se));
    c += u_eyeWhite * 0.26 * wht * (1.0 - c);
  }

  /* --- 3e. 去雾（暗通道先验的保守版）--- */
  if (u_dehaze > 0.001){
    float a = u_dehaze;
    float mn = min(min(c.r, c.g), c.b);
    float mx = max(max(c.r, c.g), c.b);
    float sat0 = (mx - mn) / max(mx, 1e-3);
    // 雾的特征：暗通道也偏高（发白）且低饱和；高饱和物体（皮肤/红衣）不判为雾
    float haze = clamp((mn - 0.35) / 0.65, 0.0, 1.0) * (1.0 - smoothstep(0.15, 0.45, sat0));
    float A = 0.88;
    float trans = clamp(1.0 - 0.55 * haze, 0.62, 1.0);
    vec3 dh = (c - A) / trans + A;
    dh = (dh - 0.5) * (1.0 + 0.10 * a) + 0.5;   // 顺带补一点对比
    c = mix(c, dh, a * 0.85);
  }

  /* --- 3i. 背景虚化（脸保持清晰，四周柔化）--- */
  if (u_blur > 0.001) {
    float br = u_blur * 0.011 * min(u_size.x, u_size.y);
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    // 黄金角螺旋采样，分布均匀，避免放射状拉丝
    for (int i = 0; i < 12; i++) {
      float fi = float(i);
      float ba = fi * 2.39996323;
      float brr = sqrt((fi + 0.5) / 12.0) * br;
      vec2 off = vec2(cos(ba), sin(ba)) * brr;
      acc += texture(u_img, v_uv + off / u_size).rgb;
      ws += 1.0;
    }
    acc /= ws;
    c = mix(c, acc, (1.0 - skin) * u_blur);   // 皮肤不动，其余虚化
  }

  /* --- 4. 全局调色 --- */
  {
    c *= pow(2.0, u_exposure);
    float lum = dot(c, LUMA);
    c += u_shadow    * 0.45 * (1.0 - lum) * (1.0 - lum) * (1.0 - c);
    c += u_highlight * 0.35 * lum * lum * (1.0 - c);
    c = (c - 0.5) * (1.0 + u_contrast * 0.85) + 0.5;
    float g2 = dot(c, LUMA);
    c = mix(vec3(g2), c, 1.0 + u_saturation);
    c.r += u_temp * 0.085; c.b -= u_temp * 0.085;
    c = clamp(c, 0.0, 1.0);
  }

  /* --- 5. LUT --- */
  if (u_hasLut > 0 && u_lutAmount > 0.001){
    c = mix(c, applyLut(c), u_lutAmount);
  }

  /* --- 6. 柔光 --- */
  if (u_softglow > 0.001){
    vec3 g = clamp(sm * 0.55 + 0.45, 0.0, 1.0);
    vec3 screen = 1.0 - (1.0 - c) * (1.0 - g);
    c = mix(c, screen, u_softglow * 0.32);
  }

  /* --- 7. 锐化（皮肤区减弱）--- */
  if (u_sharpen > 0.001){
    c += (c - sm) * u_sharpen * 1.5 * (1.0 - 0.72 * skin);
  }

  o = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

/* ---------------------------------------------------------------- 参数表 */
const SLIDERS_SKIN = [
  ['smooth',     '磨皮', 48],
  ['texture',    '质感', 45],
  ['unify',      '匀肤', 38],
];
const SLIDERS_SHAPE = [
  ['slimFace',   '瘦脸', 50],
  ['slimChin',   '尖下巴', 20],
  ['eyes',       '大眼', 28],
  ['slimNose',   '瘦鼻', 18],
  ['smile',      '微笑唇', 10],
  ['lift',       '提拉', 12],
];
const SLIDERS_TONE = [
  ['whiten',     '美白', 32],
  ['complexion', '气色', 22],
  ['brightEye',  '亮眼', 16],
  ['eyeWhite',   '眼白', 12],
  ['lip',        '唇彩', 12],
  ['teeth',      '牙齿', 10],
];
const SLIDERS_COLOR = [
  ['exposure',   '曝光', 50],
  ['contrast',   '对比', 50],
  ['saturation', '饱和', 50],
  ['temp',       '色温', 50],
  ['shadow',     '阴影', 50],
  ['highlight',  '高光', 50],
  ['dehaze',     '去雾', 0],
  ['blur',       '虚化', 0],
  ['softglow',   '柔光', 5],
  ['sharpen',    '锐化', 10],
  ['lutAmount',  '滤镜', 0],
];
const SLIDER_GROUPS = { skin: SLIDERS_SKIN, shape: SLIDERS_SHAPE, tone: SLIDERS_TONE, color: SLIDERS_COLOR };
const ALL_SLIDERS = [...SLIDERS_SKIN, ...SLIDERS_SHAPE, ...SLIDERS_TONE, ...SLIDERS_COLOR];
/* 滑块 0..100 → uniform 值 */
const MAP = {
  smooth:    (v) => v / 100,
  texture:   (v) => v / 100,
  unify:     (v) => v / 100,
  whiten:    (v) => v / 100,
  complexion:(v) => v / 100,
  brightEye: (v) => v / 100,
  eyeWhite:  (v) => v / 100,
  lip:       (v) => v / 100,
  teeth:     (v) => v / 100,
  slimFace:  (v) => v / 100,
  slimChin:  (v) => v / 100,
  eyes:      (v) => v / 100,
  slimNose:  (v) => v / 100,
  smile:     (v) => v / 100,
  lift:      (v) => v / 100,
  exposure:  (v) => (v - 50) / 50 * 0.5,
  contrast:  (v) => (v - 50) / 50,
  saturation:(v) => (v - 50) / 50,
  temp:      (v) => (v - 50) / 50,
  shadow:    (v) => (v - 50) / 50,
  highlight: (v) => (v - 50) / 50,
  dehaze:    (v) => v / 100,
  blur:      (v) => v / 100,
  softglow:  (v) => v / 100,
  sharpen:   (v) => v / 100,
  lutAmount: (v) => v / 100,
};
const DEFAULTS = {};
ALL_SLIDERS.forEach(([k, , d]) => { DEFAULTS[k] = d; });

/* ?only=slimChin:100 把所有滑块归零后只留指定项；?p=slimChin:100,smile:80 只覆盖指定项 */
const NEUTRAL_SLIDER = { exposure: 50, contrast: 50, saturation: 50, temp: 50, shadow: 50, highlight: 50 };
const URL_PARAMS = (() => {
  const sp = new URLSearchParams(location.search);
  const set = {};
  let zeroOthers = false;
  const take = (str) => {
    for (const pair of str.split(',')) {
      const [k, v] = pair.split(':');
      if (k && k in DEFAULTS && v !== undefined && !Number.isNaN(+v)) {
        set[k] = Math.max(0, Math.min(100, +v));
      }
    }
  };
  const only = sp.get('only');
  if (only) { zeroOthers = true; take(only); }
  const raw = sp.get('p');
  if (raw) take(raw);
  return { zeroOthers, set };
})();

function applyUrlOverrides(values) {
  if (URL_PARAMS.zeroOthers) {
    for (const k in values) values[k] = NEUTRAL_SLIDER[k] !== undefined ? NEUTRAL_SLIDER[k] : 0;
  }
  Object.assign(values, URL_PARAMS.set);
  return values;
}

const PRESETS = {
  portrait: { smooth: 48, texture: 45, unify: 38, whiten: 32, complexion: 22,
              brightEye: 16, lip: 12, slimFace: 55, slimChin: 20, eyes: 28,
              slimNose: 18, smile: 10, lift: 12,
              exposure: 53, contrast: 56, saturation: 53, temp: 50, shadow: 50,
              highlight: 50, dehaze: 0, softglow: 5, sharpen: 10, lutAmount: 0 },
  natural:  { smooth: 22, texture: 70, unify: 16, whiten: 14, complexion: 10,
              brightEye: 8, lip: 5, slimFace: 18, slimChin: 6, eyes: 10,
              slimNose: 6, smile: 4, lift: 4,
              exposure: 52, contrast: 53, saturation: 52, temp: 50, shadow: 50,
              highlight: 50, dehaze: 0, softglow: 0, sharpen: 12, lutAmount: 0 },
  glow:     { smooth: 34, texture: 50, unify: 30, whiten: 32, complexion: 16,
              brightEye: 12, lip: 10, slimFace: 22, slimChin: 8, eyes: 14,
              slimNose: 8, smile: 6, lift: 6,
              exposure: 55, contrast: 50, saturation: 55, temp: 53, shadow: 58,
              highlight: 46, dehaze: 0, softglow: 22, sharpen: 6, lutAmount: 20 },
};

/* ---------------------------------------------------------------- 状态 */
const state = {
  gl: null, prog: null, tex: null,
  size: { w: 1, h: 1 },
  img: null,            // HTMLImageElement / ImageBitmap
  face: null,           // {cx, cy, w, h, zygoY, chinY, eyeL:[x,y], eyeR:[x,y], eyeRad}
  lut: null,            // WebGLTexture
  lutName: '',
  values: applyUrlOverrides({ ...DEFAULTS }),
  warpMode: false, warpLast: null, warpRadius: 90,
  warpW: 0, warpH: 0, warpFx: null, warpU8: null, warpTex: null,
  warpScale: 1, size0W: 0,
};

/* ---------------------------------------------------------------- GL 初始化 */
function initGL() {
  const canvas = $('#gl');
  const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error('浏览器不支持 WebGL2');
  state.gl = gl;

  const compile = (type, src) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error('shader 编译失败: ' + gl.getShaderInfoLog(sh));
    }
    return sh;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error('program 链接失败: ' + gl.getProgramInfoLog(prog));
  }
  gl.useProgram(prog);
  state.prog = prog;

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'a_pos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  // 空纹理占位
  state.tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, state.tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  setTexParams(gl);

  // 脸部蒙版纹理（未检测到人脸时 1×1 黑）
  state.maskTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, state.maskTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  setTexParams(gl);

  // 手动液化位移场（1×1 中灰 = 零位移）
  state.warpTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, state.warpTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 128, 255]));
  setTexParams(gl);
}

function setTexParams(gl) {
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
}

/* ---------------------------------------------------------------- 图片 */
const MAX_SIDE = 1600;

async function loadImageFile(file) {
  const url = URL.createObjectURL(file);
  try { await loadImageURL(url); } finally { URL.revokeObjectURL(url); }
}

async function loadImageURL(url) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise((res, rej) => { img.onload = res; img.onerror = () => rej(new Error('图片加载失败')); img.src = url; });

  // 缩小到 MAX_SIDE
  const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);

  const gl = state.gl;
  const canvas = $('#gl');
  canvas.width = w; canvas.height = h;
  state.size = { w, h };
  state.size0W = w;
  state.warpScale = 1;
  initWarpField(w, h);

  gl.bindTexture(gl.TEXTURE_2D, state.tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  setTexParams(gl);

  state.img = img;
  state.face = null;
  // 同一张图的人脸结果缓存到 sessionStorage（省掉每次 20s+ 的模型推理）
  const ckey = `xt:face:${w}x${h}:${String(url).slice(-64)}`;
  try {
    const hit = sessionStorage.getItem(ckey);
    if (hit) { state.face = JSON.parse(hit); state.face._baseW = w; }
  } catch (_) {}
  $('#empty').classList.add('hidden');
  $('#toolbar').classList.remove('hidden');
  $('#stage').classList.add('hasImg');
  $('#btnSave').disabled = false;
  $('#btnHeal').disabled = false;
  $('#btnAuto').disabled = false;
  $('#btnWarp').disabled = false;
  $('#btnWarpClear').disabled = false;

  render();
  updateStatus();

  if (state.face) {
    applyFaceMask(state.face);
    setStatus(`人脸 1 · ${w}×${h}`);
  } else {
    // 异步检测人脸（不阻塞首帧）
    detectFace(img, w, h, ckey).catch((e) => {
      console.warn('人脸检测跳过:', e.message);
      setStatus('人脸检测失败: ' + String(e.message || e).slice(0, 40));
    });
  }
}

/* 把屏幕坐标换算成图像像素坐标 */
function canvasPos(e) {
  const c = $('#gl');
  const r = c.getBoundingClientRect();
  const { w, h } = state.size;
  return [
    Math.max(0, Math.min(w, (e.clientX - r.left) / Math.max(r.width, 1) * w)),
    Math.max(0, Math.min(h, (e.clientY - r.top) / Math.max(r.height, 1) * h)),
  ];
}

/* ---------------------------------------------------------------- 手动液化 */
/* 位移场按 1/4 分辨率存（液化本身是低频场），RG 两通道编码 -32..32 像素：
   这样 float 精度够、又可以用 LINEAR 过滤（float32 纹理在 WebGL2 里只能 NEAREST）。 */
const WARP_DIV = 4;
const WARP_RANGE = 96;

function initWarpField(w, h) {
  state.warpW = Math.max(1, Math.ceil(w / WARP_DIV));
  state.warpH = Math.max(1, Math.ceil(h / WARP_DIV));
  const n = state.warpW * state.warpH;
  state.warpFx = new Float32Array(n * 2);
  state.warpU8 = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    state.warpU8[i * 4] = 128;
    state.warpU8[i * 4 + 1] = 128;
    state.warpU8[i * 4 + 2] = 128;
    state.warpU8[i * 4 + 3] = 255;
  }
  uploadWarpField();
}

function uploadWarpField() {
  const gl = state.gl;
  if (!gl || !state.warpU8) return;
  const { warpW, warpH, warpFx, warpU8 } = state;
  for (let i = 0; i < warpW * warpH; i++) {
    const dx = Math.max(-WARP_RANGE, Math.min(WARP_RANGE, warpFx[i * 2]));
    const dy = Math.max(-WARP_RANGE, Math.min(WARP_RANGE, warpFx[i * 2 + 1]));
    warpU8[i * 4] = Math.round((dx + WARP_RANGE) / (WARP_RANGE * 2) * 255);
    warpU8[i * 4 + 1] = Math.round((dy + WARP_RANGE) / (WARP_RANGE * 2) * 255);
  }
  if (!state.warpTex) state.warpTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, state.warpTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, warpW, warpH, 0, gl.RGBA, gl.UNSIGNED_BYTE, warpU8);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

/* 在图像坐标 (cx,cy) 处沿 (dx,dy) 推一把 */
function warpPush(cx, cy, dx, dy, radius) {
  const { warpW, warpH, warpFx } = state;
  const scale = 1 / WARP_DIV;
  const wx = cx * scale, wy = cy * scale, wr = radius * scale;
  const x0 = Math.max(0, Math.floor(wx - wr)), x1 = Math.min(warpW - 1, Math.ceil(wx + wr));
  const y0 = Math.max(0, Math.floor(wy - wr)), y1 = Math.min(warpH - 1, Math.ceil(wy + wr));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const ddx = x - wx, ddy = y - wy;
      const d = Math.sqrt(ddx * ddx + ddy * ddy);
      if (d > wr) continue;
      const t = 1 - d / Math.max(wr, 1e-3);
      const w = t * t * (3 - 2 * t);      // smoothstep 衰减
      const i = (y * warpW + x) * 2;
      warpFx[i] += dx * w;
      warpFx[i + 1] += dy * w;
    }
  }
}

function resetWarpField() {
  if (!state.warpFx) return;
  state.warpFx.fill(0);
  uploadWarpField();
  draw();
}
/* 先量照片（脸型比例 / 肤色明暗 / 肤色均匀度 / 清晰度 / 色温），再决定参数。
   醒图那套是固定档位，这里是按照片自适应的。 */
function measurePhoto() {
  const img = state.img;
  if (!img) return null;
  const w = state.size.w, h = state.size.h;
  const S = 320 / Math.max(w, h);
  const sw = Math.max(24, Math.round(w * S)), sh = Math.max(24, Math.round(h * S));
  const cv = document.createElement('canvas');
  cv.width = sw; cv.height = sh;
  const g = cv.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0, sw, sh);
  const d = g.getImageData(0, 0, sw, sh).data;

  // 脸区（缩放到小图上）
  const f = state.face;
  let fx0 = 0, fx1 = sw, fy0 = 0, fy1 = sh, hasFace = false;
  if (f) {
    hasFace = true;
    fx0 = Math.max(0, Math.round((f.cx - f.w * 0.5) * S));
    fx1 = Math.min(sw, Math.round((f.cx + f.w * 0.5) * S));
    fy0 = Math.max(0, Math.round((f.cy - f.h * 0.5) * S));
    fy1 = Math.min(sh, Math.round((f.cy + f.h * 0.5) * S));
  }

  let n = 0, sum = 0, sum2 = 0, sr = 0, sg = 0, sb = 0;
  for (let y = fy0; y < fy1; y++) {
    for (let x = fx0; x < fx1; x++) {
      const j = (y * sw + x) * 4;
      const r = d[j] / 255, gg = d[j + 1] / 255, bb = d[j + 2] / 255;
      const l = r * 0.299 + gg * 0.587 + bb * 0.114;
      if (l < 0.12) continue;
      const mx = Math.max(r, gg, bb), mn = Math.min(r, gg, bb);
      if ((mx - mn) / Math.max(mx, 1e-3) > 0.62) continue;   // 跳过衣服/口红等强色
      n++; sum += l; sum2 += l * l; sr += r; sg += gg; sb += bb;
    }
  }
  const lumaMean = n ? sum / n : 0.5;
  const lumaStd = n ? Math.sqrt(Math.max(0, sum2 / n - lumaMean * lumaMean)) : 0.1;
  const warmth = n ? (sr - sb) / n : 0;

  // 清晰度：拉普拉斯方差（整图缩小版）
  let lapSum = 0, lapSum2 = 0, m = 0;
  for (let y = 1; y < sh - 1; y++) {
    for (let x = 1; x < sw - 1; x++) {
      const at = (yy, xx) => {
        const j = (yy * sw + xx) * 4;
        return (d[j] * 0.299 + d[j + 1] * 0.587 + d[j + 2] * 0.114) / 255;
      };
      const v = 4 * at(y, x) - at(y - 1, x) - at(y + 1, x) - at(y, x - 1) - at(y, x + 1);
      lapSum += v; lapSum2 += v * v; m++;
    }
  }
  const lapVar = m ? lapSum2 / m - (lapSum / m) ** 2 : 0;

  // 斑点估计：脸区内局部显著偏暗的小块占比
  let spotRatio = 0;
  if (hasFace && f) {
    const fw = fx1 - fx0, fh = fy1 - fy0;
    if (fw > 8 && fh > 8) {
      const sub = g.getImageData(fx0, fy0, fw, fh);
      const sd = sub.data;
      const lum = new Float32Array(fw * fh);
      for (let i = 0; i < fw * fh; i++) {
        const j = i * 4;
        lum[i] = (sd[j] * 0.299 + sd[j + 1] * 0.587 + sd[j + 2] * 0.114) / 255;
      }
      const mean = boxBlur1(lum, fw, fh, 3);
      let cnt = 0;
      for (let i = 0; i < fw * fh; i++) {
        if (lum[i] > 0.12 && lum[i] < 0.93 && mean[i] - lum[i] > 0.055) cnt++;
      }
      spotRatio = cnt / (fw * fh);
    }
  }

  const ratio = f ? f.w / Math.max(f.h, 1) : 0.75;
  return { lumaMean, lumaStd, warmth, lapVar, spotRatio, ratio, hasFace };
}

function autoTune() {
  const q = measurePhoto();
  if (!q) return null;
  const cl = (v, a, b) => Math.max(a, Math.min(b, v));

  // 肤色偏暗/偏灰 → 加强美白与气色；越亮就越轻
  const whiten = cl((0.585 - q.lumaMean) * 190, 6, 52);
  // 肤色不均 → 匀肤；纹理粗糙 → 磨皮
  const unify = cl((q.lumaStd - 0.055) * 330, 8, 52);
  const smooth = cl(28 + (q.lumaStd - 0.08) * 240, 18, 62);
  // 脸偏宽才瘦脸；脸本来就窄就不动
  const slimFace = cl((q.ratio - 0.70) * 190, 4, 52);
  const slimChin = cl(slimFace * 0.42, 0, 26);
  // 斑点越多，越该点祛痘（参数只提示，动作留给按钮）
  const healHint = q.spotRatio > 0.0016;
  // 画面发灰（暗部多、清晰度低）才去雾；清晰的图不加
  const dehaze = cl((0.0016 - q.lapVar) * 22000, 0, 26);
  // 偏黄 → 冷一点；偏冷 → 暖一点
  const temp = cl(50 - q.warmth * 42, 40, 60);
  const exposure = cl(50 + (0.50 - q.lumaMean) * 55, 38, 66);
  const sharpen = cl((0.0022 - q.lapVar) * 9000, 2, 26);
  const complexion = cl((0.56 - q.lumaMean) * 120, 4, 30);

  const vals = {
    smooth, texture: 46, unify, whiten, complexion, brightEye: 14, eyeWhite: 10,
    lip: 10, teeth: 8,
    slimFace, slimChin, eyes: cl(14 + smooth * 0.18, 10, 26),
    slimNose: cl(slimFace * 0.32, 2, 18), smile: 8, lift: cl(slimFace * 0.26, 2, 14),
    exposure, contrast: 52, saturation: 52, temp, shadow: 52, highlight: 49,
    dehaze, softglow: 4, sharpen, lutAmount: 0,
  };
  Object.assign(state.values, vals);
  syncSliders();
  render();
  return { q, vals, healHint };
}
function boxBlur1(src, w, h, r) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const d = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sum = 0;
    for (let x = -r; x <= r; x++) sum += src[row + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum / d;
      const xa = Math.min(w - 1, x + r + 1);
      const xr = Math.max(0, x - r);
      sum += src[row + xa] - src[row + xr];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) sum += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / d;
      const ya = Math.min(h - 1, y + r + 1);
      const yr = Math.max(0, y - r);
      sum += tmp[ya * w + x] - tmp[yr * w + x];
    }
  }
  return out;
}

/* 为一个斑点找最合适的"干净源块"：在周围一圈里找不含候选像素、且亮度最接近的区域 */
function pickPatchSource(px, w, h, cand, b, bw, bh) {
  const R = Math.max(bw, bh) + 8;
  let best = null, bestScore = Infinity;
  for (let dy = -R; dy <= R; dy += 2) {
    for (let dx = -R; dx <= R; dx += 2) {
      if (Math.abs(dx) < bw * 0.75 && Math.abs(dy) < bh * 0.75) continue;   // 别选回自己
      const sx = b.minX + dx, sy = b.minY + dy;
      if (sx < 0 || sy < 0 || sx + bw > w || sy + bh > h) continue;
      let ok = true, score = 0, n = 0;
      for (let y = 0; y < bh && ok; y += 2) {
        for (let x = 0; x < bw; x += 2) {
          const sp = (sy + y) * w + (sx + x);
          if (cand[sp]) { ok = false; break; }
          const qp = (b.minY + y) * w + (b.minX + x);
          const d = px[sp * 4] - px[qp * 4];
          score += d * d;
          n++;
        }
      }
      if (!ok || !n) continue;
      score /= n;
      if (score < bestScore) { bestScore = score; best = { sx, sy }; }
    }
  }
  return best;
}

function findBlobs(cand, w, h) {  const seen = new Uint8Array(w * h);
  const blobs = [];
  const stack = [];
  for (let i = 0; i < w * h; i++) {
    if (!cand[i] || seen[i]) continue;
    const pts = [];
    stack.length = 0;
    stack.push(i);
    seen[i] = 1;
    let minX = w, maxX = 0, minY = h, maxY = 0;
    while (stack.length) {
      const p = stack.pop();
      const x = p % w, y = (p - x) / w;
      pts.push(x, y);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (x > 0 && cand[p - 1] && !seen[p - 1]) { seen[p - 1] = 1; stack.push(p - 1); }
      if (x < w - 1 && cand[p + 1] && !seen[p + 1]) { seen[p + 1] = 1; stack.push(p + 1); }
      if (y > 0 && cand[p - w] && !seen[p - w]) { seen[p - w] = 1; stack.push(p - w); }
      if (y < h - 1 && cand[p + w] && !seen[p + w]) { seen[p + w] = 1; stack.push(p + w); }
    }
    const n = pts.length / 2;
    const bw = maxX - minX + 1, bh = maxY - minY + 1;
    if (n >= 5 && n <= 520 && Math.max(bw, bh) / Math.max(1, Math.min(bw, bh)) < 4.2) {
      blobs.push({ pts, minX, maxX, minY, maxY });
    }
  }
  return blobs;
}

/* 在脸部皮肤区域内找斑点（局部明显偏暗/偏红的小块），用周围像素均值补掉 */
function healBlemishes(strength) {
  const img = state.img;
  if (!img) return 0;
  const w = state.size.w, h = state.size.h;
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const g = cv.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0, w, h);
  const id = g.getImageData(0, 0, w, h);
  const px = id.data;

  let mask = null;
  if (state.maskCanvas) {
    const mg = state.maskCanvas.getContext('2d', { willReadFrequently: true });
    mask = mg.getImageData(0, 0, w, h).data;
  }

  const n = w * h;
  const lum = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const j = i * 4;
    lum[i] = (px[j] * 0.299 + px[j + 1] * 0.587 + px[j + 2] * 0.114) / 255;
  }
  const mean = boxBlur1(boxBlur1(lum, w, h, 4), w, h, 2);

  const cand = new Uint8Array(n);
  const th = 0.052 + 0.045 * (1 - strength);
  for (let i = 0; i < n; i++) {
    if (mask && mask[i * 4] < 120) continue;         // 只在脸部皮肤内
    if (lum[i] < 0.10 || lum[i] > 0.94) continue;    // 太黑太亮跳过
    if (mean[i] - lum[i] > th) cand[i] = 1;          // 局部偏暗 = 斑点候选
  }

  const blobs = findBlobs(cand, w, h);
  let done = 0;
  for (const b of blobs) {
    const pad = 3;
    let r = 0, gg = 0, bb = 0, cnt = 0;
    const x0 = Math.max(0, b.minX - pad), x1 = Math.min(w - 1, b.maxX + pad);
    const y0 = Math.max(0, b.minY - pad), y1 = Math.min(h - 1, b.maxY + pad);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const p = y * w + x;
        if (cand[p]) continue;
        const j = p * 4;
        r += px[j]; gg += px[j + 1]; bb += px[j + 2]; cnt++;
      }
    }
    if (!cnt) continue;
    r /= cnt; gg /= cnt; bb /= cnt;

    // 优先用周围最像的干净小块贴过来（保住毛孔纹理），找不到才退回均值+噪声
    const bw = b.maxX - b.minX + 1, bh = b.maxY - b.minY + 1;
    const src = (bw >= 3 && bh >= 3) ? pickPatchSource(px, w, h, cand, b, bw, bh) : null;
    for (let k = 0; k < b.pts.length; k += 2) {
      const x = b.pts[k], y = b.pts[k + 1];
      const p = y * w + x;
      const j = p * 4;
      if (src) {
        const sp = ((src.sy + (y - b.minY)) * w + (src.sx + (x - b.minX))) * 4;
        // 边缘一圈与周围均值混合，消掉接缝
        const edge = Math.min(x - b.minX, b.maxX - x, y - b.minY, b.maxY - y);
        const mixk = edge >= 2 ? 1 : 0.45 + 0.275 * edge;
        px[j] = px[sp] * mixk + r * (1 - mixk);
        px[j + 1] = px[sp + 1] * mixk + gg * (1 - mixk);
        px[j + 2] = px[sp + 2] * mixk + bb * (1 - mixk);
      } else {
        const noise = (Math.random() - 0.5) * 9;
        px[j] = Math.max(0, Math.min(255, r + noise));
        px[j + 1] = Math.max(0, Math.min(255, gg + noise));
        px[j + 2] = Math.max(0, Math.min(255, bb + noise));
      }
    }
    done++;
  }

  if (done) {
    g.putImageData(id, 0, 0);
    const gl = state.gl;
    gl.bindTexture(gl.TEXTURE_2D, state.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cv);
    setTexParams(gl);
    render();
  }
  return done;
}
let landmarker = null;
let landmarkerTried = false;

const MODEL_CANDIDATES = [
  'models/face_landmarker.task',
  'https://hf-mirror.com/Skulleton12/mediapipe-face-landmarker/resolve/main/face_landmarker.task',
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
];

async function getLandmarker() {
  if (landmarker || landmarkerTried) return landmarker;
  landmarkerTried = true;
  const V = '0.10.14';
  const vision = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${V}/vision_bundle.mjs`);
  const fileset = await vision.FilesetResolver.forVisionTasks(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${V}/wasm`);

  for (const url of MODEL_CANDIDATES) {
    for (const delegate of ['GPU', 'CPU']) {
      try {
        landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: url, delegate },
          runningMode: 'IMAGE',
          numFaces: 1,
          minFaceDetectionConfidence: 0.3,
        });
        return landmarker;
      } catch (e) {
        console.warn('[face]', delegate, url, e && e.message);
      }
    }
  }
  return landmarker;
}

async function detectFace(img, w, h, ckey) {
  const lm = await getLandmarker();
  if (!lm) return;
  const res = lm.detect(img);
  if (!res || !res.faceLandmarks || !res.faceLandmarks.length) {
    setStatus('未检测到人脸');
    return;
  }
  const pts = res.faceLandmarks[0].map((p) => [p.x * w, p.y * h]);
  const P = (i) => pts[i];

  // 用全部关键点的包围盒定脸框（比单点索引稳）
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const face = {
    top: minY,
    chinY: maxY,
    zygoY: minY + 0.45 * (maxY - minY),
    eyeL: [(P(33)[0] + P(133)[0]) / 2, (P(33)[1] + P(133)[1]) / 2],
    eyeR: [(P(362)[0] + P(263)[0]) / 2, (P(362)[1] + P(263)[1]) / 2],
  };
  face.w = maxX - minX;
  face.h = maxY - minY;
  face.cx = (minX + maxX) / 2;
  face.cy = (minY + maxY) / 2;
  face.eyeRad = Math.max(Math.abs(P(133)[0] - P(33)[0]), Math.abs(P(362)[0] - P(263)[0])) * 1.05;
  // 嘴：61 左嘴角 / 291 右嘴角 / 13 上唇 / 14 下唇
  face.mouthC = [(P(61)[0] + P(291)[0]) / 2, (P(13)[1] + P(14)[1]) / 2];
  face.mouthW = Math.max(Math.abs(P(291)[0] - P(61)[0]), face.w * 0.18);
  // 鼻：取 1 号鼻尖
  face.noseY = P(1)[1];
  face._baseW = w;
  state.face = face;
  if (ckey) { try { sessionStorage.setItem(ckey, JSON.stringify(face)); } catch (_) {} }
  applyFaceMask(face);
  render();
  setStatus(`人脸 1 · ${w}×${h}`);
}

/* 用 468 点画一张脸部皮肤蒙版（白=皮肤，挖掉眼/眉/唇），比纯色彩阈值稳得多 */
function buildFaceMask(face, w, h) {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const g = cv.getContext('2d');
  g.fillStyle = '#000';
  g.fillRect(0, 0, w, h);

  g.fillStyle = '#fff';
  g.beginPath();
  g.ellipse(face.cx, face.cy - face.h * 0.03, face.w * 0.50, face.h * 0.55, 0, 0, Math.PI * 2);
  g.fill();

  g.fillStyle = '#000';
  const er = Math.max(face.eyeRad, 4);
  for (const e of [face.eyeL, face.eyeR]) {
    g.beginPath(); g.ellipse(e[0], e[1], er * 1.28, er * 1.00, 0, 0, Math.PI * 2); g.fill();
    g.beginPath(); g.ellipse(e[0], e[1] - er * 1.95, er * 1.60, er * 0.60, 0, 0, Math.PI * 2); g.fill();
  }
  g.beginPath();
  g.ellipse(face.mouthC[0], face.mouthC[1], face.mouthW * 0.64, face.mouthW * 0.44, 0, 0, Math.PI * 2);
  g.fill();

  // 羽化，避免蒙版硬边在脸上留痕
  const soft = document.createElement('canvas');
  soft.width = w; soft.height = h;
  const gs = soft.getContext('2d');
  gs.filter = `blur(${Math.max(2, Math.round(w / 500))}px)`;
  gs.drawImage(cv, 0, 0);
  return soft;
}

function applyFaceMask(face) {
  const gl = state.gl;
  const { w, h } = state.size;
  if (!gl || !face) return;
  const canvas = buildFaceMask(face, w, h);
  state.maskCanvas = canvas;
  if (!state.maskTex) state.maskTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, state.maskTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
  setTexParams(gl);
}

/* ---------------------------------------------------------------- LUT */
async function loadLut(name) {
  if (!name) { state.lut = null; state.lutName = ''; render(); return; }
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = `luts/${name}.png`; });
  const gl = state.gl;
  if (!state.lut) state.lut = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, state.lut);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  state.lutName = name;
  render();
}

/* 滤镜页：按钮网格（比下拉在手机上好点） */
const LUT_LIST = [
  ['', '无', '原色'],
  ['cinematic', '电影青橙', '冷影暖光'],
  ['film_warm', '日系暖调', '柔暖胶片'],
  ['cool_clean', '清透冷调', '干净通透'],
  ['fade', '褪色胶片', '低饱和'],
  ['vivid', '浓郁', '高饱和'],
  ['sunset', '落日', '暖橙'],
  ['mint', '薄荷', '青绿'],
  ['night', '夜蓝', '冷暗'],
  ['sepia', '复古棕', '怀旧'],
  ['bw', '黑白', '去色'],
];

function buildLutGrid() {
  const box = $('#lutGrid');
  if (!box) return;
  box.innerHTML = '';
  LUT_LIST.forEach(([name, title, sub]) => {
    const b = document.createElement('div');
    b.className = 'lutbtn' + (state.lutName === name ? ' on' : '');
    b.innerHTML = `${title}<small>${sub}</small>`;
    b.onclick = async () => {
      try {
        await loadLut(name);
        if (name && state.values.lutAmount < 1) {
          state.values.lutAmount = 70;
          syncSliders();
        }
        if (!name) state.values.lutAmount = 0;
        document.querySelectorAll('.lutbtn').forEach((x) => x.classList.remove('on'));
        b.classList.add('on');
        setStatus(name ? `滤镜 · ${title}` : '滤镜已关闭');
        draw();
      } catch (e) {
        setStatus('滤镜加载失败');
      }
    };
    box.appendChild(b);
  });
}

/* ---------------------------------------------------------------- 导出 */
/* 三档：1600 快速 / 2400 高清 / 0 原图。
   选了比当前预览更大的尺寸时，按目标分辨率重跑一遍管线再导出，导完恢复预览。 */
function blobFromCanvas(canvas) {
  return new Promise((res) => canvas.toBlob(res, 'image/jpeg', 0.95));
}

async function saveExport() {
  const target = state.exportSize || MAX_SIDE;
  if (!state.img) return;

  const curMax = Math.max(state.size.w, state.size.h);
  const srcMax = Math.max(state.srcW || curMax, curMax);
  const want = target === 0 ? srcMax : Math.min(target, srcMax);

  if (want <= curMax + 1) {   // 目标不比现在大，直接导出
    const blob = await blobFromCanvas($('#gl'));
    download(blob);
    return;
  }

  // 需要更高分辨率：临时按目标尺寸重建一次
  setStatus('按高清重算…');
  $('#busy').classList.remove('hidden');
  await new Promise((r) => setTimeout(r, 30));   // 让提示先画出来

  const w0 = state.size.w, h0 = state.size.h;
  const face0 = state.face;
  const warpBak = state.warpFx ? Float32Array.from(state.warpFx) : null;
  const warpW0 = state.warpW, warpH0 = state.warpH;

  try {
    await rebuildAt(want, warpBak, warpW0, warpH0);
    download(await blobFromCanvas($('#gl')));
  } catch (e) {
    console.warn(e);
    setStatus('高清导出失败：' + String(e.message || e).slice(0, 30));
  } finally {
    await restoreAt(w0, h0, face0, warpBak, warpW0, warpH0);
    $('#busy').classList.add('hidden');
  }
}

/* 位移场重采样（缩放尺寸时用最近邻，够用） */
function resampleWarp(bak, bw, bh) {
  const nw = state.warpW, nh = state.warpH;
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(bh - 1, Math.floor((y * bh) / nh));
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(bw - 1, Math.floor((x * bw) / nw));
      const si = (sy * bw + sx) * 2, di = (y * nw + x) * 2;
      state.warpFx[di] = bak[si];
      state.warpFx[di + 1] = bak[si + 1];
    }
  }
  uploadWarpField();
}

function download(blob) {
  if (!blob) { setStatus('导出失败'); return; }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `点睛_${Date.now()}.jpg`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  setStatus(`已导出 ${Math.round(blob.size / 1024)}KB`);
}

/* 人脸坐标等比缩放（face 里全是像素坐标） */
function faceScaled(f, k) {
  const o = Object.assign({}, f);
  ['w', 'h', 'cx', 'cy', 'top', 'chinY', 'zygoY', 'noseY', 'eyeRad', 'mouthW'].forEach((n) => {
    if (typeof f[n] === 'number') o[n] = f[n] * k;
  });
  o.eyeL = [f.eyeL[0] * k, f.eyeL[1] * k];
  o.eyeR = [f.eyeR[0] * k, f.eyeR[1] * k];
  o.mouthC = [f.mouthC[0] * k, f.mouthC[1] * k];
  return o;
}

/* 以指定长边重新上传纹理并渲染（导出用）。state.img 本身就是全分辨率 Image */
async function rebuildAt(maxSide, warpBak, warpW0, warpH0) {
  const img = state.img;
  const gl = state.gl;
  const nw = img.naturalWidth || img.width;
  const nh = img.naturalHeight || img.height;
  const k = Math.min(1, maxSide / Math.max(nw, nh));
  const w = Math.round(nw * k), h = Math.round(nh * k);

  const canvas = $('#gl');
  canvas.width = w; canvas.height = h;
  state.size = { w, h };

  initWarpField(w, h);
  if (warpBak) resampleWarp(warpBak, warpW0, warpH0);
  state.warpScale = w / (state.size0W || w);

  gl.bindTexture(gl.TEXTURE_2D, state.tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  setTexParams(gl);

  if (state.face && state.face._baseW) {
    state.face = faceScaled(state.face, w / state.face._baseW);
    applyFaceMask(state.face);
  }
  draw();
}

async function restoreAt(w, h, face, warpBak, warpW0, warpH0) {
  const gl = state.gl;
  const canvas = $('#gl');
  canvas.width = w; canvas.height = h;
  state.size = { w, h };
  initWarpField(w, h);
  if (warpBak) resampleWarp(warpBak, warpW0, warpH0);
  state.warpScale = 1;
  gl.bindTexture(gl.TEXTURE_2D, state.tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, state.img);
  setTexParams(gl);
  state.face = face;
  if (face) applyFaceMask(face);
  updateStatus();
  draw();
}
let rafPending = false;

function render() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; draw(); });
}

function draw() {
  const { gl, prog, size, values, face } = state;
  if (!gl || !state.img) return;

  gl.viewport(0, 0, size.w, size.h);
  gl.useProgram(prog);

  // 图像纹理 → unit 0
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.tex);
  gl.uniform1i(gl.getUniformLocation(prog, 'u_img'), 0);

  // 脸部蒙版 → unit 2
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, state.maskTex || state.tex);
  gl.uniform1i(gl.getUniformLocation(prog, 'u_faceMask'), 2);

  // 手动液化位移场 → unit 3
  gl.activeTexture(gl.TEXTURE3);
  gl.bindTexture(gl.TEXTURE_2D, state.warpTex || state.tex);
  gl.uniform1i(gl.getUniformLocation(prog, 'u_warp'), 3);

  // LUT → unit 1
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, state.lut || state.tex);
  gl.uniform1i(gl.getUniformLocation(prog, 'u_lut'), 1);
  gl.uniform1i(gl.getUniformLocation(prog, 'u_hasLut'), state.lut ? 1 : 0);

  gl.uniform2f(gl.getUniformLocation(prog, 'u_size'), size.w, size.h);
  gl.uniform2f(gl.getUniformLocation(prog, 'u_texel'), 1 / size.w, 1 / size.h);

  for (const k of Object.keys(MAP)) {
    const u = gl.getUniformLocation(prog, 'u_' + k);
    if (u) gl.uniform1f(u, MAP[k](values[k] ?? 0));
  }
  gl.uniform1f(gl.getUniformLocation(prog, 'u_warpScale'), state.warpScale || 1);

  if (face) {
    gl.uniform1i(gl.getUniformLocation(prog, 'u_hasFace'), 1);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_faceC'), face.cx, face.cy);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_faceW'), face.w);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_faceH'), face.h);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_zygoY'), face.zygoY);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_chinY'), face.chinY);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_eyeL'), face.eyeL[0], face.eyeL[1]);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_eyeR'), face.eyeR[0], face.eyeR[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_eyeRad'), face.eyeRad);
    gl.uniform2f(gl.getUniformLocation(prog, 'u_mouthC'), face.mouthC[0], face.mouthC[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_mouthW'), face.mouthW);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_noseY'), face.noseY);
  } else {
    gl.uniform1i(gl.getUniformLocation(prog, 'u_hasFace'), 0);
  }

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  if (!draw._diag && location.search.includes('debug')) {
    draw._diag = true;
    const err = gl.getError();
    const px = new Uint8Array(4);
    gl.readPixels(Math.floor(size.w / 2), Math.floor(size.h / 2), 1, 1,
      gl.RGBA, gl.UNSIGNED_BYTE, px);
    console.log('[diag]', { err, center: Array.from(px), size: [size.w, size.h] });
  }
}

/* ---------------------------------------------------------------- UI */
function buildSliders(pageId, defs) {
  const box = $(pageId);
  box.innerHTML = '';
  for (const [key, label, def] of defs) {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<label>${label}</label>
      <input type="range" min="0" max="100" value="${def}" data-k="${key}">
      <span class="val">${def}</span>`;
    box.appendChild(row);
    const input = row.querySelector('input');
    const val = row.querySelector('.val');
    input.addEventListener('input', () => {
      state.values[key] = +input.value;
      val.textContent = input.value;
      state.presetName = 'custom';
      markPreset('custom');
      scheduleDraw();
    });
  }
}

let drawTimer = null;
function scheduleDraw() {
  if (drawTimer) clearTimeout(drawTimer);
  drawTimer = setTimeout(() => { drawTimer = null; draw(); }, 90);
}

/* 底部四组互斥切换 */
function showGroup(g) {
  document.querySelectorAll('.group').forEach((el) => el.classList.toggle('on', el.dataset.g === g));
  ['skin', 'shape', 'tone', 'color'].forEach((k) => {
    const p = $('#p-' + k);
    if (p) p.classList.toggle('hidden', k !== g);
  });
}

function syncSliders() {
  document.querySelectorAll('input[type=range]').forEach((input) => {
    const k = input.dataset.k;
    const v = state.values[k];
    if (v == null) return;
    input.value = v;
    input.parentElement.querySelector('.val').textContent = Math.round(v);
  });
}

function setStatus(t) { $('#status').textContent = t; }
function updateStatus() {
  const { size } = state;
  setStatus(`${size.w}×${size.h}`);
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  Object.assign(state.values, p);
  applyUrlOverrides(state.values);
  syncSliders();
  draw();
  markPreset(name);
}

/* ---------------------------------------------------------------- 自定义预设 */
const PRESET_KEY = 'xt:presets';

function userPresets() {
  try { return JSON.parse(localStorage.getItem(PRESET_KEY) || '[]'); } catch (_) { return []; }
}
function setUserPresets(list) {
  try { localStorage.setItem(PRESET_KEY, JSON.stringify(list)); } catch (_) {}
}

/* 预设列表渲染进底部弹层 */
const BUILTIN_PRESETS = [['portrait', '人像'], ['natural', '自然'], ['glow', '柔光']];

function markPreset(name) {
  state.presetName = name;
  document.querySelectorAll('#presetList .pitem').forEach((el) => {
    el.classList.toggle('on', el.dataset.p === name);
  });
}

function buildPresetList() {
  const box = $('#presetList');
  if (!box) return;
  const cur = state.presetName || '';
  let html = '<div class="sheetTitle">内置</div>';
  BUILTIN_PRESETS.forEach(([k, label]) => {
    html += `<div class="pitem${cur === k ? ' on' : ''}" data-p="${k}">
      <span>${label}</span><span class="dot"></span></div>`;
  });
  const list = userPresets();
  html += '<div class="sheetTitle">我的预设</div>';
  if (!list.length) {
    html += '<div class="pitem act" style="cursor:default"><span>还没有保存过</span></div>';
  } else {
    list.forEach((p, i) => {
      html += `<div class="pitem${cur === 'u' + i ? ' on' : ''}" data-p="u${i}">
        <span>${escapeHtml(p.name)}</span>
        <span class="sub" data-del="${i}">删除</span></div>`;
    });
  }
  html += '<div class="sheetTitle">操作</div>';
  html += '<div class="pitem act" data-p="__save"><span>＋ 存为当前参数</span></div>';
  box.innerHTML = html;

  box.querySelectorAll('[data-del]').forEach((el) => {
    el.onclick = (e) => {
      e.stopPropagation();
      const i = +el.dataset.del;
      const arr = userPresets();
      const gone = (arr[i] || {}).name || '';
      arr.splice(i, 1);
      setUserPresets(arr);
      buildPresetList();
      setStatus('已删除预设 · ' + gone);
    };
  });
  box.querySelectorAll('.pitem[data-p]').forEach((el) => {
    el.onclick = async () => {
      const v = el.dataset.p;
      if (v === '__save') { openPresetBar(); return; }
      closePresetSheet();
      if (v.startsWith('u')) { await applyUserPreset(+v.slice(1)); return; }
      applyPreset(v);
    };
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function openPresetSheet() {
  buildPresetList();
  $('#presetSheet').classList.remove('hidden');
}
function closePresetSheet() {
  $('#presetSheet').classList.add('hidden');
}

function currentRecipe() {
  const r = {};
  Object.keys(MAP).forEach((k) => { r[k] = state.values[k] ?? 0; });
  r._lut = state.lutName || '';
  return r;
}

/* 命名弹层（不用 window.prompt，PWA 独立模式下更可靠） */
function openPresetBar() {
  const bar = $('#presetBar');
  if (!bar) return;
  bar.classList.remove('hidden');
  const inp = $('#presetName');
  inp.value = '';
  setTimeout(() => inp.focus(), 30);
}

function closePresetBar() {
  const bar = $('#presetBar');
  if (bar) bar.classList.add('hidden');
}

function commitPreset() {
  const name = (($('#presetName') || {}).value || '').trim() || '我的预设';
  const list = userPresets();
  const idx = list.findIndex((p) => p.name === name);
  const item = { name, recipe: currentRecipe(), at: Date.now() };
  if (idx >= 0) list[idx] = item; else list.unshift(item);
  setUserPresets(list);
  closePresetBar();
  markPreset('u' + (idx >= 0 ? idx : 0));
  setStatus('已保存预设 · ' + name);
}

async function applyUserPreset(idx) {
  const p = userPresets()[idx];
  if (!p) return;
  const r = Object.assign({}, p.recipe);
  const lut = r._lut; delete r._lut;
  Object.assign(state.values, r);
  if (lut !== undefined) {
    try { await loadLut(lut); } catch (_) {}
    document.querySelectorAll('.lutbtn').forEach((x) => x.classList.remove('on'));
    const i = LUT_LIST.findIndex(([n]) => n === (lut || ''));
    if (i >= 0) {
      const btns = document.querySelectorAll('.lutbtn');
      if (btns[i]) btns[i].classList.add('on');
    }
  }
  syncSliders();
  draw();
  markPreset('u' + idx);
  setStatus('预设 · ' + p.name);
}


async function boot() {
  try {
    initGL();
  } catch (e) {
    $('#empty').innerHTML = `<h2>无法启动</h2><p>${e.message}</p>`;
    return;
  }
  Object.keys(SLIDER_GROUPS).forEach((g) => buildSliders('#g-' + g, SLIDER_GROUPS[g]));
  buildLutGrid();
  applyPreset('portrait');

  $('#btnOpen').onclick = () => $('#file').click();
  $('#btnEmptyOpen').onclick = () => $('#file').click();
  $('#file').onchange = (e) => { const f = e.target.files[0]; if (f) loadImageFile(f); };

  $('#btnSave').onclick = () => saveExport();

  /* 导出档位：点一下循环 1600 → 2400 → 原图 */
  const EXP_STEPS = [['1600', '1600'], ['2400', '2400'], ['0', '原图']];
  state.exportSize = 1600;
  $('#expBtn').onclick = () => {
    const i = EXP_STEPS.findIndex(([v]) => +v === state.exportSize);
    const [v, label] = EXP_STEPS[(i + 1) % EXP_STEPS.length];
    state.exportSize = +v;
    $('#expBtn').textContent = label;
    setStatus('导出尺寸 · ' + (label === '原图' ? '原图' : label + 'px'));
  };

  /* 预设：底部弹层 */
  $('#btnPreset').onclick = () => openPresetSheet();
  $('#presetSheet').onclick = (e) => { if (e.target.id === 'presetSheet') closePresetSheet(); };

  /* 滤镜：切到调色组（滤镜网格在调色组底部，自己滑即可） */
  $('#btnFilter').onclick = () => showGroup('color');

  $('#presetOk').onclick = () => commitPreset();
  $('#presetCancel').onclick = () => closePresetBar();
  $('#presetName').onkeydown = (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commitPreset(); }
    if (e.key === 'Escape') { closePresetBar(); }
  };

  /* 分组切换 */
  document.querySelectorAll('.group').forEach((g) => {
    g.onclick = () => showGroup(g.dataset.g);
  });

  /* 曲线全屏面板 */
  $('#cvClose').onclick = () => $('#curvePanel').classList.add('hidden');

  // 长按看原图
  const stage = $('#stage');
  const showOriginal = (on) => {
    if (!state.img) return;
    state.values._orig = on;
    const gl = state.gl;
    // 简单实现：临时把参数清零重绘
    if (on) {
      state._saved = { ...state.values };
      const zero = {};
      for (const k of Object.keys(MAP)) zero[k] = (k === 'exposure' || k === 'contrast' ||
        k === 'saturation' || k === 'temp' || k === 'shadow' || k === 'highlight' ||
        k === 'lutAmount') ? 50 : 0;
      zero.lutAmount = 0;
      state.values = { ...state.values, ...zero };
      draw();
      $('#hint').classList.add('show');
    } else {
      state.values = state._saved;
      draw();
      $('#hint').classList.remove('show');
    }
  };
  stage.addEventListener('pointerdown', (e) => {
    if (state.warpMode) {
      const [x, y] = canvasPos(e);
      state.warpLast = [x, y];
      stage.setPointerCapture?.(e.pointerId);
      e.preventDefault();
      return;
    }
    showOriginal(true);
  });
  stage.addEventListener('pointermove', (e) => {
    if (!state.warpMode || !state.warpLast) return;
    const [x, y] = canvasPos(e);
    const dx = x - state.warpLast[0];
    const dy = y - state.warpLast[1];
    if (Math.hypot(dx, dy) > 0.6) {
      // 取负号：让画面跟着手指走
      warpPush(x, y, -dx, -dy, state.warpRadius);
      uploadWarpField();
      draw();
      state.warpLast = [x, y];
    }
    e.preventDefault();
  });
  stage.addEventListener('pointerup', (e) => {
    if (state.warpMode) {
      state.warpLast = null;
      stage.releasePointerCapture?.(e.pointerId);
      return;
    }
    showOriginal(false);
  });
  stage.addEventListener('pointercancel', () => {
    state.warpLast = null;
    if (!state.warpMode) showOriginal(false);
  });
  stage.addEventListener('pointerleave', () => {
    state.warpLast = null;
    if (!state.warpMode) showOriginal(false);
  });

  // 拖放
  ['dragover', 'drop'].forEach((t) => {
    document.addEventListener(t, (e) => e.preventDefault());
  });
  document.addEventListener('drop', (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f && f.type.startsWith('image/')) loadImageFile(f);
  });

  // 示例图
  $('#btnSample').onclick = () => loadImageURL('sample.jpg').catch(() => setStatus('示例图缺失'));

  $('#btnAuto').onclick = () => {
    if (!state.img) return;
    try {
      const r = autoTune();
      if (!r) return;
      setStatus(r.healHint ? '已按照片自适应 · 建议点一下祛痘' : '已按照片自适应');
    } catch (e) {
      setStatus('智能调参失败: ' + String(e.message || e).slice(0, 40));
    }
  };

  $('#btnWarp').onclick = () => {
    state.warpMode = !state.warpMode;
    state.warpLast = null;
    $('#btnWarp').classList.toggle('on', state.warpMode);
    $('#btnWarpClear').classList.toggle('hidden', !state.warpMode);
    stage.style.cursor = state.warpMode ? 'grab' : '';
    setStatus(state.warpMode ? '在画面上拖动，画面会跟着手走' : `人脸 ${state.face ? 1 : 0} · ${state.size.w}×${state.size.h}`);
    $('#hint').textContent = state.warpMode ? '按住往想去的方向拖' : '按住画面可看原图';
  };

  $('#btnWarpClear').onclick = () => {
    resetWarpField();
    setStatus('液化已清除');
  };

  $('#btnHeal').onclick = () => {
    if (!state.img) return;
    setStatus('找斑点…');
    setTimeout(() => {
      try {
        const cnt = healBlemishes(1.0);
        setStatus(cnt ? `祛掉 ${cnt} 处斑点` : '没找到明显斑点');
      } catch (e) {
        setStatus('祛痘失败: ' + String(e.message || e).slice(0, 40));
      }
    }, 20);
  };

  // ?warptest=1 自动模拟一次液化拖动（用来验证位移场链路）
  if (new URLSearchParams(location.search).get('warptest')) {
    setTimeout(() => {
      if (!state.face || !state.warpFx) return;
      const f = state.face;
      const r = f.w * 0.60;
      for (let k = 0; k < 3; k++) {
        warpPush(f.cx - r * 0.80, f.cy + r * 0.15, 7, 0, r * 0.50);   // 左脸颊往里推
        warpPush(f.cx + r * 0.80, f.cy + r * 0.15, -7, 0, r * 0.50);  // 右脸颊同理
      }
      uploadWarpField();
      draw();
      setStatus('warptest ok');
    }, 2000);
  }
  const openParam = new URLSearchParams(location.search).get('open');
  if (openParam) {
    loadImageURL(openParam === '1' ? 'sample.jpg' : openParam).catch(() => {});
  }

  // 换行/尺寸变化
  window.addEventListener('resize', () => draw());

  // PWA：离线缓存 + 可添加到主屏幕
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }
}

boot();
