## Run the code on the custom dataset

First, we take THUMAN-DEMO1 as an example and assume that the input data contains multi-view images, masks and camera parameters.

```
thuman_demo/
├── img
│   └── 0001
│       ├── xxx.png  --- the image of view xxx
├── mask
│   └── 0001
│       ├── xxx.png  --- the mask of view xxx
└── parameter        --- We use perspecive camera model to render images
    └── 0001   
        ├── xxx_extrinsic.npy  --- the extrinsic of view xxx (3x4 world-to-camera matrix)
        ├── xxx_intrinsic.npy  --- the intrinsic of view xxx (3x3 matrix)
```

### Camera parameters

Our camera model is the same with OpenCV camera model (without distortion parameters). Please see [this link](https://learnopencv.com/geometry-of-image-formation/) to fully understand our camera model.

### Reconstruct a coarse model

Next, you can reconstruct a coarse model by VisualHull, Multi-view PIFu or DoubleField(recomended) and convert it to an OBJ model. Here, we provide an initial model recovered by DoubleField for THUman-DEMO1 in [this link](https://mailstsinghuaeducn-my.sharepoint.com/:u:/g/personal/shaorz20_mails_tsinghua_edu_cn/EfnD0PHZmNROqxKrFpbNtScB1eGfcoFEnKoLs720rPq8pA?e=wUgR2A).

### Render depth and normal maps

Then you need render depth and normal maps for the coarse model. You can directly use our taichi code to render them:

```
# Install Taichi and Taichi-glsl
taichi==0.6.39 or 0.7.15
taichi_glsl==0.0.10

# Move OBJ model to assets/ directory
mv xxx.obj assets/

# Render depth and normal maps using taichi code
python taichi_render_gpu/render_depth_map.py --dataroot dataset/thuman_demo/ --obj_path assets/ --obj_format inference_eval_%s_0.obj --is_2k \
    --yaw_list 0 20 90 110 180 200 270 290
```

**Please notice that our depth map records `1/Z` value rather than `Z` value.** More details can be found in `taichi_render_gpu/taichi_three/geometry`. 

### Run our DiffuStereo code

After rendering depth and normal maps, you can run our DiffuStereo code to reconstruct high-quality depth point cloud as mentioned in the `README.md`.