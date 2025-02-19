# 🤖🌊 Learning Garment Manipulation Policies towards Robot-Assisted Dressing

![pipeline](doc/pipeline.png "pipeline")

[![Static Badge](https://img.shields.io/badge/arXiv-2409.01083-B31B1B?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2409.01083)

A reference implementation for the paper at Science Robotics `Learning garment manipulation policies toward robot-assisted dressing`.

* Paper page: https://www.science.org/doi/10.1126/scirobotics.abm6010
* Project page: https://fan6zh.github.io/sr/
* Author: Fan Zhang (f.zhang16@imperial.ac.uk), Yiannis Demiris


#### This repo contains
* __tactile tracing__
    * Tactile garment edge tracing for unfolding.
* __ridgeback__
    * Combine baxter and ridgeback urdf.
* __active pre-grasp manipulation__
    * It integrates pytorch, openai gym and blender sumulation.
* __self-supervised learning garment physics__
    * It follows the framework of simCLR.
* __robot pipeline__
    * It has been tested with Baxter robot, Clearpath Ridgeback and Robotic grippers.
* __pipeline in simulation__
    * It is bult in Blender simulation.

    
#### Dependencies
* pytorch-blender (https://github.com/cheind/pytorch-blender)
* hospital bed and rail models could be found here: https://www.turbosquid.com/3d-model/free/bed/obj
* robotiq gripper models could be found here: https://robotiq.com/products/2f85-140-adaptive-robot-gripper?ref=nav_product_new_button
* Gelsight mini (https://github.com/gelsightinc/gsrobotics)
