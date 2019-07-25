# README

## Introduction

VRenderer is a PT/PM renderer based on OptiX and CUDA, accelerated by RTX tech of NVIDIA.

![box](./Pics/box.PNG)

![renderer](./Pics/scene_editor.PNG)

## BUILD

Only support windows 10, x64.

To build it, you will need to install OptiX and CUDA first.

Make sure your GPU driver is the newest!

Git bash in Root dir, then:

`./FindOptiX.sh 'YOUR_OPTIX_INSTALL_ROOT_DIR'                 `

Open CMake, click Configure, change the generator of CMake to VS2017/2019 Win64. After everything is done, click Generate.

Open VRenderer.sln and build the project.

## Log
* 2019.7.25
  1. add “Handle” to scene editor
* 2019.7.24
  1. add “Hierarchy” and “Inspector”
  2. fix bug of “default_light”
* 2019.7.23
  1. add “pick up”
  2. fix bug of ".gitignore"
* 2019.7.20
  1. add scene description file, with an example scene "Cornell Box"
  2. add camera support
* 2019.7.19
  1. new “pipeline” feature
  2. refactor underlying
