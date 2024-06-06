# ComfyUI_densediffusion
DenseDiffusion custom node for ComfyUI. Implements the [DenseDiffusion](https://github.com/naver-ai/DenseDiffusion)-like method for regional prompt used in [Omost](https://github.com/lllyasviel/Omost) project.

## What this repo implements
Normal attention calculation can be written as `y=softmax(q@k)@v`. DenseDiffusion introduces the method of attention manipulation on `q@k`, which makes the expression look like `y=softmax(modify(q@k))@v`.
The original DenseDiffusion's implementation does not perform very well according to my testing so here I only implemented the version used in Omost repo. Refer to https://github.com/lllyasviel/Omost#regional-prompter for other regional prompt methods.

## How to use
![image](https://github.com/huchenlei/ComfyUI_densediffusion/assets/20929282/d75c1354-8f62-4e84-9b9c-67698e2a5f32)

## Limitation [IMPORTANT]
Currently ComfyUI's attention replacements do not compose with each other, so this regional prompt method does not compose with IPAdapter. I am currently working on a universal model patcher to solve this issue.
