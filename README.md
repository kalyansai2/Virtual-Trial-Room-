
# Virtual Trial Room – Garment Try-On and Recoloring (CatVTON + SDXL) 

This project builds a Virtual Try-On system using the VITON-HD dataset, CatVTON inference, and a custom SDXL Inpainting recoloring module. The pipeline supports garment segmentation, recoloring, try-on synthesis, and evaluation using four standardized metrics.


## Create a Python Environment

1. Create a Python Environment

```bash
conda create -n vtr python=3.10 -y
conda activate vtr
```
2. Install Dependencies

```bash
pip install -r requirements.txt
```
This installs PyTorch, Diffusers (SDXL), Detectron2 dependencies, and evaluation libraries.




## Dataset Setup

We use the VITON-HD dataset for virtual try-on.

Dataset link:
https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view

After downloading, structure it as:

```bash
data/
  VITON-HD/
    test/
      image/
      cloth/
      cloth-mask/
    train/
```




## CatVTON Inference 

To generate full try-on results:

```bash 
python src/inference.py \
  --input_dir data/VITON-HD/test \
  --output_dir results/
```
## Recoloring the Required Garment 

The sd_recolor.py script recolors only the garment region while preserving identity, pose, and background using Stable Diffusion XL.

```bash 
CUDA_VISIBLE_DEVICES=0 python src/sd_recolor.py \
  --input_image data/VITON-HD/test/image/00013_00.jpg \
  --mask_image data/VITON-HD/test/cloth-mask/00013_00.jpg \
  --prompt "replace the shirt color with bright red, keep everything else identical" \ #can specify any prompt as required 
  --output_dir results/vitonhd-recolored \
  --num_steps 40 \
  --guidance_scale 12 \
  --strength 0.9
  ```
Prompt can be of specification as per user requirements for color change

Outputs are auto-named as:

```bash 
recolored_<id>.png
```

# Explanation of Key SDXL Arguments
```bash
--num_steps        Number of diffusion steps (40 = high-quality recoloring)
--guidance_scale   How strongly SDXL follows the text prompt (12 = strong)
--strength         How much of the masked region is modified (0.9 = heavy recolor)
```



## Evaluation Metrics

Our pipeline computes four common VTON metrics:


| Metric | Description | Goal |
|--------|-------------|------|
| **FID** | Measures realism of generated images | Lower is better |
| **SSIM** | Measures structural similarity | Higher is better |
| **LPIPS** | Perceptual similarity using deep features | Lower is better |
| **KID** | Stable version of FID for small datasets | Lower is better |

Running the evaluation

```bash 
python src/plot_metrics.py \
  --pred results/ \
  --gt data/VITON-HD/test/image \
  --output_dir results/
```

These two plots in results visualize the performance of our virtual try-on pipeline.  
The bar chart shows the individual metric values, while the radar plot provides a holistic comparison across four dimensions:
- FID (realism)
- SSIM (structural consistency)
- LPIPS (perceptual similarity)
- KID (kernel-based realism)

Together, they give a complete picture of image quality and model behavior.

## References

CatVTON
```bash
@misc{chong2024catvtonconcatenationneedvirtual,
 title={CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models}, 
 author={Zheng Chong and Xiao Dong and Haoxiang Li and Shiyue Zhang and Wenqing Zhang and Xujie Zhang and Hanqing Zhao and Xiaodan Liang},
 year={2024},
 eprint={2407.15886},
 archivePrefix={arXiv},
 primaryClass={cs.CV},
 url={https://arxiv.org/abs/2407.15886}, 
}
```

VITON-HD Dataset
https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view




