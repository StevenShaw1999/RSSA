
# Few Shot Generative Model Adaption via Relaxed Spatial Structural Alignment

## Generating & Testing
We provide the pre-trained models for different source and target GAN models. Download the model from this Google Drive link. Store the source model into the `./checkpoints_ori` directory and the target model into the `./checkpoints` directory.

### Generate images
To generate images from a pre-trained source GAN and target GAN, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /path/to/source_model/ --ckpt_target /path/to/target_model/ --task 10(5) --source source_domain --target target_domain --latent_dir /path/to/latent/ --mode viz_imgs
```

This will save synthesis samples into `./viz_img` directory. Use the `--load_noise` option to use the noise vectors used for some samples shown in the main paper. For example:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source ./checkpoints_ori/face.pt --ckpt_target ./checkpoints/face2sketches_self_dis_proj_10/final.pt --task 10 --source face --target sketches --latent_dir latent/sketches/latent/ --mode viz_imgs --load_noise noise.pt
```

### Generate interpolation images
To generate interpolation images from source and target GAN, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /path/to/source_model --ckpt_target /path/to/target_model/ --task 10(5) --source source_domain --target target_domain --latent_dir /path/to/latent/ --mode viz_gif --load_noise /path/to/noise_vector/
```

This will save synthesis interpolation images (for source and target) into `./viz_gif` directory. For example:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source ./checkpoints_ori/face.pt --ckpt_target ./checkpoints/face2VanGogh_self_dis_proj_10/final.pt --task 10 --source face --target VanGogh --latent_dir latent/VanGogh_face/latent/ --mode viz_gif
```

### Evaluating Inception Score
First use the trained target model to synthesis images via the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_target /path/to/target_model/ --task 10(5) --source source_domain --target target_domain --latent_dir /path/to/latent/ --mode eval_IS
```
This will synthesis 1000 samples for target domain by default and save them into the `./eval_IS` directory. For example:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_target ./checkpoints/face2VanGogh_self_dis_proj_10/final.pt --task 10 --source face --target VanGogh --latent_dir latent/VanGogh_face/latent/ --mode eval_IS
```
Then run `eval.py` to calculate the mean Inception Score for the synthesis images as below. Note to verify the save directory of the images synthesis by the above command.
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --mode IS --img_pth /path/to/eval4IS/images
```
For example:
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --mode IS --img_pth ./eval_IS/face2VanGogh_10
```

### Evaluating SCS Score
First generate cross-domain image pairs via the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /path/to/source_model/ --ckpt_target /path/to/target_model/ --task 10(5) --source source_domain --target sketches --latent_dir /path/to/latent/ --mode eval_SCS --SCS_samples n
```
This will save synthesis cross-domain pairs in `./eval_SCS/` (500 pairs by default), verify the task directory (`./eval_SCS/church2VanGogh_10` for example) and run `eval.py` with following command to get the edges for the synthesis pairs and calculate the mean pair-wise SCS:
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --mode SCS --img_pth /path/to/synthesis/pairs/
```
For example:
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --mode SCS --img_pth eval_SCS/church2VanGogh_10
```



## Training (adapting) your own GAN
### Data preparation
The raw images are saved in the `./data/` directory, and the processed images are saved in the `./processed_data/` directory. If you want to train model on your own data, save them in `./data/` and run `prepare_data.py` to preprocess your raw data as follow:

- `python prepare_data.py --out processed_data/<dataset_name> --size 256 ./data/<dataset_name>`

### GAN inversion
First invert the training images using `invert_gan.py`, we also provide inverted latent code in the `./latent/` directory. The image2stylegan code base in this repo does not guarantee good reconstruction performance (shown in the `latent/target_domain/images`), you can use your own inversion method if you get better results. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python invert_gan.py --image_dir data/caricatures/images/ --stylegan2_path checkpoints_ori/face.pt --latent_dir latent/caricatures/
```

### Running examples
```bash
CUDA_VISIBLE_DEVICES=0 python train_church_proj.py --size 256 --ckpt checkpoints_ori/church.pt --data_path processed_data/vangogh_houses10/ --exp church_to_van_gogh_10_scc_proj_dcc --iter 2502 --n_train 10 --task 10 --exp_name van_gogh --proj Yes --self_sim_loss_new Yes --sp_inter_sim Yes
```

### Sample images from a model

To generate images from a pre-trained GAN, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_target /path/to/target_model/ --ckpt_source /path/to/source_model/ --exp_name van_gogh --source church --task 10
```

This will save the images in the `test_samples/` directory.

