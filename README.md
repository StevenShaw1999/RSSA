
# Few Shot Generative Model Adaption via Relaxed Spatial Structural Alignment

## Generating & Testing
We provide the pre-trained models for different source and target GAN models. Download the model from this Google Drive link. Store the source model into the `./checkpoints_ori` directory and the target model into the `./checkpoints` directory.

### Generate
To generate images from a pre-trained GAN, run the following command:





## Training (adapting) your own GAN
Invert the training images and save them in `./exps/` using `invert_gan.py`, you need to change the path in the scripts according to your own environments.
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

