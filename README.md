
## Run

Our source is heavily based on Ojha's code base, so download it from `https://github.com/utkarshojha/few-shot-gan-adaptation` first and put our scripts into its folder.

Download the datasets from `https://github.com/utkarshojha/few-shot-gan-adaptation` and prepare the dataset utilizing `prepare_data.py`

Download the pre-trained models for different source domains from Ojha's link. 
| Source GAN: G<sub>s</sub> | Target GAN: G<sub>s&#8594;t</sub> |
| ------------------------- | --------------------------------- |
| [FFHQ](https://drive.google.com/file/d/1TQ_6x74RPQf03mSjtqUijM4MZEMyn7HI/view?usp=sharing) | [[Sketches](https://drive.google.com/file/d/1Qkdeyk_-1pqgvrIFy6AzsSYNgZMtwKX3/view?usp=sharing)] [[Caricatures](https://drive.google.com/file/d/1CX8uYEWqlZaY7or_iuLp3ZFBcsOOXMt8/view?usp=sharing)] [[Amedeo Modigliani](https://drive.google.com/file/d/1WvBtThEakKNqNFBCuHHoNNI1jojFAvan/view?usp=sharing)] [[Babies](https://drive.google.com/file/d/1d5JNwQBSyFaruAoLZBlXFVPc_I6WZjhm/view?usp=sharing)] [[Sunglasses](https://drive.google.com/file/d/1D6HOdcHG4j6kQmOCjwQakK7REgykPOYy/view?usp=sharing)] [[Rafael](https://drive.google.com/file/d/1K6xWnlfQ-qT_I_QTY8SiQ9fvRylMFeND/view?usp=sharing)] [[Otto Dix](https://drive.google.com/file/d/1I8gmuiDcARmwZNimlYEalPsKcRot-ijZ/view?usp=sharing)] |
| [LSUN Church](https://drive.google.com/file/d/18NlBBI8a61aGBHA1Tr06DQYlf-DRrBOH/view?usp=sharing) | [[Haunted houses]()] [[Van Gogh houses]() [[Landscapes]()] [[Caricatures]()] |
| [LSUN Cars](https://drive.google.com/file/d/1O-yWYNvuMmirN8Q0Z4meYoSDtBfJEjGc/view?usp=sharing) | [[Wrecked cars]()] [[Landscapes]()] [[Haunted houses]()] [[Caricatures]()] | 
| [LSUN Horses](https://drive.google.com/file/d/1ED4JPQsxnBUMFHiooCL7oK2x4FfIf-bt/view?usp=sharing) | [[Landscapes]()] [[Caricatures]()] [[Haunted houses]()] |
| [Hand gestures](https://drive.google.com/file/d/1LBXphEMT8C2JJ3AXf2CETeIFvoAz5n2T/view?usp=sharing) | [[Google Maps]()] [[Landscapes]()] |

Download the pre-trained model(s), and store it into `./checkpoints_ori` directory.


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

