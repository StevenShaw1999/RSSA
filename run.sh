CUDA_VISIBLE_DEVICES=0 python invert_gan.py --image_dir data/caricatures/images/ --stylegan2_path /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --latent_dir latent/caricatures/

CUDA_VISIBLE_DEVICES=0 python train_face_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --data_path /data/jiayu_xiao/few-shot-gan-adaptation/processed_data/caricatures/ --exp face2cari --iter 1502 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/caricatures/latent/ --task 10 --exp_name caricatures --n_train 10

CUDA_VISIBLE_DEVICES=0 python train_face_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --data_path /data/jiayu_xiao/few-shot-gan-adaptation/processed_data/caricatures/ --exp face2cari --iter 1502 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/caricatures/latent/ --task 5 --exp_name caricatures --n_train 5

python prepare_data.py --out processed_data/sketches_5 --size 256 ./data/sketches_5

CUDA_VISIBLE_DEVICES=0 python invert_gan.py --image_dir data/sketches/images/ --stylegan2_path /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --latent_dir latent/sketches/ --iter_num 4002

CUDA_VISIBLE_DEVICES=0 python train_face_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --data_path processed_data/sketches_5/  --exp face2sketches --iter 2002 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/sketches/latent/ --task 5 --exp_name sketches

CUDA_VISIBLE_DEVICES=0 python train_face_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --data_path processed_data/sketches/  --exp face2sketches --iter 2002 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/sketches/latent/ --task 10 --exp_name sketches

CUDA_VISIBLE_DEVICES=0 python invert_gan.py --image_dir data/VanGogh_face/images/ --stylegan2_path /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --latent_dir latent/VanGogh_face/ --iter_num 4002

CUDA_VISIBLE_DEVICES=0 python train_face_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --data_path processed_data/VanGogh_face_5/  --exp face2VanGogh --iter 2002 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/VanGogh_face/latent/ --task 5 --exp_name VanGogh

CUDA_VISIBLE_DEVICES=0 python train_face_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --data_path processed_data/VanGogh_face/  --exp face2VanGogh --iter 2002 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/VanGogh_face/latent/ --task 10 --exp_name VanGogh

CUDA_VISIBLE_DEVICES=0 python invert_gan.py --image_dir data/VanGogh_vil/images/ --stylegan2_path /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --latent_dir latent/VanGogh_vil/

CUDA_VISIBLE_DEVICES=0 python train_church_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --data_path processed_data/VanGogh_vil/  --exp church2VanGogh --iter 2002 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/VanGogh_vil/latent/ --task 10 --exp_name VanGogh

CUDA_VISIBLE_DEVICES=0 python train_church_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --data_path processed_data/VanGogh_vil_5/  --exp church2VanGogh --iter 2002 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/VanGogh_vil/latent/ --task 5 --exp_name VanGogh

CUDA_VISIBLE_DEVICES=0 python invert_gan.py --image_dir data/haunted_house/images/ --stylegan2_path /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --latent_dir latent/haunted_house/

CUDA_VISIBLE_DEVICES=0 python train_church_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --data_path processed_data/haunted_house/  --exp church2haunted --iter 2002 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/haunted_house/latent/ --task 10 --exp_name haunted

CUDA_VISIBLE_DEVICES=0 python train_church_proj.py --ckpt /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --data_path processed_data/haunted_house_5/  --exp church2haunted --iter 1302 --self_corr_loss --proj --dis_corr_loss --latent_dir latent/haunted_house/latent/ --task 5 --exp_name haunted

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --ckpt_target checkpoints/face2sketches_self_dis_proj_10/final.pt --task 10 --source face --target sketches --latent_dir latent/sketches/latent/ --mode viz_images

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --ckpt_target checkpoints/face2sketches_self_dis_proj_5/final.pt --task 5 --source face --target sketches --latent_dir latent/sketches/latent/ --mode viz_imgs

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --ckpt_target checkpoints/face2VanGogh_self_dis_proj_10/final.pt --task 10 --source face --target VanGogh --latent_dir latent/VanGogh_face/latent/ --mode viz_imgs

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --ckpt_target checkpoints/face2VanGogh_self_dis_proj_5/final.pt --task 5 --source face --target VanGogh --latent_dir latent/VanGogh_face/latent/ --mode viz_imgs

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --ckpt_target checkpoints/face2cari_self_dis_proj_10/final.pt --task 10 --source face --target caricatures --latent_dir latent/caricatures/latent/ --mode viz_imgs

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/source_ffhq.pt --ckpt_target checkpoints/face2cari_self_dis_proj_5/final.pt --task 5 --source face --target caricatures --latent_dir latent/caricatures/latent/ --mode viz_imgs

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --ckpt_target checkpoints/church2haunted_self_dis_proj_5/final.pt --task 5 --source church --target haunted --latent_dir latent/haunted_house/latent/ --mode viz_imgs

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --ckpt_target checkpoints/church2haunted_self_dis_proj_10/final.pt --task 10 --source church --target haunted --latent_dir latent/haunted_house/latent/ --mode viz_imgs

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --ckpt_target checkpoints/church2VanGogh_self_dis_proj_10/final.pt --task 10 --source church --target VanGogh --latent_dir latent/VanGogh_vil/latent/ --mode viz_imgs

CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_source /data/jiayu_xiao/few-shot-gan-adaptation/checkpoints_ori/church.pt --ckpt_target checkpoints/church2VanGogh_self_dis_proj_5/final.pt --task 5 --source church --target VanGogh --latent_dir latent/VanGogh_vil/latent/ --mode viz_imgs
