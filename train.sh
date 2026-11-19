export CUDA_VISIBLE_DEVICES=1
#${1}
# CMPF（完整表达式 + 自底向上门控解码器）：仅使用物理 GPU 1。
# python train.py --dataset refsegrs --ngpu 1 --time 17 --savename coupled_gated_decoder_refsegrs --visulize 0 --batch_size 4 --nb_epoch 55 --lr 5e-5
python train.py --dataset risbench --ngpu 1 --time 17 --savename coupled_gated_decoder_risbench --visulize 0 --batch_size 8 --nb_epoch 40 --lr 3e-5
# python train.py --dataset rrsisd --ngpu 1 --time 17 --savename coupled_gated_decoder_rrsisd --visulize 0 --batch_size 8 --nb_epoch 40 --lr 3e-5
