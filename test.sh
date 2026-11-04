#ngpu=$1
#CUDA_VISIBLE_DEVICES=${ngpu}
export CUDA_VISIBLE_DEVICES=1
# CMPF（完整表达式 + 门控解码器）评测：验证集 + 测试集。
# python test.py --dataset rrsisd --pretrain /home/ubuntu/glk/code/private-code/any_try/InferSeg/Lab2/saved_models/couple_decoder/co_de_rrsisd_best.pth --visulize 0
# python test.py --dataset refsegrs --pretrain /home/ubuntu/glk/code/private-code/any_try/InferSeg/Lab2/saved_models/couple_decoder/coupe_de_refsegrs_best.pth --visulize 0
python test.py --dataset risbench --pretrain /home/ubuntu/glk/code/private-code/any_try/InferSeg/Lab2/saved_models/couple_decoder/coupe_de_risbench_best.pth --visulize 0
