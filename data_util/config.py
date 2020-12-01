import os

root_dir = "data"
train_data_path = os.path.join(root_dir, "chunked/train_*")
eval_data_path = os.path.join(root_dir, "train_8000.bin")
decode_data_path = os.path.join(root_dir, "train_8000.bin")
vocab_path = os.path.join(root_dir, "vocab")

# train_data_path = os.path.join(root_dir, "val.bin")
# eval_data_path = os.path.join(root_dir, "val.bin")
# decode_data_path = os.path.join(root_dir, "val.bin")
# vocab_path = os.path.join(root_dir, "vocab")
log_root = os.path.join("model_data", "log")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 8
max_enc_steps=400 # input tokens of articles
max_dec_steps=100 # output summaries
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = True
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 7 #500000

use_gpu=True

lr_coverage=0.15