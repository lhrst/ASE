import os

alpha = 10
eta = 10

a1 = 0.1
a2 = 0.01
attSize = 102
batch_size = 64
classifier_lr = 0.0005
latent_size = 102
nclass_all = 717
ndh = 4096
ngh = 4096
nz = 102
resSize = 2048
syn_num = 400
  

feature = "res101"
attri = "att"
dataset = "SUN"

for i in range(1):
    for j in range (5):
        os.system('CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 \
            python OpenMax_train_test.py --gzsl --a1 {a1}  --a2 {a2} --attSize {attSize} --preprocessing \
            --cuda --batch_size {batch_size} --classifier_lr {classifier_lr} --latent_size {latent_size} \
                --nclass_all {nclass_all} --ndh {ndh} --ngh {ngh} --nz {nz} --resSize {resSize} --syn_num {syn_num} \
                --image_embedding {feature} --class_embedding {attri} --dataset {dataset} \
                    --openmax_alpha {openmax_alpha} --openmax_eta {openmax_eta}'.format(a1 = a1, a2 = a2, attSize = attSize, batch_size = batch_size,
                                                                                                                          classifier_lr = classifier_lr, latent_size = latent_size, nclass_all = nclass_all,
                                                                                                                          ndh = ndh, ngh = ngh, nz = nz, resSize = resSize, syn_num = syn_num, feature = feature,
                                                                                                                          attri = attri, dataset = dataset, openmax_alpha = alpha, openmax_eta = eta))
