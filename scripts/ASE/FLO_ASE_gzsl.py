import os

count_ASE = 1000
unknown_num = 1
iter_num = 200
distance_weight = 0.01
ASE_speed = 0.1
use_energy_cfloss = 1
energy_temp = 1

a1 = 0.5
a2 = 0.5
attSize = 1024
batch_size = 64
classifier_lr = 0.001
latent_size = 1024
nclass_all = 102
ndh = 4096
ngh = 4096
nz = 1024
resSize = 2048
syn_num = 1200
  

feature = "res101"
attri = "att"
dataset = "FLO"

for i in range(1):
    for j in range (1):
        os.system('CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
            python ASE_train_test.py --gzsl --a1 {a1}  --a2 {a2} --attSize {attSize} --preprocessing \
            --cuda --batch_size {batch_size} --classifier_lr {classifier_lr} --latent_size {latent_size} \
                --nclass_all {nclass_all} --ndh {ndh} --ngh {ngh} --nz {nz} --resSize {resSize} --syn_num {syn_num} \
                --image_embedding {feature} --class_embedding {attri} --dataset {dataset} \
                    --count_ASE {count_ASE} --unknown_num {unknown_num} --iter_num {iter_num} --distance_weight {distance_weight} \
                        --ASE_speed {ASE_speed}  --energy_temp {energy_temp}'.format(a1 = a1, a2 = a2, attSize = attSize, batch_size = batch_size,
                                                                                                                          classifier_lr = classifier_lr, latent_size = latent_size, nclass_all = nclass_all,
                                                                                                                          ndh = ndh, ngh = ngh, nz = nz, resSize = resSize, syn_num = syn_num, feature = feature,
                                                                                                                          attri = attri, dataset = dataset, count_ASE = count_ASE, unknown_num = unknown_num, iter_num = iter_num,
                                                                                                                          distance_weight = distance_weight, ASE_speed = ASE_speed, energy_temp = energy_temp))