import os

count_ASE = 8000
unknown_num = 50
iter_num = 800
distance_weight = 5
ASE_speed = 15
use_energy_cfloss = 1
energy_temp = 5

a1 = 0.01
a2 = 0.01
attSize = 85
batch_size = 64
classifier_lr = 0.001
latent_size = 85
nclass_all = 50
ndh = 4096
ngh = 4096
nz = 85
resSize = 2048
syn_num = 1800
  

feature = "res101"
attri = "att"
dataset = "AWA2"

for i in range(1):
    for j in range (1):
        os.system('CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
            python ASE_train_test.py --a1 {a1}  --a2 {a2} --attSize {attSize} --preprocessing \
            --cuda --batch_size {batch_size} --classifier_lr {classifier_lr} --latent_size {latent_size} \
                --nclass_all {nclass_all} --ndh {ndh} --ngh {ngh} --nz {nz} --resSize {resSize} --syn_num {syn_num} \
                --image_embedding {feature} --class_embedding {attri} --dataset {dataset} \
                    --count_ASE {count_ASE} --unknown_num {unknown_num} --iter_num {iter_num} --distance_weight {distance_weight} \
                        --ASE_speed {ASE_speed}  --energy_temp {energy_temp}'.format(a1 = a1, a2 = a2, attSize = attSize, batch_size = batch_size,
                                                                                                                          classifier_lr = classifier_lr, latent_size = latent_size, nclass_all = nclass_all,
                                                                                                                          ndh = ndh, ngh = ngh, nz = nz, resSize = resSize, syn_num = syn_num, feature = feature,
                                                                                                                          attri = attri, dataset = dataset, count_ASE = count_ASE, unknown_num = unknown_num, iter_num = iter_num,
                                                                                                                          distance_weight = distance_weight, ASE_speed = ASE_speed, energy_temp = energy_temp))