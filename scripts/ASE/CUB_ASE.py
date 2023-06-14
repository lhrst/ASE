import os


count_ASE = 1250
unknown_num = 5
iter_num = 600
distance_weight = 5
ASE_speed = 2
use_energy_cfloss = 1
energy_temp = 1000

a1 = 1
a2 = 1
attSize = 312
batch_size = 64
classifier_lr = 0.001
latent_size = 312
nclass_all = 200
ndh = 4096
ngh = 4096
nz = 312
resSize = 2048
syn_num = 300
  

feature = "res101"
attri = "att_2022"
dataset = "CUB"

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