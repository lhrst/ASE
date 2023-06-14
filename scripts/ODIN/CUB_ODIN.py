import os

magnitude = 0.0045
temperature = 1760
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
        os.system('CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 \
            python ODIN_train_test.py --a1 {a1}  --a2 {a2} --attSize {attSize} --preprocessing \
            --cuda --batch_size {batch_size} --classifier_lr {classifier_lr} --latent_size {latent_size} \
                --nclass_all {nclass_all} --ndh {ndh} --ngh {ngh} --nz {nz} --resSize {resSize} --syn_num {syn_num} \
                --image_embedding {feature} --class_embedding {attri} --dataset {dataset} \
                    --odin_temperature {odin_temperature} --odin_magnitude {odin_magnitude}'.format(a1 = a1, a2 = a2, attSize = attSize, batch_size = batch_size,
                                                                                                                          classifier_lr = classifier_lr, latent_size = latent_size, nclass_all = nclass_all,
                                                                                                                          ndh = ndh, ngh = ngh, nz = nz, resSize = resSize, syn_num = syn_num, feature = feature,
                                                                                                                          attri = attri, dataset = dataset, odin_temperature = temperature, odin_magnitude = magnitude))