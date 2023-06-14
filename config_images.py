import argparse

parser = argparse.ArgumentParser()



parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--work_dir', default='./expr', help='path to save the log files')
parser.add_argument('--model_dir', default='./output', help='path to save the models')
parser.add_argument('--dataroot', default='datasets', help='path to dataset')
parser.add_argument('--image_embedding', default='image.png101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG_D2', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaD2', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument("--latent_size", type=int, default=312)
parser.add_argument("--conditional", action='store_true',default=True)
###

parser.add_argument('--a1', type=float, default=1.0)
parser.add_argument('--a2', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=1.0, help='recons_weight for decoder')
parser.add_argument('--feedback_loop', type=int, default=2)
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')

#ASE
parser.add_argument('--count_ASE', type=int, default=256, help='Number of ASEs(unknown attributes) to generate')
parser.add_argument('--unknown_num', type=int, default=10, help='Number of adversarial features every unknown attributes to generate')
parser.add_argument('--iter_num', type=int, default=500, help='ASE iters')
parser.add_argument('--distance_weight', type=float, default=0.5, help='Weight for latent distance loss [default: 1]')
parser.add_argument('--ASE_speed', type=float, default=.1, help='Learning rate for ASE descent [default: .1]')
parser.add_argument('--energy_temp', type=int, default=1, help='energy temperature of ASE')

#Energy
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--m_in', type=float, default=-23)
parser.add_argument('--m_out', type=float, default=-5)
parser.add_argument('--gamma1', type=float, default=0.1, help='energy loss weight')

#Placeholder
parser.add_argument('--alpha1', type=float, default=1, help='the loss2 weight')
parser.add_argument('--alpha2', type=float, default=1, help='the loss3 weight')
parser.add_argument('--placeholder_temp', type=float, default=1024, help='the placeholder temperature')

#ODIN
parser.add_argument('--odin_temperature', type=float, default=1000, help='odin temperature')
parser.add_argument('--odin_magnitude', type=float, default=0.1, help='odin magnitude')

#OpenMax
parser.add_argument('--openmax_alpha', type=float, default=10, help='openmax alpha')
parser.add_argument('--openmax_eta', type=float, default=10, help='openmax eta')

#Ablation
parser.add_argument('--ablation_method', type = str, default='counterfactual')
parser.add_argument('--cf_batch_size', type=int, default=64,  help='cf_batchsize')

opt = parser.parse_args()
opt.lambda2 = opt.lambda1
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize
opt.latent_size = opt.attSize


