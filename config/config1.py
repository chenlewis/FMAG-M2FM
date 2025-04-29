import torch as t
import warnings

class DefaultConfig(object):
    env = 'default'
    vis_port = 8097

    model = 'ViTB16'
    # model = 'SwinB'
    # model = 'BeiTB'
    # model = 'DiNATB'
    # model = 'FocalNetB'
    # model = 'ConvNeXtTiny'


    '''
    CNN路径加载
    '''
    # train_data_root_cnn_1 = '/home/data1/lyj/ROD/images_with_moire_FMAG/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM1/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_Train/DM_0/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_Train/DM_Copy/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_Train/DM_Copy1/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_Train/DM_TMTP/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_Train/DM_TMTP1/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_Train/DM_TMTP_dot/images/'
    train_data_root_cnn_1 = '/pubdata/lipeiquan/lpq_dataset/DM_Cyclegan/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_Train/DM_Cyclegan/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_Train/DM_AUG0/images/'
    # train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_Train/DM1_hpf/images/'

    # test_data_root = '/home/data1/lyj/CMA/SWOMP_s/images/'
    # test_data_root = '/home/data1/lyj/CMA/SRDID162_patch/images/'
    # test_data_root = '/home/data1/lyj/CMA/SwMP_patch/images/'
    test_data_root = '/home/data1/lyj/CMA/SwoMP_patch/images/'
    # test_data_root = '/pubdata/lipeiquan/lpq_dataset/SwoMP_patch/images/'
    # test_data_root = '/home/data1/lyj/CMA/DLC/images/'
    # test_data_root = '/pubdata/lipeiquan/lpq_dataset/DLC/images/'

    # test_data_root = '/home/data1/lyj/CMA/SRDID162_patch_mini/images/'
    # test_data_root = '/pubdata/lipeiquan/lpq_dataset/SRDID162_patch_mini/images/'
    # test_data_root = '/home/data1/lyj/CMA/SwoMP_mini_patch/images/'
    # test_data_root = '/home/data1/lyj/CMA/SwMP_mini_patch/images/'
    # test_data_root = '/pubdata/lipeiquan/lpq_dataset/SwoMP_mini_patch/images/'
    # test_data_root = '/home/data1/lyj/ROD/M&F/images/'


    # load_model_path = '/home/lyj/MFM-master/models/ViTB16_FAG.pth'
    # load_model_path = '/home/lyj/CNN/weight/TMTP/ViT1/ViT-15.pth'
    # load_model_path = '/home/lyj/CNN/weight/TMTP/SwinB1/Swin-19.pth'
    # load_model_path = '/home/lyj/CNN/weight/TMTP1/SwinB/Swin-14.pth'
    # load_model_path = '/home/lyj/CNN/weight/HPF/SwinB/Swin-4.pth'
    load_model_path = '/home/lyj/CNN/weight/HPF/ViT/ViT-4.pth'

    # load_model_path = '/home/lyj/CNN/weight/TMTP_dot/ViT/ViT-12.pth'
    # load_model_path = '/home/lyj/CNN/weight/Original/ViT1/ViT-11.pth'
    # load_model_path = '/home/lyj/CNN/weight/Original/Swin1/Swin-2.pth'
    # load_model_path = '/home/lyj/CNN/weight/FMAG/ViT/ViT-9.pth'
    # load_model_path = '/home/lyj/CNN/weight/FMAG/SwinB/Swin-0.pth'

    # load_model_path = '/home/lyj/CNN/weight/Cyclegan/ViT/ViT-10.pth'
    # load_model_path = '/home/lyj/CNN/weight/Cyclegan/SwinB1/Swin-1.pth'
    # load_model_path = '/home/lyj/CNN/weight/Cyclegan/BeiTB/BeiT-9.pth'
    # load_model_path = '/home/lyj/CNN/weight/Cyclegan/FocalNetB/FocalNet-4.pth'
    # load_model_path = '/pubdata/lipeiquan/CNN/weight/Cyclegan/DiNATB/DiNAT-6.pth'
    # load_model_path = '/home/lyj/CNN/weight/Cyclegan/ConvNeXtTiny/ConvNeXtTiny-4.pth'

    # load_model_path = '/home/lyj/CNN/weight/FMAG/ViT/ViTB16_FAG.pth'
    # load_model_path = '/home/lyj/CNN/weight/FMAG/SwinB/SwinB_FAG.pth'

    batch_size_0 = 128
    batch_size_1 = 64

    use_gpu = True
    use_multi_gpu = False

    num_workers = 0
    print_freq = 20
    debug_file = ''
    ##test
    # result_file = '/home/data1/lyj/CMA/test/SRDID162/FMAG/DMRODAUG0/Vit/Train2/SRDID162_Vit_FMAG_2.csv'
    # result_file = '/home/lyj/code_lpq/test/SRDID162/FMAG/M&F&SRDID162_ViTB16_FAG.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/FAG-ViT-SwoMP_mini_patch.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/FAG-ViT-162.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT1/Ori-ViT11-SwoMP.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT1/Ori-ViT11-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB1/Ori-Swin2-SwoMP.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB1/Ori-Swin2-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT1/TMTP-ViT1-SwoMP.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT1/TMTP-ViT15-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB1/TMTP1-Swin1-SwoMP.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB1/TMTP1-Swin1-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB/HPF-Swin4-SwoMP.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB/HPF-Swin4-DLC.csv'
    result_file = '/home/lyj/MFM-master/results/ViT/HPF-ViT4-SwoMP.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/HPF-ViT4-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/Cyclegan-ViT10-SwMP.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/Cyclegan-ViT10-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB1/Cyclegan-Swin5-SwoMP.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB1/Cyclegan-Swin5-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/FocalNetB/Cyclegan-FocalNet4-SwoMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/FocalNetB/Cyclegan-FocalNet4-SwoMP.csv'
    # result_file = '/home/lyj/MFM-master/results/FocalNetB/Cyclegan-FocalNet4-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/BeiTB/Cyclegan-BeiT9-SwoMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/BeiTB/Cyclegan-BeiT9-SwoMP.csv'
    # result_file = '/home/lyj/MFM-master/results/BeiTB/Cyclegan-BeiT9-DLC.csv'
    # result_file = '/pubdata/lipeiquan/CNN/results/DiNATB/Cyclegan-DiNAT6-SwoMP_mini.csv'
    # result_file = '/pubdata/lipeiquan/CNN/results/DiNATB/Cyclegan-DiNAT6-SwoMP.csv'
    # result_file = '/pubdata/lipeiquan/CNN/results/DiNATB/Cyclegan-DiNAT6-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/ConvNeXtTiny/Cyclegan-ConvNeXtTiny4-SwoMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/ConvNeXtTiny/Cyclegan-ConvNeXtTiny4-SwMP.csv'
    # result_file = '/home/lyj/MFM-master/results/ConvNeXtTiny/Cyclegan-ConvNeXtTiny4-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/ViTB16_FAG-162_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB/FMAG-Swin0-SwMP_mini.csv'

    # result_file = '/home/lyj/MFM-master/results/ViT1/Ori-ViT11-SwoMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT1/TMTP-ViT1-SwoMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/Cyclegan-ViT10-SwoMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB1/Cyclegan-Swin5-SwoMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB/TMTP-Swin19-SwMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB1/TMTP1-Swin14-SwoMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB/HPF-Swin4-SwMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/HPF-ViT4-SwMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB1/Ori-Swin2-SwMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/TMTP_dot-ViT12-SwoMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/ViTB16_FAG-SwMP_mini.csv'
    # result_file = '/home/lyj/MFM-master/results/ViT/ViTB16_FAG-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB/SwinB_FAG-SwMP.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB/SwinB_FAG-DLC.csv'
    # result_file = '/home/lyj/MFM-master/results/SwinB/FMAG-Swin0-SwMP.csv'

    max_epoch = 20
    lr = 0.0001
    lr_decay = 0.5
    # lr = 0.0001
    # lr_decay = 0.1
    weight_decay = 1e-3
    # weight_decay = 5e-4
    # weight_decay = 0.05 # x

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # for k, v in kwargs.items():
        #     if not hasattr(self, k):
        #         warnings.warn("Warning: opt has not attribut %s" % k)
        #     setattr(self, k, v)
        #
        # opt.device = t.device('cuda:2') if opt.use_gpu else t.device('cpu')
        # print('user config:')
        # for k, v in self.__class__.__dict__.items():
        #     if not k.startswith('_'):
        #         print(k, getattr(self, k))
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
        # 修改设备选择逻辑以适应多卡
        if self.use_gpu:
            if self.use_multi_gpu:
                if t.cuda.device_count() > 1:
                    self.device = t.device('cuda')  # 使用 DataParallel 时，不指定具体的GPU
                else:
                    self.device = t.device('cuda:1')  # 如果只有一个GPU，默认使用 cuda:0
            else:
                self.device = t.device('cuda:1')  # 如果只有一个GPU，默认使用 cuda:0
        else:
            self.device = t.device('cpu')
        print('device',self.device)
        print('User config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
opt = DefaultConfig()

