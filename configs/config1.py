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
    路径加载
    '''
    train_data_root_cnn_1 = '/home/data1/lyj/CMA/DM_FAG/images/' # train data path

    test_data_root = '/home/data1/lyj/CMA/SwoMP_patch/images/' # test data path

    load_model_path = 'save model checkpoint path'

    batch_size_0 = 128
    batch_size_1 = 64

    use_gpu = True
    use_multi_gpu = False

    num_workers = 0
    print_freq = 20
    debug_file = ''
    ##test
    result_file = '/home/lyj/MFM-master/results/ViT/FAG-ViT-SwoMP_patch.csv' # save test results path

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

