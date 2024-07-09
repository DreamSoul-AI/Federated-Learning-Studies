from config import cfg


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['batch_size'] = int(cfg['control']['batch_size'])
    cfg['step_period'] = int(cfg['control']['step_period'])
    cfg['num_steps'] = int(cfg['control']['num_steps'])
    cfg['eval_period'] = int(cfg['control']['eval_period'])
    cfg['optimizer'] = cfg['control']['optimizer']
    cfg['optimizer']['optimizer_name'] = cfg['optimizer']['optimizer_name']
    cfg['optimizer']['lr'] = float(cfg['optimizer']['lr'])
    cfg['optimizer']['momentum'] = [float(x) for x in str(cfg['optimizer']['momentum']).split('-')]
    cfg['optimizer']['momentum'] = cfg['optimizer']['momentum'][0] \
        if len(cfg['optimizer']['momentum']) == 1 else cfg['optimizer']['momentum']
    cfg['optimizer']['scheduler_name'] = cfg['optimizer']['scheduler_name']
    # cfg['num_epochs'] = 400

    cfg['collate_mode'] = 'dict'

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                  'CIFAR100': [3, 32, 32]}
    target_size = {'MNIST': 10, 'FashionMNIST': 10, 'SVHN': 10, 'CIFAR10': 10, 'CIFAR100': 100}
    cfg['model']['data_shape'] = data_shape[cfg['data_name']]
    cfg['model']['target_size'] = target_size[cfg['data_name']]
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['model']['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = cfg['optimizer']['optimizer_name']
    cfg[tag]['optimizer']['lr'] = cfg['optimizer']['lr']
    cfg[tag]['optimizer']['momentum'] = cfg['optimizer']['momentum']
    # cfg[tag]['optimizer']['weight_decay'] = 5e-4
    cfg[tag]['optimizer']['weight_decay'] = 0
    cfg[tag]['optimizer']['nesterov'] = True if cfg[tag]['optimizer']['momentum'] != 0 else False
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = cfg['optimizer']['scheduler_name']

    if 'data_mode' in cfg['control']:
        cfg['data_mode'] = cfg['control']['data_mode']
        cfg['data_mode']['num_splits'] = int(cfg['data_mode']['num_splits'])

        cfg['dist_mode'] = cfg['control']['dist_mode']
        cfg['dist_mode']['active_ratio'] = float(cfg['dist_mode']['active_ratio'])
        cfg['dist_mode']['num_steps'] = int(cfg['dist_mode']['num_steps'])

        cfg[tag]['local'] = {}
        cfg[tag]['local']['device'] = cfg['device']
        cfg[tag]['local']['model'] = cfg['model']
        cfg[tag]['local']['num_steps'] = cfg['dist_mode']['num_steps']

        cfg[tag]['local']['optimizer'] = cfg['dist_mode']['local_optimizer']
        cfg[tag]['local']['optimizer']['lr'] = float(cfg[tag]['local']['optimizer']['lr'])
        cfg[tag]['local']['optimizer']['momentum'] = float(cfg[tag]['local']['optimizer']['momentum'])
        cfg[tag]['local']['optimizer']['weight_decay'] = 0
        cfg[tag]['local']['optimizer']['nesterov'] = True if cfg[tag]['local']['optimizer']['momentum'] != 0 else False
        cfg[tag]['local']['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
        cfg[tag]['local']['optimizer']['step_period'] = cfg['step_period']
        cfg[tag]['local']['optimizer']['num_steps'] = cfg['num_steps']

        cfg[tag]['fed'] = {}
        cfg[tag]['fed']['device'] = cfg['device']
        cfg[tag]['fed']['model'] = cfg['model']

        cfg[tag]['fed']['optimizer'] = cfg['dist_mode']['fed_optimizer']
        cfg[tag]['fed']['optimizer']['lr'] = float(cfg[tag]['fed']['optimizer']['lr'])
        cfg[tag]['fed']['optimizer']['momentum'] = float(cfg[tag]['fed']['optimizer']['momentum'])
        cfg[tag]['fed']['optimizer']['weight_decay'] = 0
        cfg[tag]['fed']['optimizer']['nesterov'] = True if cfg[tag]['fed']['optimizer']['momentum'] != 0 else False
        cfg[tag]['fed']['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
        cfg[tag]['fed']['optimizer']['step_period'] = cfg['step_period']
        cfg[tag]['fed']['optimizer']['num_steps'] = cfg['dist_mode']['num_steps']

        cfg[tag]['global'] = {}
        cfg[tag]['global']['device'] = cfg['device']
        cfg[tag]['global']['model'] = cfg['model']
        cfg[tag]['global']['optimizer'] = cfg[tag]['optimizer']
    return
