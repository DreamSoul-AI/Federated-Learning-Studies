import datetime
import numpy as np
import time
import torch
from config import cfg
from dataset import make_data_loader, split_dataset
from model import make_model, make_optimizer, make_scheduler, make_batchnorm

from module import to_device

class Server:
    def __init__(self, id, dataset, data_split, model, optimizer, scheduler, logger, cfg):
        self.id = id
        self.dataset = dataset
        self.data_split = data_split
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.cfg = cfg
        self.data_loader = make_data_loader(self.dataset, self.cfg['optimizer']['batch_size'], shuffle=False)

    # def synchronize(self, active_client_id, model_state_dict, optimizer_state_dict, scheduler_state_dict):
    #     with torch.no_grad():
    #         if len(model_state_dict) > 0:
    #             self.optimizer['global'].zero_grad()
    #             valid_data_size = [len(self.data_split['data'][active_client_id[i]])
    #                               for i in range(len(active_client_id))]
    #             weight = torch.tensor(valid_data_size)
    #             weight = weight / weight.sum()
    #             for k, v in self.model.named_parameters():
    #                 parameter_type = k.split('.')[-1]
    #                 if 'weight' in parameter_type or 'bias' in parameter_type:
    #                     tmp_v = v.data.new_zeros(v.size())
    #                     for i in range(len(active_client_id)):
    #                         tmp_v += weight[i] * model_state_dict[i][k]
    #                     v.grad = (v.data - tmp_v).detach()
    #             self.optimizer['global'].step()
    #             self.scheduler['global'].step()
    #             self.optimizer['local'].load_state_dict(optimizer_state_dict[0])
    #             self.scheduler['local'].load_state_dict(scheduler_state_dict[0])
    #     return

    def synchronize(self, active_client_id, model_state_dict, optimizer_state_dict, scheduler_state_dict):
    
        initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
        # fed level
        with torch.no_grad():
            if len(model_state_dict) > 0:
    
                valid_data_size = [len(self.data_split['data'][active_client_id[i]])
                                    for i in range(len(active_client_id))]
                weight = torch.tensor(valid_data_size)
                weight = weight / weight.sum()
                for i in range(len(active_client_id)):

                    self.optimizer['fed'].zero_grad()
                    for k, v in self.model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            diff = v.data - model_state_dict[i][k]
    
                            v.grad = diff.detach()
                    self.optimizer['fed'].step()
                    self.scheduler['fed'].step()
    
            # global level
            diffs = {}
            for k, v in self.model.named_parameters():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    diff = initial_model_state[k] - v.data
                    diffs[k] = diff.clone().detach()
    
            self.model.load_state_dict(initial_model_state)
    
            self.optimizer['global'].zero_grad()
            for k, v in self.model.named_parameters():
                if k in diffs:
                    v.grad = diffs[k].detach()
            self.optimizer['global'].step()
            self.scheduler['global'].step()
    
            # local level
            # merged_state = {}
            # possible_state_keys = ['momentum_buffer', 'exp_avg', 'exp_avg_sq']
            #
            # local_optimizer_state_dict = self.optimizer['local'].state_dict()
            #
            # for i in range(len(active_client_id)):
            #     state_dict = optimizer_state_dict[i]
            #     for param_idx, param_state in state_dict['state'].items():
            #         if param_idx not in merged_state:
            #             merged_state[param_idx] = {}
            #
            #         for state_name in possible_state_keys + ['step']:
            #             if state_name in param_state:
            #                 state_value = param_state[state_name]
            #                 if state_name not in merged_state[param_idx]:
            #
            #                     if state_name == 'step':
            #                         merged_state[param_idx][state_name] = local_optimizer_state_dict['state'].get(
            #                             param_idx, {}).get('step', torch.tensor(0.0))
            #                     else:
            #                         if state_value is not None:
            #                             merged_state[param_idx][state_name] = state_value.new_zeros(
            #                                 size=state_value.size())
            #                 # step max or not
            #                 # if state_name == 'step':
            #                 #     merged_state[param_idx][state_name] = max(merged_state[param_idx][state_name],
            #                 #                                               state_value)
            #                 if state_name != 'step' and state_value is not None:
            #                     merged_state[param_idx][state_name] += state_value * weight[i]
            #
            # optimizer_state_to_load = {'state': merged_state,
            #                            'param_groups': local_optimizer_state_dict['param_groups']}
            # self.optimizer['local'].load_state_dict(optimizer_state_to_load)
    
            self.optimizer['local'].load_state_dict(optimizer_state_dict[0])
    
            self.scheduler['local'].load_state_dict(scheduler_state_dict[0])
    
        return

    def train(self, client):
        start_time = time.time()
        num_active_clients = int(np.ceil(cfg['dist_mode']['active_ratio'] * len(client)))


        clients_per_round = num_active_clients
        total_clients = len(client)


        num_rounds = total_clients // clients_per_round


        if 'current_round' not in cfg:
            cfg['current_round'] = 0


        start_idx = cfg['current_round'] * clients_per_round
        end_idx = start_idx + clients_per_round


        if end_idx > total_clients:
            end_idx = total_clients
            start_idx = total_clients - clients_per_round


        active_client_id = list(range(start_idx, end_idx))
        active_client = [client[i] for i in active_client_id]


        cfg['current_round'] = (cfg['current_round'] + 1) % num_rounds


        # active_client_id = torch.randperm(len(client))[:num_active_clients]
        # active_client = [client[i] for i in range(len(client)) if i in active_client_id]
        result = []

        # # Debug
        # print(f"Total clients: {total_clients}")
        # print(f"Clients per round: {clients_per_round}")
        # print(f"Current round: {cfg['current_round']}")
        # print(f"Selected active client IDs: {active_client_id}")
        # print(f"Selected active clients: {active_client}")


        for i in range(len(active_client)):
            result_i = active_client[i].train(self.model.state_dict(),
                                              self.optimizer['local'].state_dict(),
                                              self.scheduler['local'].state_dict())
            result.append(result_i)

        model_state_dict = [result[i]['model_state_dict'] for i in range(len(result))]
        optimizer_state_dict = [result[i]['optimizer_state_dict'] for i in range(len(result))]
        scheduler_state_dict = [result[i]['scheduler_state_dict'] for i in range(len(result))]
        logger_state_dict = [result[i]['logger_state_dict'] for i in range(len(result))]
        lr = optimizer_state_dict[0]['param_groups'][0]['lr']
        for i in range(len(logger_state_dict)):
            self.logger.update_state_dict(logger_state_dict[i])
        step_time = (time.time() - start_time)
        exp_finished_time = datetime.timedelta(
            seconds=round((cfg['num_steps'] - (cfg['step'] + 1)) * step_time))
        info = {'info': ['Model: {}'.format(cfg['tag']),
                         'Train Epoch (C): {}'.format(cfg['step'] + 1),
                         'Learning rate: {:.6f}'.format(lr),
                         'Experiment Finished Time: {}'.format(exp_finished_time)]}
        self.logger.append(info, 'train')
        print(self.logger.write('train'))


        cfg['step'] += cfg[cfg['tag']]['local']['num_steps']

        return active_client_id, model_state_dict, optimizer_state_dict, scheduler_state_dict

    def make_batchnorm_server(self, momentum, track_running_stats):
        with torch.no_grad():
            flag = make_batchnorm(self.model, momentum, track_running_stats=track_running_stats)
            if flag and track_running_stats:
                self.model.train(True)
                for i, input in enumerate(self.data_loader['train']):
                    self.model(input)
        return

    def test_server(self):
        with torch.no_grad():
            self.model.train(False)
            for i, input in enumerate(self.data_loader['test']):
                input = to_device(input, self.cfg['device'])
                input_size = input['data'].size(0)
                self.model.to(cfg['device'])
                output = self.model(input)
                evaluation = self.logger.evaluate('test', 'batch', input, output)
                self.logger.append(evaluation, 'test', input_size)
            evaluation = self.logger.evaluate('test', 'full')
            self.logger.append(evaluation, 'test', input_size)
            info = {'info': ['Model: {}'.format(cfg['tag']),
                             'Test Epoch (S): {}'.format(cfg['step'])]}
            self.logger.append(info, 'test')
            print(self.logger.write('test'))

        return

    def make_batchnorm_client(self, client, momentum, track_running_stats):
        num_active_clients = int(np.ceil(cfg['dist_mode']['active_ratio'] * len(client)))
        result = []
        for i in range(0, len(client), num_active_clients):
            result_i = []
            upper_bound = min(i + num_active_clients, len(client))
            for j in range(i, upper_bound):
                result_i_j = client[i + j].make_batchnorm.remote(self.model.state_dict(), momentum, track_running_stats)
                result_i.append(result_i_j)
            result.extend(result_i)
        model_state_dict = [result[i]['model_state_dict'] for i in range(len(result)) if result[i] is not None]

        with torch.no_grad():
            if len(model_state_dict) > 0:
                data_size = [len(self.data_split['data'][i]) for i in range(len(client))]
                model_state_dict_ = self.model.state_dict()
                for k, v in model_state_dict_.items():
                    parameter_type = k.split('.')[-1]
                    if 'running_mean' in parameter_type:
                        global_running_mean = v.data.new_zeros(v.size())
                        for i in range(len(model_state_dict)):
                            global_running_mean += data_size[i] * model_state_dict[i][k]
                        global_running_mean = global_running_mean / sum(data_size)
                        v.data.copy_(global_running_mean)
                    if 'running_var' in parameter_type:
                        running_mean_k = k.replace('running_var', 'running_mean')
                        global_running_var = v.data.new_zeros(v.size())
                        for i in range(len(model_state_dict)):
                            global_running_var += model_state_dict[i][k] * (data_size[i] - 1) + data_size[i] * (
                                    model_state_dict[i][running_mean_k] - global_running_mean) ** 2
                        global_running_var = global_running_var / (sum(data_size) - len(data_size))
                        v.data.copy_(global_running_var)
                self.model.load_state_dict(model_state_dict_)
        return

    def test_client(self, client):
        num_active_clients = int(np.ceil(cfg['dist_mode']['active_ratio'] * len(client)))
        result = []
        for i in range(0, len(client), num_active_clients):
            result_i = []
            upper_bound = min(i + num_active_clients, len(client))
            for j in range(i, upper_bound):
                result_i_j = client[i + j].test.remote(self.model.state_dict())
                result_i.append(result_i_j)
            result.extend(result_i)
        logger_state_dict = [result[i]['logger_state_dict'] for i in range(len(result))]
        for i in range(len(logger_state_dict)):
            self.logger.update_state_dict(logger_state_dict[i])
        info = {'info': ['Model: {}'.format(cfg['tag']),
                         'Test Epoch (C): {}'.format(cfg['step'])]}
        self.logger.append(info, 'test')
        print(self.logger.write('test'))
        return
