from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch 

import torch.nn as nn     # for payload
from rl_games.algos_torch.network_builder import PayloadEstimator   # for payload


class A2CAgent(a2c_common.ContinuousA2CBase):

    def __init__(self, base_name, params):
        print("--- a2c_continuous.py: 받은 params의 모든 키 ---")
        print(params.keys())
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        # for payload
        # 정책망의 입력 크기는 (환경 관측값 크기 + 추정된 페이로드 1개)가 됩니다.
        policy_obs_shape = (self.obs_shape[0] + 1,)

        policy_build_config = {
            'actions_num' : self.actions_num, 'input_shape' : policy_obs_shape,
            'num_seqs' : self.num_actors * self.num_agents, 'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value, 'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)

        # for payload
        # <<< 2. 페이로드 추정 신경망(self.estimator_model) 생성 >>>
        # network_builder에 새로 추가한 PayloadEstimator를 사용합니다.
        # YAML 파일에서 'estimator_network' 설정을 가져옵니다.
        estimator_params = self.config['estimator_network']
        estimator_build_config = {
            'input_shape': self.obs_shape,
        }

        self.estimator_model = PayloadEstimator(input_size=self.obs_shape[0], units=[128, 64])
        self.estimator_model.to(self.ppo_device)

        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'

        # for payload 
        # <<< 3. 두 개의 독립된 옵티마이저(Optimizer) 생성 >>>
        # 정책망을 위한 옵티마이저
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        # 추정망을 위한 옵티마이저
        self.estimator_optimizer = optim.Adam(self.estimator_model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

    # for payload
    # a2c_common에도 있어 그런데 덮어쓸려고 다시 만듬 
    # <<< 4. 부모의 기능을 덮어쓰는 새로운 get_action_values 메소드를 추가/재정의합니다. >>>
    def get_action_values(self, obs):
        self.model.eval()
        self.estimator_model.eval()
        processed_obs = self._preproc_obs(obs['obs'])

        with torch.no_grad():
            # 1단계: 추정기로 페이로드 추정
            estimated_payload = self.estimator_model(processed_obs)
            # 2단계: 정책망 입력 생성 (obs + 추정 페이로드)
            policy_input = torch.cat([processed_obs, estimated_payload], dim=-1)
            
            input_dict = {
                'is_train': False, 'prev_actions': None, 
                'obs' : policy_input, 'rnn_states' : self.rnn_states
            }
            # 3단계: 정책망으로 최종 행동 계산
            res_dict = self.model(input_dict)
            
            # a2c_common.py의 피드백 루프를 위해 추정값을 결과에 포함
            res_dict['estimates'] = estimated_payload
        
        if self.has_central_value:
            states = obs['states']
            input_dict = {
                'is_train': False,
                'states' : states,
            }
            value = self.get_central_value(input_dict)
            res_dict['values'] = value
        return res_dict
    
    # for payload
    # a2c_common에도 있어 그런데 덮어쓸려고 다시 만듬 
    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state['estimator_model'] = self.estimator_model.state_dict()
        state['estimator_optimizer'] = self.estimator_optimizer.state_dict()
        return state

    # for payloadd
    def set_full_state_weights(self, weights, set_epoch=True):
        super().set_full_state_weights(weights, set_epoch)
        if 'estimator_model' in weights:
            self.estimator_model.load_state_dict(weights['estimator_model'])
        if 'estimator_optimizer' in weights:
            self.estimator_optimizer.load_state_dict(weights['estimator_optimizer'])


    
    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        # for payload
        # --- 학습 단계 1: 페이로드 추정기 업데이트 ---
        self.estimator_optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            estimated_payload = self.estimator_model(obs_batch)
            payload_loss = nn.functional.mse_loss(estimated_payload, ground_truth_payload)
        
        self.scaler.scale(payload_loss * self.config['estimator_coef']).backward()
        self.scaler.step(self.estimator_optimizer)
        self.scaler.update()

        # --- 학습 단계 2: 정책 신경망 업데이트 ---
        self.optimizer.zero_grad()
        
        # 정책망 입력을 만듭니다: obs + 추정된 페이로드 (역전파 방지)
        policy_input = torch.cat([obs_batch, estimated_payload.detach()], dim=-1)

        batch_dict = {
            'is_train': True, 'prev_actions': input_dict['actions'], 'obs' : policy_input
        }



        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            policy_loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(policy_loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask


        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, kl_dist, self.last_lr, 1.0, 
                             mu.detach(), sigma.detach(), b_loss, payload_loss.detach())


    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

# for payload
# <<< 1. 추정기 신경망 클래스를 이 파일 안에 직접 정의합니다. >>>
class PayloadEstimator(nn.Module):
    def __init__(self, input_size, units=[128, 64], activation='relu'):
        super().__init__()
        
        activation_func = nn.ReLU if activation == 'relu' else nn.ELU
        
        layers = []
        in_size = input_size
        for unit in units:
            layers.append(nn.Linear(in_size, unit))
            layers.append(activation_func())
            in_size = unit
            
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_size, 1)

    def forward(self, x):
        return self.output_layer(self.mlp(x))