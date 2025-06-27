from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch 
from rl_games.algos_torch import model_builder

from rl_games.algos_torch.running_mean_std import RunningMeanStd # 1. 정규화를 위해 import 추가
from torch.nn import functional as F # 2. 손실 함수를 위해 import 추가

class A2CAgent(a2c_common.ContinuousA2CBase):

    def __init__(self, base_name, params):
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
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        
        # for payload
        self.has_payload_estimator = 'estimator_network' in self.params['config']


        if self.has_payload_estimator:
            # 설정 파일에서 추정기의 입력 크기를 가져옴
            estimator_obs_size = self.config['env_config_full']['env']['numEstimatorObservations']
            
            # 추정기 네트워크 빌드 설정
            estimator_config = {
                'actions_num': 1, 
                'input_shape': (estimator_obs_size,),
                'num_seqs': self.num_actors * self.num_agents,
                'normalize_value': False,
                # 정책망의 빌더를 그대로 사용하므로, network_builder에서 normalize_input을 처리하도록 전달
                'normalize_input': False, 
            }
            
            # 추정기 모델 생성
            self.payload_estimator_model = self.network.build(estimator_config)
            self.payload_estimator_model.to(self.ppo_device)
            
            # 추정기 옵티마이저 생성
            estimator_lr = self.config.get('estimator_lr', 1e-4) 
            self.payload_estimator_optimizer = optim.Adam(self.payload_estimator_model.parameters(), float(estimator_lr))

            # --- [요청 1] 추정기 입력 정규화 로직 추가 ---
            self.normalize_estimator_input = self.config.get('normalize_estimator_input', False)
            if self.normalize_estimator_input:
                self.estimator_obs_rms = RunningMeanStd(shape=(estimator_obs_size,)).to(self.ppo_device)

    

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

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
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

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
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

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)

    
    # for payload
    def train_actor_critic(self, input_dict):
        # 1. 정책(Policy) 신경망 학습
        self.calc_gradients(input_dict)
        
        # 2. 페이로드 추정기(Estimator) 신경망 학습
        if self.has_payload_estimator:
            # 추정기 입력을 가져옴
            estimator_obs = input_dict['e_obs']
            true_payload = input_dict['e_true_payload']

            # --- [요청 1] 정규화 적용 ---
            if self.normalize_estimator_input:
                self.estimator_obs_rms.update(estimator_obs)
                estimator_obs = self.estimator_obs_rms.normalize(estimator_obs)
            
            # Autocast와 함께 추정기 순전파 및 손실 계산
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                predicted_payload = self.payload_estimator_model({'obs': estimator_obs})['mus']
                estimator_loss = F.mse_loss(predicted_payload, true_payload)

            # 추정기 역전파 및 가중치 업데이트
            self.payload_estimator_optimizer.zero_grad()
            self.scaler.scale(estimator_loss).backward()
            self.scaler.step(self.payload_estimator_optimizer)
            
            # self.scaler.update()는 calc_gradients 내부에서 한 번만 호출됨
            
            # TensorBoard에 손실 기록
            self.writer.add_scalar('losses/estimator_loss', estimator_loss, self.epoch_num, self.frame)
        
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


