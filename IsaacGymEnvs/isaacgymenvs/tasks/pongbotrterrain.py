# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os, time
import json

from isaacgym import gymtorch
from isaacgym import gymapi
from .base.vec_task import VecTask

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Joy


import torch
from typing import Tuple, Dict
import random
from isaacgymenvs.utils.torch_jit_utils import quat_from_euler_xyz, quat_rotate,my_quat_rotate,quat_conjugate, to_torch, get_axis_params, torch_rand_float, normalize, quat_apply, quat_rotate_inverse,get_euler_xyz,matrix_to_quaternion,quat_mul
from isaacgymenvs.tasks.base.vec_task import VecTask


class PongBotRTerrain(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.height_samples = None
        self.custom_origins = False
        # self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.init_done = False
        # 2_stand + 4_stand + 2_stand + 4_stand
        self.FR_RL_swing_time = 0.2
        self.FL_RR_swing_time = 0.2
        self.nonswing_time = 0.05   # 네발 지지구간
        self.freeze_steps = 50      # freeze 명령 시 지속 스텝 수
        self.min_swing_time = 0.1   # 최소 스윙 시간
        # self.cycle_time = self.FR_RL_swing_time + self.nonswing_time + self.FL_RR_swing_time + self.nonswing_time
        self.cycle_time = 0.5       # 전체 보행 사이클 시간
        self.ref_torques = 0.       # 기준 토크 초기값
        self.plot_cnt = 1           # 시각화 카운트
        self.velocity_threshold = 0.1 # 속도 임계값 (단위/초)

        
        self.ema_alpha_foot = 0.5 # 각 발 속도 방향 EMA 평활화 계수 (0 < alpha < 1), 각 발의 속도 방향 필터

        # 카메라 움직임 필터
        self.smoothed_cam_pos = None
        self.smoothed_cam_target = None
        self.smoothing_alpha = 0.1 # EMA 필터 계수 

        self.test_mode = self.cfg["env"]["test"]
        
        # 테스트 모드 여부 확인
        if self.test_mode:
            self.observe_envs = 0
            self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
            self.cam_change_flag = True
            self.cam_change_cnt = 0
            print('\033[103m' + '\033[33m' + '_________________________________________________________' + '\033[0m')
            print('\033[103m' + '\033[33m' + '________________________' + '\033[30m' + 'Test Mode' + '\033[33m' + '\033[103m'+ '________________________' + '\033[0m')
            print('\033[103m' + '\033[33m' +'_________________________________________________________' + '\033[0m')
        else:
            self.observe_envs = 101    # 학습 시 시뮬레이션 환경 수?  그러면 학습할 때 환경 수가 601개 라는건가? ex? 저글러 보는 환경 번호101,201~~

        rospy.init_node('plot_juggler_node')

        # obs_scales
        self.obs_scales = {}
        for key, value in self.cfg["env"]["learn"]["obs_scales"].items():  # rewards 섹션 순회
            self.obs_scales[key] = float(value) 

        # rew_scales
        self.rew_scales = {}
        self.reward_container ={}
        for key, value in self.cfg["env"]["learn"]["reward"].items():  # rewards 섹션 순회
            self.rew_scales[key] = float(value) 
        # print("rew_scales : ", self.rew_scales)


        # action_scale
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.soft_dof_pos_limit = self.cfg["env"]["learn"]["soft_dof_pos_limit"]

        # randomization
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        #command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # envs ranges
        self.stand_env_range = [0, self.cfg["env"]["EnvsNumRanges"]["stand_env_range"] - 1]
        self.only_plus_x_envs_range = [self.stand_env_range[1] + 1, self.stand_env_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_plus_x_envs_range"]]
        self.only_minus_x_envs_range = [self.only_plus_x_envs_range[1] + 1, self.only_plus_x_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_minus_x_envs_range"]]
        self.only_plus_y_envs_range = [self.only_minus_x_envs_range[1] + 1, self.only_minus_x_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_plus_y_envs_range"]]
        self.only_minus_y_envs_range = [self.only_plus_y_envs_range[1] + 1, self.only_plus_y_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_minus_y_envs_range"]]
        self.only_plus_yaw_envs_range = [self.only_minus_y_envs_range[1] + 1, self.only_minus_y_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_plus_yaw_envs_range"]]
        self.only_minus_yaw_envs_range = [self.only_plus_yaw_envs_range[1] + 1, self.only_plus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_minus_yaw_envs_range"]]
        self.plus_x_plus_yaw_envs_range = [self.only_minus_yaw_envs_range[1] + 1, self.only_minus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["plus_x_plus_yaw_envs_range"]]
        self.plus_x_minus_yaw_envs_range = [self.plus_x_plus_yaw_envs_range[1] + 1, self.plus_x_plus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["plus_x_minus_yaw_envs_range"]]
        self.minus_x_plus_yaw_envs_range = [self.plus_x_minus_yaw_envs_range[1] + 1, self.plus_x_minus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["minus_x_plus_yaw_envs_range"]]
        self.minus_x_minus_yaw_envs_range = [self.minus_x_plus_yaw_envs_range[1] + 1, self.minus_x_plus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["minus_x_minus_yaw_envs_range"]]


        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = (self.decimation+1) * self.cfg["sim"]["dt"]


        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.change_interval = int(self.cfg["env"]["learn"]["cmdInterval_s"] / self.dt + 0.5)
        self.zero_interval = int(self.cfg["env"]["learn"]["zerocmdInterval_s"] / self.dt + 0.5)
        self.freeze_interval = int(self.cfg["env"]["learn"]["freezecmdInterval_s"] / self.dt + 0.5)
        # self.foot_freeze_interval = int(self.cfg["env"]["learn"]["footfreezecmdInterval_s"] / self.dt + 0.5)

        # self.allow_calf_contacts = self.cfg["env"]["learn"]["allowcalfContacts"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]

        # 보상 스케일에 시간 간격을 곱해서 시간 정규화
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        # 시각화용 카메라 설정
        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # 텐서 최신화 (현재 시뮬 상태 반영)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # 각 텐서를 PyTorch 텐서로 래핑하여 조작 가능하게 함
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis


        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor)
        self.rigid_body_pos   = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,:3]
        self.rigid_body_rot   = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,3:7]
        self.rigid_body_vel   = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,7:10]


        # initialize some data used later on
        # 공통 스텝 카운터, 제어 상태 변수 초기화
        self.common_step_counter = 0
        self.zero_cmd_flag = 0
        self.freeze_cnt = 0
        self.freeze_flag = False

        self.extras = {} # 추가정보 저장용
        
        # 명령 텐서: x y 속도, yaw 회전 속도
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_x          = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_y          = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_yaw        = self.commands.view(self.num_envs, 3)[..., 2] 

        # 중력 및 정면 벡터(보상 계산 등에 사용)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        
        # 사이클 기반 보행 패턴에 사용
        self.sin_cycle = torch.zeros(self.num_envs,4, dtype=torch.float, device=self.device, requires_grad=False)
        self.cos_cycle = torch.zeros(self.num_envs,4, dtype=torch.float, device=self.device, requires_grad=False)

        # 토크/액션 관련 텐서  ??action2가 있지? 과거 시점의 액션
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
       
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions2 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 스윙 페이즈 플래그
        self.FL_RR_swing_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.FR_RL_swing_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 이전 속도 방향 각도 저장
        self.prev_velocity_angles = torch.zeros(self.num_envs, device=self.device) 

        # 리셋 및 카메라 상태
        self.need_reset = False
        self.cam_mode = 0
        
        # 관절 복구 상태, 명령 백업, 관절 속도 저장용 텐서
        self.dof_recovery = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.original_commands = torch.zeros_like(self.commands)
        # self.last_dof_pos = torch.zeros_like(self.dof_pos)
        # self.last_dof_pos2 = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_vel2 = torch.zeros_like(self.dof_vel)

        # 관절 가속도 계산용
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.last_dof_acc = torch.zeros_like(self.dof_vel)
        
        # 명령 없음 상태 표시
        self.no_commands = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 관절 위치 제한값    다시 설정해줘야하나???????  urdf는 널널하게 학습단에서는 깐깐하게[rad]
        self.roll_limits = torch.tensor([-0.349066, 0.349066], device=self.device)
        self.hp_limits = torch.tensor([-0.366519, 1.5708], device=self.device)
        self.knp_limits = torch.tensor([-2.58309, -0.488692], device=self.device)

        # 제어 오류 목표 관련 텐서    calc가 뭐지? 각도에러? target은 레퍼런스?
        # self.calc_dof_vel = torch.zeros_like(self.dof_vel)
        self.dof_pos_error = torch.zeros_like(self.dof_pos)
        self.targets = torch.zeros_like(self.dof_pos)
        
        # 레퍼런스 발높이
        self.ref_foot_height = torch.zeros(self.num_envs,4, device=self.device, dtype=torch.float)
        

        self.last_base_pos = torch.zeros(self.num_envs,3, device=self.device, dtype=torch.float)
        self.base_pos = torch.zeros(self.num_envs,3, device=self.device, dtype=torch.float)

        self.base_vel = torch.zeros(self.num_envs,6, device=self.device, dtype=torch.float)
        self.last_base_vel = torch.zeros(self.num_envs,6, device=self.device, dtype=torch.float)

        # 마지막 접촉 여부, 발 위치, 사이클 시간 정보
        self.last_contacts = torch.zeros(self.num_envs,4, device=self.device, dtype=torch.bool)
        self.foot_pos = torch.zeros(self.num_envs,4, device=self.device, dtype=torch.float)
        self.cycle_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.last_cycle_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # 초기절 위치 설정용 텐서
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 특수 조건 추적용 플래그 ?????? 플리즈에 넣기 위한건데 안쓸수도
        self.zero = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.zero_cnt = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.fall_cnt = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stop_cnt = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # 각 발의 EMA 속도 방향 각도  ema가 무슨 뜻???? 발끝의 기준으로 높이 지형 환경 알라고 사용하는 변수 예 부채꼴
        self.ema_foot_velocity_angles = torch.zeros(self.num_envs, 4, device=self.device) 
        
        # 발 스윙 시작 시간 및 상태 저장
        self.foot_swing_start_time = torch.zeros(self.num_envs, 4, dtype=torch.float,device=self.device)
        self.foot_swing_state = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)  # 각 발의 스윙 상태
        
        # 사이클 위상
        self.phi = torch.zeros(self.num_envs, 4, dtype=torch.float,device=self.device)
        
        # 명령 관련 임시 저장
        self.original_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.temp_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.command_change_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # 각 환경의 명령 변경 시간
        self.command_duration = 5.0 # 변경된 명령을 유지할 시간 (초)

        # 다양한 조건 감시용 플래그들
        self.body_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dof_limit_lower = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dof_limit_upper = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dof_vel_over = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 지형 노말 벡터(보상 계산 등에서 사용)
        self._normal = torch.zeros(self.num_envs,3, device=self.device, dtype=torch.float)
        
        # 지형 높이 샘플링 지점 초기화
        self.height_points = self.init_height_points()
        self.leg_height_points = self.init_leg_height_points()
        # self.leg_height_points = self.init_leg_height_points_sector()

        # 측정된 높이 데이터 초기화
        self.measured_heights = None
        self.measured_legs_heights = None

        # 각 관절의 기본 위치 설정???????
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(), "ang_vel_xy": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(), "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros(), "action_rate": torch_zeros(), "hip": torch_zeros()}


        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True

    # 시뮬레이션 환경을 생성하는 함수
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        terrain_type = self.cfg["env"]["terrain"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True
        # 전체 환경 생성(num_envs 개의 환경, spacing 거리, 가로*세로 개수 설정)
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        # 랜덤화가 활성화 되면 초록, 활성화가 안되면 빨강
        if self.randomize:
            print('\033[102m' + '\033[32m' + '_________________________________________________________' + '\033[0m')
            print('\033[102m' + '\033[32m' + '______________________' + '\033[30m' + 'Randomize True' + '\033[32m' + '\033[102m'+ '______________________' + '\033[0m')
            print('\033[102m' + '\033[32m' +'_________________________________________________________' + '\033[0m')
            self.apply_randomizations(self.randomization_params)
        else:
            print('\033[101m' + '\033[31m' + '_________________________________________________________' + '\033[0m')
            print('\033[101m' + '\033[31m' + '______________________' + '\033[30m' + 'Randomize False' + '\033[31m' + '\033[101m'+ '______________________' + '\033[0m')
            print('\033[101m' + '\033[31m' +'_________________________________________________________' + '\033[0m')

    # obs에 적용할 노이즈 크기 벡터
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg["env"]["learn"]["addNoise"]
        noise_level = self.cfg["env"]["learn"]["noiseLevel"]
        noise_vec[0:3] = self.cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.obs_scales["angularVelocityScale"]
        noise_vec[3:6] = self.cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[6:9] = 0.
        noise_vec[9:21] = self.cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.obs_scales["dofPositionScale"]
        noise_vec[21:33] = self.cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.obs_scales["dofVelocityScale"]
        noise_vec[33:53] = 0.
        return noise_vec

    # 시뮬레이션 환경에 평평한 지형 생성함수
    def _create_ground_plane(self):
        # gym api에서 지면 생성 함수 
        plane_params = gymapi.PlaneParams()

        # 지면의 법선 벡터 z축이 위로 향하는 평면, 즉 xy평면에 위치
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        
        # 마찰
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        
        # 반발 계수, 0: 튀지 않음, 1: 완전탄성충돌
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        # 설정한 파라미터를 사용한 지면을 시뮬레이션에 추가
        self.gym.add_ground(self.sim, plane_params)
    
    # 삼각형 메쉬 지형을 생성하는 함수, plane대신 불규칙하고 복잡한 지형을 만들 때 사용
    def _create_trimesh(self):
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        
        # gym api 에서 삼각형메시 설정 객체
        tm_params = gymapi.TriangleMeshParams()

        # 지형에서 생성된 정점(vertex), 삼각형 개수 설정???
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        
        # 지형을 월드 좌표로 옮기기 위한 변환 설정
        tm_params.transform.p.x = -self.terrain.border_size 
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0

        # 마찰, 반발계수
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        # 생성된 메시 시뮬레이터에 등록
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        
        # 높이 정보를 텐서로 변환하여 저장,, 로봇의 위치 판단, 충돌 검사, 발 위치 보정 등에 사용
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    # 시뮬레이터에서 로봇이 존재할 개별 환경들을 생성하고 초괴화하는 함수, n개의 강화학습 환경을 만들어서 각 환경에 로봇(actor)을 배치
    def _create_envs(self, num_envs, spacing, num_per_row):    #총 환경 개수, 각 환경간의 거리, 환경을 가로로 몇 개 배칠할지
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        if self.test_mode:
            asset_file = self.cfg["env"]["urdfAsset"]["test_file"]
        
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # asset 로딩 옵션 설정
        asset_options = gymapi.AssetOptions()
        
        # 힘 제어 모드
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        
        # base고정 여부, 중력 사용 등 옵션 설정
        asset_options.collapse_fixed_joints = False
        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.use_mesh_materials = False
        # asset_options.density = 0.001
        # asset_options.angular_damping = 0.01
        # asset_options.linear_damping = 0.01
        # asset_options.armature = 0.0
        # asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        # 로봇 모델을 시뮬레이터에 로딩   
        pongbotr_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(pongbotr_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(pongbotr_asset)
        
        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(pongbotr_asset)
        # for element in rigid_shape_prop:
        #     element.friction = 0.01 
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        print('\033[93m' + "friction_range : " + '\033[0m', friction_range)
        num_buckets = 500
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)

        # 로봇의 초기 위치 설정
        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])


        self.body_names = self.gym.get_asset_rigid_body_names(pongbotr_asset)
        self.dof_names = self.gym.get_asset_dof_names(pongbotr_asset)

        foot_name = self.cfg["env"]["urdfAsset"]["footName"]
        calf_name = self.cfg["env"]["urdfAsset"]["calfName"]
        feet_names = [s for s in self.body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device
                                        =self.device, requires_grad=False)
        calf_names = [s for s in self.body_names if calf_name in s]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        # 관절의 속성 설정
        dof_props = self.gym.get_asset_dof_properties(pongbotr_asset)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,requires_grad=False)
        self.reset_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,requires_grad=False)

        # 관절의 리밋, 속성 등을 설정      ?????? 시뮬레이션에서 관절 세팅 값, 실제랑 같은 모습이 나오게 세팅해야함 위치제어할거면 stiffness,damping을 변경, 토크제어를 할거면 stiffness=0으로 
        for i in range(len(dof_props)):
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            self.reset_dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0]
            self.reset_dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1]

            dof_props['stiffness'][i] = 0.
            dof_props['damping'][i] = 0.5
            dof_props['friction'][i] = 0.28
            dof_props['armature'][i] = 0.05 # 회전관성
        # print("dof_props : " , dof_props)
        dof_property_names = [
            "hasLimits", 
            "lower", 
            "upper", 
            "driveMode", 
            "velocity", 
            "effort", 
            "stiffness", 
            "damping", 
            "friction", 
            "armature"
        ]

        for i, dof_prop in enumerate(dof_props):
            print(f"관절 {i+1}:")
            for j, value in enumerate(dof_prop):
                print(f"  {dof_property_names[j]}: {value}") 



        # env origins, 환경의 시작 위치, 지형 설정
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        
        # 복잡 지형(trimesh)일 경우 사용자 지정 origin사용
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.pongbotr_handles = []
        self.envs = []

        # 환경 생성
        for i in range(self.num_envs):
            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            
            # 지형 기반 origin 위치 설정
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)



            self.gym.set_asset_rigid_shape_properties(pongbotr_asset, rigid_shape_prop)

            # 로봇(actor) 생성 및 환경에 배치
            pongbotr_handle = self.gym.create_actor(env_handle, pongbotr_asset, start_pose, "pongbotr", i, 0, 0)
            

            # 관절 속성 적용
            self.gym.set_actor_dof_properties(env_handle, pongbotr_handle, dof_props)
            
            # 핸들들을 리스트에 저장
            self.envs.append(env_handle)
            self.pongbotr_handles.append(pongbotr_handle)
        
        if self.test_mode :
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_handle, pongbotr_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.420, 0.420, 0.420))
            
            for i in range(1, 5):  # i가 1부터 시작하도록 수정
                i *= 4
                self.gym.set_rigid_body_color(
                    env_handle, pongbotr_handle, i, gymapi.MESH_VISUAL, gymapi.Vec3(0.996, 0.890, 0.420))



        # num_bodies = self.gym.get_actor_rigid_body_count(env_handle, pongbotr_handle)

        # for i in [1, 2, 5, 6, 9, 10, 13, 14]:
        #     self.gym.set_rigid_body_color(env_handle, pongbotr_handle, i, gymapi.MESH_VISUAL, gymapi.Vec3(0.803, 0.633, 0.000))
        # for i in [0, 3, 7, 11, 15]:
        #     self.gym.set_rigid_body_color(env_handle, pongbotr_handle, i, gymapi.MESH_VISUAL, gymapi.Vec3(0.154, 0.154, 0.154))
        # for i in [4, 8, 12, 16]:
        #     self.gym.set_rigid_body_color(env_handle, pongbotr_handle, i, gymapi.MESH_VISUAL, gymapi.Vec3(0.676, 0.633, 0.449))
        
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.pongbotr_handles[0], feet_names[i])
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.pongbotr_handles[0], calf_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.pongbotr_handles[0], "base")
        

    # 시뮬레이션에서 환경에 대해 에피소드를 리셋(종료)해야 하는지 판단하는 함수
    def check_termination(self):
        environment_index = 205  # 인덱스는 0부터 시작하므로 206번째 환경은 인덱스 205입니다.
        reset_reasons = [False] * self.num_envs  # 각 환경별 리셋 이유를 저장하는 리스트 초기화
        
        # 최종 리셋 여부 저장 텐서
        reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) 

        # 몸체 접촉, body 파트들이 1N 이상의 힘을 받으면 넘어진 것으로 간주
        self.body_contact = torch.any(torch.norm(self.contact_forces[:,[0,1,2,5,6,9,10,13,14],:],dim=-1) >1., dim=1)
        body_contact_reset = self.body_contact
        reset = reset | body_contact_reset
        # for i in torch.where(body_contact_reset)[0]:
        #     reset_reasons[i] = "신체 접촉 발생"

        # 에피소드 시간 초과(학습 모드에서만), 에피소드 길이 초과 여부 확인 
        if not (self.test_mode):
            # progress_buf는 현재 시뮬레이션 프레임 수 의미
            self.time_out = self.progress_buf >= self.max_episode_length - 1
            time_out_reset = self.time_out
            reset = reset | time_out_reset
            # for i in torch.where(time_out_reset)[0]:
            #     reset_reasons[i] = "에피소드 타임 아웃"

        # 테스트 모두 수동 리셋, need_reset플래그가 활성화되면 리셋 수행
        if self.test_mode:
            need_reset_reset = self.need_reset
            reset = reset | need_reset_reset
            # for i in torch.where(need_reset_reset)[0]:
            #     reset_reasons[i] = "테스트 모드에서 리셋 필요"

        # 관절 리밋
        self.dof_limit_lower = torch.any(self.dof_pos <= self.reset_dof_pos_limits[:, 0], dim=-1)
        dof_limit_lower_reset = self.dof_limit_lower
        reset = reset | dof_limit_lower_reset
        # for i in torch.where(dof_limit_lower_reset)[0]:
        #     reset_reasons[i] = "관절 하한 제한 초과"

        # if self.dof_limit_lower[environment_index]:  # 206번째 환경의 하한 제한 초과 여부 확인 및 출력
        #     print(f"206번째 환경에서 하한 제한을 벗어난 관절:")
        #     for j in range(self.dof_pos.shape[1]):  # 각 관절에 대해 반복
        #         if self.dof_pos[environment_index, j] <= self.dof_pos_limits[j, 0]:
        #             print(f"  관절 {j}: 각도 {self.dof_pos[environment_index, j].item()}, 제한 {self.dof_pos_limits[j, 0].item()}")

        self.dof_limit_upper = torch.any(self.dof_pos >= self.reset_dof_pos_limits[:, 1], dim=-1)
        dof_limit_upper_reset = self.dof_limit_upper
        reset = reset | dof_limit_upper_reset
        # for i in torch.where(dof_limit_upper_reset)[0]:
        #     reset_reasons[i] = "관절 상한 제한 초과"

        # if self.dof_limit_upper[environment_index]:  # 206번째 환경의 상한 제한 초과 여부 확인 및 출력
        #     print(f"206번째 환경에서 상한 제한을 벗어난 관절:")
        #     for j in range(self.dof_pos.shape[1]):  # 각 관절에 대해 반복
        #         if self.dof_pos[environment_index, j] >= self.dof_pos_limits[j, 1]:
        #             print(f"  관절 {j}: 각도 {self.dof_pos[environment_index, j].item()}, 제한 {self.dof_pos_limits[j, 1].item()}")

        # self.dof_vel_over = torch.any((torch.abs(self.dof_vel) >= -0.15 * torch.abs(self.torques) + 13.404),dim=-1)
        # dof_vel_over_reset = self.dof_vel_over
        # reset = reset | dof_vel_over_reset
        # for i in torch.where(dof_vel_over_reset)[0]:
        #     reset_reasons[i] = "관절 속도 제한 초과"

        # if reset[environment_index]:
        #     print(f"206번째 환경 리셋됨. 이유: {reset_reasons[environment_index]}")

        return reset

    # obs을 계산하여 policy에 전달할 obs_buf를 구성  ???왜 계산? 
    def compute_observations(self):
        base_quat = self.root_states[:, 3:7]

        # 기준 좌표계에서의 각속도 계산 후 스케일 적용, quat rotate inverse는 global -> local로 변환
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13]) * self.obs_scales["angularVelocityScale"]
        # noise = torch_rand_float(-0.085,0.085, (self.num_envs,3), device=self.device).squeeze()
        # base_ang_vel+=noise
        
        # privileged info
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        # relative_foot_height = self.foot_pos[:,:,2] - self.mean_measured_legs_heights

        # 발 접촉 여부 판단, 1N 이상이면 컨택으로 인지
        foot_contact =  torch.any(self.contact_forces[:, self.feet_indices, :],dim=-1) > 1.
        # print("foot_contact : ", foot_contact[0,:])
        dof_pos_scaled = (self.dof_pos) * self.obs_scales["dofPositionScale"]
        dof_vel_scaled = self.dof_vel * self.obs_scales["dofVelocityScale"]

        self.dof_pos_error = self.dof_pos - self.targets

        # 로봇의 기존 좌표계에서 중력 벡터를 표현(균형 관련 관측에 중요)
        self.projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        # 주기 기반 보행 컨트롤 정보
        self.phi = torch.tensor([0., torch.pi, torch.pi, 0.], device=self.device, dtype=torch.float)
        self.sin_cycle = torch.sin(2.*torch.pi*(self.cycle_t.unsqueeze(1)/self.cycle_time) + self.phi)
        self.cos_cycle = torch.cos(2.*torch.pi*(self.cycle_t.unsqueeze(1)/self.cycle_time) + self.phi)
        
        # priviledged 안씀?
        self.privileged_buf = torch.cat((self.root_states[:, 0:3],   
                                         base_lin_vel,      # 3
                                         self.mean_measured_legs_heights,       # 4
                                         self.rigid_body_pos[:,[4,8,12,16],0],  # 4
                                         self.rigid_body_pos[:,[4,8,12,16],1],  # 4
                                         self.rigid_body_pos[:,[4,8,12,16],2],  # 4
                                         self.contact_forces[:, self.feet_indices, 0]*0.01,    # 4
                                         self.contact_forces[:, self.feet_indices, 1]*0.01,    # 4
                                         self.contact_forces[:, self.feet_indices, 2]*0.01,      # 4
                                         ),dim=-1)          # 31

        self.obs_buf = torch.cat((base_ang_vel,                 # 3   [0:3]          
                                    self.projected_gravity,       # 3   [3:6]
                                    self.commands[:, :3],         # 3   [6:9]            
                                    dof_pos_scaled,               # 12  [9:21]                 
                                    dof_vel_scaled,               # 12  [21:33]                
                                    self.actions,                 # 12  [33:45]
                                    self.sin_cycle,               # 4   [45:49]        
                                    self.cos_cycle,               # 4   [49:53]
                                    ), dim=-1)                    # 54  
        
        # obs에 노이즈 추가
        self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # print("self.obs_buf_t : ", self.obs_buf_t.size())
        # self.obs_buf = torch.cat((self.obs_buf_t,
        #                           self.last_obs_bufs[4],
        #                           self.last_obs_bufs[9]), dim=-1)

        # self.obs_buf = torch.cat((self.obs_buf_t,
        #                           self.last_proprioceptive_bufs[24], # t-25
        #                           self.last_proprioceptive_bufs[49],
        #                           self.last_proprioceptive_bufs[74],
        #                           self.last_proprioceptive_bufs[99]), dim=-1)    

        # self.states_buf = torch.cat((self.obs_buf_t, 
        #                              self.privileged_buf), dim=-1)
        
        # print("========================================")        
        # print("self.obs_buf_t : ", self.obs_buf_t[0,:])
        # for i in range(100):
        #     print("self.last_obs_bufs ", i, " : ", self.last_obs_bufs[i][0,:])
        # print("========================================")

    # reward를 위한 함수
    def compute_reward(self):
        
        # 발 위치와 속도
        self.foot_pos= self.rigid_body_pos[:,[4,8,12,16],:]
        foot_velocities = self.rigid_body_vel[:,[4,8,12,16],:]
        # self.standing = torch.norm(self.commands,dim=-1) == 0.
        
        # 명령이 없는 상태인지 확인
        self.no_commands = (torch.norm(self.commands,dim=-1) == 0)
        no_cycle    = (self.cycle_t == 0) & (self.last_cycle_t == 0) 

        # self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # 발 속도 norm
        foot_velocities = torch.norm(self.rigid_body_vel[:,[4,8,12,16],:],dim=-1)

        # 종아리와 발 컨택 힘 계산
        calf_contact_forces = torch.norm(self.contact_forces[:, [3,7,11,15], :],dim=-1)
        foot_contact_forces = self.contact_forces[:, [4,8,12,16], :]

        # global 축 기준으로 접촉력 분해
        global_x = torch.tensor([1.0, 0.0, 0.0], device=self.contact_forces.device)
        global_y = torch.tensor([0.0, 1.0, 0.0], device=self.contact_forces.device)
        global_z = torch.tensor([0.0, 0.0, 1.0], device=self.contact_forces.device)

        global_x_components = foot_contact_forces * global_x.unsqueeze(0).unsqueeze(0)
        global_y_components = foot_contact_forces * global_y.unsqueeze(0).unsqueeze(0)
        global_z_components = foot_contact_forces * global_z.unsqueeze(0).unsqueeze(0)

        # print("global_z_components : ", global_z_components[0,:])

        # 발 접촉 여부 판단
        self.foot_contact = torch.sum(torch.square(global_z_components),dim=-1) > 1.

        #===========================================< Task Rewards (Pos) >==============================================    
        # velocity tracking reward , square: 제곱
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error/0.125) * self.rew_scales["rew_ang_vel_z"]

        lin_vel_error_xy = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2],dim=-1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error_xy/0.25) * self.rew_scales["rew_lin_vel_xy"] 

        #===========================================< Task Penalties (Neg) >==============================================
        # other base velocity penalties
        penalty_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["penalty_lin_vel_z"]
        penalty_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["penalty_ang_vel_xy"]
        penalty_base_acc = abs(self.last_base_vel[:,2] - self.base_vel[:,2]) * self.rew_scales["penalty_base_acc"]

        # dof_acc_zero_crossing
        # 관절 가속도가 0을 교차할 경우 패널티??????? 전환 진동 안되게
        self.dof_acc = (self.dof_vel -self.last_dof_vel)
        dof_acc_zero_crossing = torch.where(self.dof_acc * self.last_dof_acc < 0.,
                                             torch.ones_like(self.dof_acc),
                                             torch.zeros_like(self.dof_acc))
        penalty_dof_acc_zero_crossing = torch.sum(dof_acc_zero_crossing,dim=-1) * self.rew_scales["penalty_dof_acc_zero_crossing"] 
        penalty_dof_acc_zero_crossing = torch.where(no_cycle, 
                                                    penalty_dof_acc_zero_crossing,
                                                    penalty_dof_acc_zero_crossing*0.05)

        # orientation penalty
        # non = abs(self._normal[:,2])>0.95
        # dot_product = torch.sum(self.projected_gravity * self._normal, dim=-1)
        # penalty_base_ori = torch.abs(1+dot_product) * self.rew_scales["penalty_base_ori"] *non

        # 자세 안정성 (기울기)
        base_ori_err = torch.norm(self.projected_gravity[:,:2],dim=-1) 
        penalty_base_ori = base_ori_err * self.rew_scales["penalty_base_ori"]




        # base height penalty
        relative_base_height = self.base_pos[:,2]
        ref_body_height = torch.where(self.no_commands,
                              torch.tensor(0.338, device=self.device),
                              torch.tensor(0.338, device=self.device))
        base_height_err = abs((ref_body_height + self.mean_measured_heights) - relative_base_height)
        penalty_base_height = base_height_err * self.rew_scales["penalty_base_height"]
        # penalty_body_height = torch.abs(ref_body_height - self.base_pos[:,2]) * self.rew_scales["penalty_body_height"]
        

        # collision penalty, 종아리 접촉 감지 패널티
        calf_contact = torch.norm(self.contact_forces[:, self.calf_indices, :], dim=2) > 0.
        penalty_collision = torch.sum(calf_contact, dim=1) * self.rew_scales["penalty_collision"] # sum vs any ?

        # penalty_dof_pos_limits , 관절 제한 위반 시 패널티
        limits = torch.stack([self.roll_limits, self.hp_limits, self.knp_limits,
                              self.roll_limits, self.hp_limits, self.knp_limits,
                              self.roll_limits, self.hp_limits, self.knp_limits,
                              self.roll_limits, self.hp_limits, self.knp_limits])

        lower_limits = limits[:, 0]
        upper_limits = limits[:, 1]

        # print("limits : "  ,limits)

        out_of_lower = torch.where(self.dof_pos < lower_limits,
                                    abs(self.dof_pos - lower_limits),
                                    torch.zeros_like(self.dof_pos))  # lower limit 벗어남

        out_of_upper = torch.where(self.dof_pos > upper_limits,
                                    abs(self.dof_pos - upper_limits),
                                    torch.zeros_like(self.dof_pos))  # upper limit 벗어남

        penalty_dof_pos_limits = torch.sum(out_of_lower + out_of_upper, dim=1) * self.rew_scales["penalty_dof_pos_limits"]

        #penalty_actions, 액션 제한 패널티
        scaled_actions = self.action_scale * self.actions
        scaled_actions[:,[0,3,6,9]] *= 0.25
        out_of_limits_actions = torch.where(scaled_actions< lower_limits,
                                        abs(scaled_actions - lower_limits),
                                        torch.zeros_like(self.actions))
        out_of_limits_actions += torch.where(scaled_actions> upper_limits,
                                        abs(scaled_actions - upper_limits),
                                        torch.zeros_like(self.actions))
        penalty_actions = torch.sum(out_of_limits_actions, dim=1)*self.rew_scales["penalty_actions"]


        # stumbling penalty
        # stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 2.5) * (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < 1.)
        # penalty_stumble = torch.sum(stumble, dim=1) * self.rew_scales["penalty_stumble"]

        # cosmetic penalty for hip roll motion
        # 힙 롤 움직임이 명령 없는 상황에서 발생할 경우 패널티
        y_cmd_zero = self.commands_y == 0
        yaw_cmd_zero = self.commands_yaw == 0
        roll_pos = torch.sum(torch.abs(self.dof_pos[:,[0,3,6,9]]),dim=-1) * y_cmd_zero * yaw_cmd_zero
        penalty_roll_pos = roll_pos * self.rew_scales["penalty_roll_pos"] 

        # penalty_slip, 접축 중 미끄러지면 패널티
        contact = torch.norm((global_z_components),dim=-1) > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_xy_velocities = torch.sum(torch.square(self.rigid_body_vel[:,[4,8,12,16],:2]),dim=-1)
        penalty_slip = torch.sum(contact_filt * foot_xy_velocities, dim=1) * self.rew_scales["penalty_slip"]

        # penalty_num_contact_feet
        # 접촉 발이 1 or 3개 이면 패널티 
        self.num_contact_feet = torch.sum(self.contact_forces[:, [4, 8, 12, 16], 2] > 1., dim=-1)
        num_contact_feet_err = torch.where((self.num_contact_feet == 1)|(self.num_contact_feet == 3),
            torch.ones_like(penalty_slip),
            torch.zeros_like(penalty_slip))
        
        penalty_num_contact_feet = num_contact_feet_err * self.rew_scales["penalty_num_contact_feet"]

        # 접촉 발이 0인경우 패널티
        zero_contact_feet_err =torch.where((self.num_contact_feet == 0),
                        torch.ones_like(penalty_slip),
                        torch.zeros_like(penalty_slip))      
        
        penalty_zero_contact_feet = zero_contact_feet_err * self.rew_scales["penalty_zero_contact_feet"]

        # 스탠딩 시 4개의 발이 모두 닾지 않으면 패널티
        penalty_default_num_contact_feet = torch.where(self.num_contact_feet == 4,
                                                    torch.zeros_like(penalty_slip),
                                                    torch.ones_like(penalty_slip) * self.rew_scales["penalty_default_num_contact_feet"] * no_cycle)

        # grf 불균형 패널티 , 스탠딩 상태에서 각 발에 가해지는 수직 힘의 편차
        mean_grf = torch.mean(global_z_components,dim=-1)
        mean_grf_z = torch.mean(mean_grf,dim=-1)
        grf_err =  torch.sum(torch.abs(mean_grf_z.unsqueeze(1) - mean_grf),dim=-1)
        penalty_default_grf = grf_err * self.rew_scales["penalty_default_grf"] * no_cycle

        # print("mean_grf : ", mean_grf[0])
        # print("mean_grf_z : ", mean_grf_z[0])
        # print("grf_err : ", grf_err[0])



        #penalty_
        # base = quat_rotate_inverse(self.base_quat,self.base_pos)
        # last_base = quat_rotate_inverse(self.base_quat,self.last_base_pos )
        # commands = quat_rotate_inverse(self.base_quat,self.commands )
        # base_pos_err = torch.norm(base[:,:2] - last_base[:,:2]+commands[:,:2],dim=-1)

        # print("base_pos : ", self.base_pos[201])
        # print("last_base_pos : ", self.last_base_pos[201])
        # print("base_pos_err : ", base_pos_err[201])        
        # penalty_base_pos = base_pos_err * self.rew_scales["penalty_base_pos"]

        # penalty_foot_height
        
        # self.ref_foot_height_relative = abs(abs(ref_foot_height * self.cos_cycle) + abs((self.base_pos[:,2] - ref_foot_height)))
        # foot_height_err = abs(abs(self.base_pos[:,2].unsqueeze(1) - self.foot_pos[:,:,2]) - self.ref_foot_height_relative.unsqueeze(1))* ~self.foot_contact

        # 스윙 타임이 너무 짧은 경우 페널티
        self.foot_swing_state[~self.foot_contact] = True
        self.foot_swing_state[self.foot_contact] = False
        
        current_time = self.gym.get_sim_time(self.sim)  # 현재 시뮬레이션 시간
        self.foot_swing_start_time[~self.foot_contact] = current_time
        swing_time = current_time - self.foot_swing_start_time
        short_swing = (swing_time < self.min_swing_time) & self.foot_swing_state

        penalty_short_swing = torch.sum(short_swing, dim=-1) * self.rew_scales["penalty_short_swing"]

        # print("penalty_foot_height size : ", penalty_foot_height.size())
        # print("penalty_short_swing size : ", penalty_short_swing.size())

        # print("short_swing : " ,short_swing[self.observe_envs])
        # ref_foot_vel = abs(self.ref_foot_height * self.cos_cycle)
        # foot_vel_err = abs(foot_velocities[:,:,2] - ref_foot_vel.unsqueeze(1)) * ~self.foot_contact
        # penalty_foot_vel = torch.sum(foot_vel_err,dim=-1)* self.rew_scales["penalty_foot_vel"]

        # print("swing_time : ", swing_time[201])
        # foot_force = abs(self.contact_forces[:, [4,8,12,16], 2])

        # 발 접촉 상태에서 너무 큰 수직 접촉력이 발생한 경우 패널티
        contact_foot_force = torch.sum(torch.square(global_z_components),dim=-1) * self.foot_contact
        penalty_foot_contact_force =  torch.sum(contact_foot_force, dim=-1)* self.rew_scales["penalty_foot_contact_force"]

        #Swing_phase, 스윙 및 스탠스 위상에 맞지 않는 움직임 감지
        cycle_lower = 0.6
        swing_phase = self.sin_cycle > cycle_lower
        stance_phase = self.sin_cycle < -cycle_lower  

        foot_forcese_global_z_components = torch.sum(torch.abs(global_z_components),dim=-1) * 0.05
        cliped_calf_contact_forces = torch.clip(calf_contact_forces,0,50)
        swing_stance_penalty =  torch.sum(foot_velocities*stance_phase, dim=-1) + torch.sum(foot_forcese_global_z_components*swing_phase, dim=-1) + torch.sum(cliped_calf_contact_forces*swing_phase,dim=-1)
        penalty_swing_stance_phase = swing_stance_penalty * ~self.no_commands * self.rew_scales["penalty_swing_stance_phase"]

        # 발에 가해지는 수평 방향 힘의 총합이 너무 크면 패널티
        xy_foot_forces = global_x_components + global_y_components
        global_foot_xy_forces = torch.clip(torch.sum(torch.square(xy_foot_forces),dim=-1),0,50)
        penalty_global_foot_xy_forces = torch.sum(global_foot_xy_forces, dim=-1) * self.rew_scales["penalty_global_foot_xy_forces"]


        # 스윙 위상 시 발 높이가 ref와 얼마나 다른지 측정   5cm 드는거임?
        ref_foot_h = 0.05
        # relative_foot_height = self.foot_pos[:,:,2]
        # print("self.mean_measured_legs_heights : ", self.mean_measured_legs_heights.size())
        # print("self.ref_foot_h : ", ref_foot_h.size())
        # print("self.sin_cycle : ", self.sin_cycle.size())
        ref_foot_height = (self.mean_measured_legs_heights + ref_foot_h) * abs(self.sin_cycle)  # sin으로 줌
        foot_height_err = abs(self.foot_pos[:,:,2] - ref_foot_height) * swing_phase
        penalty_foot_height = torch.sum(foot_height_err,dim=-1)*~no_cycle *self.rew_scales["penalty_foot_height"]

        # 기본 위치에서 벗어난 정도(스탠딩 상태)
        default_pos_err = torch.square(self.default_dof_pos - self.dof_pos)
        penalty_default_pos_standing = torch.sum(default_pos_err,dim=-1) * no_cycle * self.rew_scales["penalty_default_pos_standing"]

        # 정지 명령일 때 관절 속도가 0이 아니면 패널티
        default_vel_default = torch.sum(torch.square(self.dof_vel),dim=-1) * no_cycle
        penalty_zero_vel_standing = default_vel_default * self.rew_scales["penalty_zero_vel_standing"]


        # penalty_default_pos, 관절별 기본 위치와의 차이(hip, knee)
        default_pos_hp_err = torch.sum(torch.square(self.default_dof_pos - self.dof_pos)[:,[1,4,7,10]]* self.foot_contact,dim=-1) 
        default_pos_knp_err = torch.sum(torch.square(self.default_dof_pos - self.dof_pos)[:,[2,5,8,11]]* self.foot_contact,dim=-1) 
        penalty_default_pos_hp = default_pos_hp_err * self.rew_scales["penalty_default_pos_hp"] 
        penalty_default_pos_knp = default_pos_knp_err * self.rew_scales["penalty_default_pos_knp"]

        # trot, 트롯 자세가 어긋났는지 측정
        trot_pitch_err = torch.sum(abs(self.dof_pos[:,[1,2,4,5]] - self.dof_pos[:,[10,11,7,8]]),dim=-1)
        penalty_trot_pitch = trot_pitch_err * self.rew_scales["penalty_trot_pitch"]

        # base_pos_xyc, 베이스가 발 중심 위에 위치하지 않으면 패널티
        sup_foot_pos_xy = self.foot_pos[:,:,:2] * self.foot_contact.unsqueeze(-1)  
        sup_foot_pos_xy_mean = torch.sum(sup_foot_pos_xy,dim=1) / (self.num_contact_feet.unsqueeze(-1) + 1e-6)
        base_foot_xy_pos_err = torch.sum(torch.square(self.base_pos[:,:2] - sup_foot_pos_xy_mean),dim=-1)
        base_foot_xy_pos_err = torch.where(torch.all(sup_foot_pos_xy_mean,dim=-1)==0,
                                           torch.tensor(3.0,device=self.device),
                                             base_foot_xy_pos_err)
        penalty_base_foot_xy_pos = base_foot_xy_pos_err * self.rew_scales["penalty_base_foot_xy_pos"]

        #===========================================< Regulation Penalties (Neg) >==============================================
        # torque penalty, 과한 힘 방지 
        penalty_torques = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["penalty_torques"]

        # 기준 토크를 초과한 경우 패널티??????????????????????????????? 노미널 넘긴 넘음
        over_nominal = torch.where(self.torques >= 75, self.torques, torch.tensor(0.0))

        # 제곱 후 합 계산
        penalty_over_nominal = torch.sum(torch.square(over_nominal), dim=1) * self.rew_scales["penalty_over_nominal"]

        # joint vel penalty
        # 조건 정의?????????????값은 무슨 기준? urdf에서 설정했었는데 여기서 다시 해야함?  시트상 속도에 따른 토크값 반영
        vel_violation = -0.15 * torch.abs(self.torques) + 13.404

        # 조건 위반 정도 계산
        threshold = 0.85 * vel_violation
        joint_vel = abs(self.dof_vel) 
        joint_vel = torch.where(joint_vel > threshold, 
                                torch.where(joint_vel > vel_violation,
                                            joint_vel*2.5,
                                            joint_vel*1.5),
                                joint_vel)
        joint_vel = torch.sum(torch.square(joint_vel), dim=1)
        penalty_joint_vel = joint_vel * self.rew_scales["penalty_joint_vel"]




        # action rate penalty
        # rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) ##* self.rew_scales["action_rate"]


        # penalty_action_smoothness1, 1차 미분 수준의 스무스
        action_smoothness1_diff = torch.square(self.actions-self.last_actions)
        action_smoothness1_diff *= (self.last_actions != 0)
        penalty_action_smoothness1 = torch.sum(action_smoothness1_diff, dim=1) * self.rew_scales["penalty_action_smoothness1"]
        
        # penalty_action_smoothness2, 2차 미분 수준의 액션 변화율 패널티(jerk 최소화)
        action_smoothness2_diff = torch.square(self.actions - 2*self.last_actions + self.last_actions2)
        action_smoothness2_diff *= (self.last_actions != 0)
        action_smoothness2_diff *= (self.last_actions2 != 0)
        penalty_action_smoothness2 = torch.sum(action_smoothness2_diff, dim=1) * self.rew_scales["penalty_action_smoothness2"]

        
        #===========================================< Calculate Rewards >==============================================

        task_reward = rew_lin_vel_xy + rew_ang_vel_z #+rew_lin_vel_x + rew_lin_vel_y

        task_penalty =  penalty_actions+ penalty_lin_vel_z + penalty_ang_vel_xy + penalty_base_ori + penalty_num_contact_feet+\
                        penalty_dof_acc_zero_crossing+penalty_base_acc  + penalty_zero_vel_standing+\
                        penalty_default_grf + penalty_short_swing + penalty_zero_contact_feet + penalty_default_num_contact_feet + penalty_base_foot_xy_pos + penalty_foot_contact_force + penalty_collision +\
                        penalty_roll_pos + penalty_slip +penalty_default_pos_standing+ penalty_default_pos_hp+penalty_default_pos_knp+\
                        penalty_global_foot_xy_forces + penalty_trot_pitch + penalty_swing_stance_phase + penalty_dof_pos_limits + penalty_foot_height + penalty_base_height
        
        regulation_penalty = penalty_over_nominal+ penalty_torques + penalty_action_smoothness1 + penalty_action_smoothness2 + penalty_joint_vel
        
        total_rew = task_reward + task_penalty + regulation_penalty

        for key in self.cfg["env"]["learn"]["reward"].keys():
                self.reward_container[key] = locals()[key][self.observe_envs]
        
        # print("reward_container : ", self.reward_container)
        # self.reward_container ={
        #     #====================== <Rewards> ====================== 
        #     "rew_lin_vel_x"            : rew_lin_vel_x[self.observe_envs],
        #     "rew_lin_vel_y"            : rew_lin_vel_y[self.observe_envs],
        #     "rew_ang_vel_z"             : rew_ang_vel_z[self.observe_envs],

        #     # #====================== <Penalty> ====================== 
        #     "penalty_lin_vel_z"                : penalty_lin_vel_z[self.observe_envs],
        #     "penalty_ang_vel_xy"               : penalty_ang_vel_xy[self.observe_envs],
        #     "penalty_base_ori"                 : penalty_base_ori[self.observe_envs],
        #     "penalty_base_height"              : penalty_base_height[self.observe_envs],
        #     "penalty_collision"                : penalty_collision[self.observe_envs],
        #     "penalty_stumble"                  : penalty_stumble[self.observe_envs],
        #     "penalty_roll_pos"                 : penalty_roll_pos[self.observe_envs],
        #     "penalty_slip"                     : penalty_slip[self.observe_envs],
        #     "penalty_foot_height"              : penalty_foot_height[self.observe_envs],
        #     "penalty_num_contact_feet"         : penalty_num_contact_feet[self.observe_envs],
        #     "penalty_swing_phase"              : penalty_swing_phase[self.observe_envs],
        #     "penalty_default_pos_hp"              : penalty_default_pos_hp[self.observe_envs],
        #     "penalty_default_pos_knp"              : penalty_default_pos_knp[self.observe_envs],
        #     "penalty_dof_pos_limits"           : penalty_dof_pos_limits[self.observe_envs],
        #     "penalty_dof_acc_zero_crossing"   : penalty_dof_acc_zero_crossing[self.observe_envs],
        #     "penalty_base_acc"               : penalty_base_acc[self.observe_envs],
        #     "penalty_foot_vel"              : penalty_foot_vel[self.observe_envs],

        #     "penalty_torques"                  : penalty_torques[self.observe_envs],
        #     "penalty_action_smoothness1"       : penalty_action_smoothness1[self.observe_envs],   
        #     "penalty_action_smoothness2"       : penalty_action_smoothness2[self.observe_envs],
        #     "penalty_joint_vel"                : penalty_joint_vel[self.observe_envs],

        #     "total_rew"                        : torch.clip(total_rew[self.observe_envs],0,None)
        # }

        # print("===========================================================")
        # for reward_name, reward_tensor in self.reward_container.items():                    
        #             if "rew" in reward_name:
        #                 print(f"rew/{reward_name}",reward_tensor /self.dt)
        # print("===========================================================")
        # for reward_name, reward_tensor in self.reward_container.items():                    
        #             if "penalty" in reward_name:
        #                 print(f"Penalty/{reward_name}",reward_tensor/self.dt)
        # print("===========================================================")

        # print("===========================================================")
        # print("commands x : ",self.commands[0,0])
        # print("base_lin_x : ",self.base_lin_vel[0,0])
        # print("===========================================================")
        # print("commands y : ",self.commands[0,1])
        # print("base_lin_y : ",self.base_lin_vel[0,1])
        # print("===========================================================")
        # print("commands yaw : ",self.commands[0,2])
        # print("base_ang_vel_z : ",self.base_ang_vel[0,2])
        # print("===========================================================")

        # total reward
        self.rew_buf = total_rew
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

    def plot_juggler(self):
        
        base_lin_vel_x = self.base_lin_vel[self.observe_envs,0]
        base_lin_vel_y = self.base_lin_vel[self.observe_envs,1]
        base_ang_vel_z = self.base_ang_vel[self.observe_envs,2]

        commands_x  = self.commands_x[self.observe_envs]
        commands_y  = self.commands_y[self.observe_envs]
        commands_yaw  = self.commands_yaw[self.observe_envs]

        # commands        
        command_obs_msg = Twist()
        command_obs_msg.linear.x = commands_x
        command_obs_msg.linear.y = commands_y
        command_obs_msg.angular.z = commands_yaw

        # swing_foot_height = torch.mean(self.foot_pos[self.observe_envs,:,2] * ~self.foot_contact,dim=1)

        # ref_foot_height = Float32()
        # foot_height = Float32()

        # ref_foot_height.data = self.ref_foot_height_relative[self.observe_envs]
        # foot_height.data = self.base_pos[self.observe_envs,2] - swing_foot_height[self.observe_envs]
        
        # ref_foot_height_pub = rospy.Publisher('/ref_foot_height_pub', Float32, queue_size=10)
        # foot_height_pub = rospy.Publisher('/foot_height_pub', Float32, queue_size=10)

        # ref_foot_height_pub.publish(ref_foot_height)
        # foot_height_pub.publish(foot_height)

        # cycle
        # cycle = Point()
        # cycle.x = self.sin_cycle[self.observe_envs]
        # cycle.y = self.cos_cycle[self.observe_envs]
        # cycle.z = self.cycle_t[self.observe_envs]

        contact_ref = Point()
        contact_ref.x = self.FL_RR_swing_phase[self.observe_envs]
        contact_ref.y = self.FR_RL_swing_phase[self.observe_envs]

        # base_state
        twist_msg = Twist()
        twist_msg.linear.x = base_lin_vel_x
        twist_msg.linear.y = base_lin_vel_y
        twist_msg.angular.z = base_ang_vel_z

        # foot_contact_state
        fl_contact_msg = Bool()
        fr_contact_msg = Bool()
        rl_contact_msg = Bool()
        rr_contact_msg = Bool()

        fl_contact_msg.data = self.foot_contact[self.observe_envs, 0]  # FL
        fr_contact_msg.data = self.foot_contact[self.observe_envs, 1]  # FR
        rl_contact_msg.data = self.foot_contact[self.observe_envs, 2]  # RL
        rr_contact_msg.data = self.foot_contact[self.observe_envs, 3]  # RR

        for reward_name, reward_value in self.reward_container.items():
            reward_msg = Float32()
            reward_msg.data = reward_value /self.dt
            reward_pub = rospy.Publisher(f"/{reward_name}", Float32, queue_size=10)
            reward_pub.publish(reward_msg)

        tot_reward_msg = Float32()
        tot_reward_msg.data = self.rew_buf[self.observe_envs] / self.dt
        tot_reward_pub = rospy.Publisher(f"/tot_reward", Float32, queue_size=10)
        tot_reward_pub.publish(tot_reward_msg)
        

        Joint_state = JointState()
        Joint_state.name = self.dof_names
        Joint_state.position = abs(self.dof_pos[self.observe_envs,:])
        Joint_state.velocity = abs(self.dof_vel[self.observe_envs,:])
        Joint_state.effort   = abs(self.torques[self.observe_envs,:])
        

        self.ref_torques += self.plot_cnt
        if (self.ref_torques >= 19) | (self.ref_torques <= 0):
            self.plot_cnt *= -1

        ref_dof_vel = -0.15 * self.ref_torques + 13.404

        Joint_torque_vel = Point()
        Joint_torque_vel.x = self.ref_torques
        Joint_torque_vel.y = ref_dof_vel
        Joint_torque_vel_pub = rospy.Publisher('/Joint_torque_vel', Point, queue_size=10)
        Joint_torque_vel_pub.publish(Joint_torque_vel)


        FR_foot_pos = Point()
        FR_foot_pos.x= self.foot_pos[self.observe_envs,1,0]
        FR_foot_pos.y= self.foot_pos[self.observe_envs,1,0]
        FR_foot_pos.z= self.foot_pos[self.observe_envs,1,0]

        actions = JointState()
        actions.name = self.dof_names
        actions.position = self.actions[self.observe_envs,:]

        # ================================= publish =================================

        # ROS Publisher 생성
        command_obs_pub = rospy.Publisher('/command_obs_pub', Twist, queue_size=10)
        cmd_vel_obs_pub = rospy.Publisher('/cmd_vel_obs_pub', Twist, queue_size=10)
        
        # 발 컨택
        fl_contact_pub = rospy.Publisher('/fl_contact', Bool, queue_size=10)
        fr_contact_pub = rospy.Publisher('/fr_contact', Bool, queue_size=10)
        rl_contact_pub = rospy.Publisher('/rl_contact', Bool, queue_size=10)
        rr_contact_pub = rospy.Publisher('/rr_contact', Bool, queue_size=10)

        # cycle_pub = rospy.Publisher('/cycle', Point, queue_size=10)
        contact_ref_pub = rospy.Publisher('/contact_ref', Point, queue_size=10)

        #Joint_state
        joint_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        joint_state_pub.publish(Joint_state)

        #actions_pub
        actions_pub = rospy.Publisher('/isaacgym_actions', JointState, queue_size=10)
        actions_pub.publish(actions)

        # commands
        command_obs_pub.publish(command_obs_msg)
        cmd_vel_obs_pub.publish(twist_msg)

        # cycle
        # sin_cycle_pub.publish(sin_cycle)
        # cos_cycle_pub.publish(cos_cycle)
        # cycle_pub.publish(cycle)
        contact_ref_pub.publish(contact_ref)
        # 발 컨택
        fl_contact_pub.publish(fl_contact_msg)
        fr_contact_pub.publish(fr_contact_msg)
        rl_contact_pub.publish(rl_contact_msg)
        rr_contact_pub.publish(rr_contact_msg)
    
    def joy_callback(self, data):

        self.commands_x[:] = data.axes[1] * self.command_x_range[1]  # x vel
        self.commands_y[:] = data.axes[0] * self.command_y_range[1]  # y vel
        self.commands_yaw[:] = data.axes[3] * self.command_yaw_range[1]  # yaw vel

        self.need_reset = data.buttons[0] * data.buttons[1]

        if self.cam_change_flag == False:
            if self.cam_change_cnt < 50:
                self.cam_change_cnt += 1
            else:
                self.cam_change_flag = True
        
        if (self.cam_change_flag)&(data.buttons[4] == 1 and data.buttons[5] == 1):
            self.cam_mode = (self.cam_mode + 1) % 4  # 0, 1, 2, 3 순환
            if self.cam_mode == 0:
                print(f"fix_cam 상태 변경: {self.cam_mode} (자유 시점)")
            elif self.cam_mode == 1:
                print(f"fix_cam 상태 변경: {self.cam_mode} (고정 시점)")
            elif self.cam_mode == 2:
                print(f"fix_cam 상태 변경: {self.cam_mode} (1인칭 시점)")
            elif self.cam_mode == 3:
                print(f"fix_cam 상태 변경: {self.cam_mode} (3인칭 시점)")
            self.cam_change_flag = False
            self.cam_change_cnt = 0


        if data.buttons[3]:
            self.push_robots()

        # print("self.ref_foot_height : " , self.ref_foot_height[self.observe_envs])
        # print("self.foot_h : " , self.foot_pos[self.observe_envs,:,2])

    # 발 보행 위상을 설정하는 함수
    def set_phi(self, env_ids):
        # 트롯을 위한 패턴 설정
        self.phi[env_ids] = torch.tensor([0., torch.pi, torch.pi, 0.], device=self.device, dtype=torch.float)
        
        # env_ids 길이만큼 0~1 사이의 난수를 생성하여 각 환경에 대해 어떤 보행 패턴을 적용할지 결정할 확률
        probabilities = torch.rand(len(env_ids), device=self.device)
        
        # 50% 확률로 트롯 보행 선택
        trot_ids = env_ids[probabilities < 0.5]
        
        # 25% 확률로 러닝 보행????????
        running_ids = env_ids[(probabilities >= 0.5) & (probabilities < 0.75)]

        # 25% 확률로 페이싱 보행
        pacing_ids = env_ids[probabilities >= 0.75]

        self.phi[trot_ids] = torch.tensor([0., torch.pi, torch.pi, 0.], device=self.device, dtype=torch.float)
        self.phi[running_ids] = torch.tensor([0., 0.5* torch.pi, torch.pi, 1.5*torch.pi], device=self.device, dtype=torch.float)
        self.phi[pacing_ids] = torch.tensor([0., torch.pi, 0., torch.pi], device=self.device, dtype=torch.float)

    # 여러 환경에 대해 이동 명령을 설정하는 함수
    # 로봇의 x, y, yaw 명령을 무작위 혹은 조건에 따라 주입, 각 명령은 환경 ID(env_ids)마다 다르게 설정됨     
    # 전체 환경의 수 4096, env_ids: 리셋은 되는 환경은1
    def set_cmd(self,env_ids):
        #x, y, yaw 명령을 무작위로 생성하여 설정
        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids),1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids),1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids),1), device=self.device).squeeze()

        #0.05이하는 0으로 처리 , 정지 명령으로 간주
        self.commands_x[env_ids] = torch.where(torch.abs(self.commands_x[env_ids]) <= 0.05, torch.tensor(0.0, device=self.device), self.commands_x[env_ids])
        self.commands_y[env_ids] = torch.where(torch.abs(self.commands_y[env_ids]) <= 0.05, torch.tensor(0.0, device=self.device), self.commands_y[env_ids])
        self.commands_yaw[env_ids] = torch.where(torch.abs(self.commands_yaw[env_ids]) <= 0.05, torch.tensor(0.0, device=self.device), self.commands_yaw[env_ids])


        # 특정 환경 범위에 대해 명령 고정 설정
        # 학습용 커맨드, stand상태만 유지 
        no_command_env_ids = env_ids[(env_ids >= self.stand_env_range[0]) & (env_ids <= self.stand_env_range[1])]
        self.commands_x[no_command_env_ids] = 0.
        self.commands_y[no_command_env_ids] = 0.
        self.commands_yaw[no_command_env_ids] = 0. 
        
        # only_plus_x = env_ids[(env_ids >= self.only_plus_x_envs_range[0]) & (env_ids <= self.only_plus_x_envs_range[1])]
        # self.commands_x[only_plus_x] = torch_rand_float(0, self.command_x_range[1], (len(only_plus_x),1), device=self.device).squeeze()
        # self.commands_y[only_plus_x] = 0.
        # self.commands_yaw[only_plus_x] = 0. 
        # only_minus_x = env_ids[(env_ids >= self.only_minus_x_envs_range[0]) & (env_ids <= self.only_minus_x_envs_range[1])]  
        # self.commands_x[only_minus_x] = torch_rand_float(self.command_x_range[0], 0, (len(only_minus_x),1), device=self.device).squeeze()
        # self.commands_y[only_minus_x] = 0.
        # self.commands_yaw[only_minus_x] = 0. 

        # x 방향으로만 이동 
        only_plus_x = env_ids[(env_ids >= self.only_plus_x_envs_range[0]) & (env_ids <= self.only_plus_x_envs_range[1])]
        self.commands_x[only_plus_x] = torch_rand_float(0, self.command_x_range[1], (len(only_plus_x),1), device=self.device).squeeze()
        self.commands_y[only_plus_x] = 0.
        self.commands_yaw[only_plus_x] = 0. 
        only_minus_x = env_ids[(env_ids >= self.only_minus_x_envs_range[0]) & (env_ids <= self.only_minus_x_envs_range[1])]  
        self.commands_x[only_minus_x] = torch_rand_float(self.command_x_range[0], 0, (len(only_minus_x),1), device=self.device).squeeze()
        self.commands_y[only_minus_x] = 0.
        self.commands_yaw[only_minus_x] = 0. 

        # y 방향으로만 이동
        only_plus_y = env_ids[(env_ids >= self.only_plus_y_envs_range[0]) & (env_ids <= self.only_plus_y_envs_range[1])]
        self.commands_x[only_plus_y] = 0.
        self.commands_y[only_plus_y] = self.command_y_range[1]
        self.commands_yaw[only_plus_y] = 0. 
        only_minus_y = env_ids[(env_ids >= self.only_minus_y_envs_range[0]) & (env_ids <= self.only_minus_y_envs_range[1])]
        self.commands_x[only_minus_y] = 0.
        self.commands_y[only_minus_y] = self.command_y_range[0]
        self.commands_yaw[only_minus_y] = 0. 

        # yaw 회전만 수행
        only_plus_yaw = env_ids[(env_ids >= self.only_plus_yaw_envs_range[0]) & (env_ids <= self.only_plus_yaw_envs_range[1])]
        self.commands_x[only_plus_yaw] = 0.
        self.commands_y[only_plus_yaw] = 0.
        self.commands_yaw[only_plus_yaw] = self.command_yaw_range[1]
        only_minus_yaw = env_ids[(env_ids >= self.only_minus_yaw_envs_range[0]) & (env_ids <= self.only_minus_yaw_envs_range[1])]
        self.commands_x[only_minus_yaw] = 0.
        self.commands_y[only_minus_yaw] = 0.
        self.commands_yaw[only_minus_yaw] = self.command_yaw_range[0]

        # +x 이동 + yaw 회전 동시에 수행
        plus_x_plus_yaw_envs_range = env_ids[(env_ids >= self.plus_x_plus_yaw_envs_range[0]) & (env_ids <= self.plus_x_plus_yaw_envs_range[1])]
        self.commands_x[plus_x_plus_yaw_envs_range] = self.command_x_range[1]
        self.commands_y[plus_x_plus_yaw_envs_range] = 0.
        self.commands_yaw[plus_x_plus_yaw_envs_range] = self.command_yaw_range[1]
        plus_x_minus_yaw_envs_range = env_ids[(env_ids >= self.plus_x_minus_yaw_envs_range[0]) & (env_ids <= self.plus_x_minus_yaw_envs_range[1])]
        self.commands_x[plus_x_minus_yaw_envs_range] = self.command_x_range[1]
        self.commands_y[plus_x_minus_yaw_envs_range] = 0.
        self.commands_yaw[plus_x_minus_yaw_envs_range] = self.command_yaw_range[0]

        # -x 이동 + yaw 회전 동시에 수행
        minus_x_plus_yaw_envs_range = env_ids[(env_ids >= self.minus_x_plus_yaw_envs_range[0]) & (env_ids <= self.minus_x_plus_yaw_envs_range[1])]
        self.commands_x[minus_x_plus_yaw_envs_range] = self.command_x_range[0]
        self.commands_y[minus_x_plus_yaw_envs_range] = 0.
        self.commands_yaw[minus_x_plus_yaw_envs_range] = self.command_yaw_range[1]
        minus_x_minus_yaw_envs_range = env_ids[(env_ids >= self.minus_x_minus_yaw_envs_range[0]) & (env_ids <= self.minus_x_minus_yaw_envs_range[1])]
        self.commands_x[minus_x_minus_yaw_envs_range] = self.command_x_range[0]
        self.commands_y[minus_x_minus_yaw_envs_range] = 0.
        self.commands_yaw[minus_x_minus_yaw_envs_range] = self.command_yaw_range[0]

    # 강화학습 환경의 일부 환경 인스턴스를 초기화
    # 보통 에피소드가 종료되거나 로봇이 쓰러졌을 때 해당 환경을 재설정
    def reset_idx(self, env_ids):
        # 환경의 물리 특성을 랜덤화하여 일반화 능력 샹항에 도움
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # 관절 위치는 기본자세어서 랜덤을 줘? 왜?   
        positions_offset = torch_rand_float(0.925, 1.075, (len(env_ids), self.num_dof), device=self.device)

        # 관절 속도 -0,1 ~ +0.1 사이 랜덤 값으로 초기화
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]  * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # 루트 상태 초기화(기본 위치와 자세)
        # custom_origins가 커져 있으면 환경마다 다른 위치로 로봇을 배치하고 terrain level동 갱신
        # +-0.25 정도 위치 오프셋을 적용해서 더 다양한 초기 위치에서 학습 가능
        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.25, 0.25, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state

        # isaac gym 내부 상태 업데이트
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # 보상 관련 설정 및 커맨드 초기화
        # 해당 환경에 대해 종료 시 보상 관련 초기화 
        # command도 새로 설정하여 새 에피소드에서 수행할 목표를 지정  
        self.termination_rew(env_ids)
        self.set_cmd(env_ids)
        # self.set_phi(env_ids)

               
        values = torch.arange(0.075, 0.15 + 0.005, 0.005)
        indices = torch.randint(len(values), (len(env_ids), 1))
        values += 0.02 

        # self.ref_foot_height[env_ids] = values[indices].to(self.device).squeeze() 
        # self.ref_foot_height = torch.where(torch.norm(self.commands,dim=-1) == 0,
        #                                    self.ref_foot_height * 0 + 0.02,
        #                                    self.ref_foot_height)
        # print("self.ref_foot_height 0  : ", self.ref_foot_height[0])
        # print("self.ref_foot_height 400  : ", self.ref_foot_height[400])
        # print("self.ref_foot_height 800  : ", self.ref_foot_height[800])


        # self.ref_foot_height[env_ids] = torch_rand_float([0.075, 0.15] (len(env_ids),1), device=self.device).squeeze()
        
        # 변수 초기화
        self.original_commands[env_ids] = self.commands[env_ids].clone()

        self.last_contacts[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fall_cnt[env_ids] = 0
        self.dof_recovery[env_ids] =0.
        self.last_base_pos[env_ids] =0.
        
        self.last_actions[env_ids] = 0.
        self.last_actions2[env_ids] = 0.

        self.last_dof_vel[env_ids] = 0.
        self.last_dof_vel2[env_ids] = 0.

        self.dof_acc[env_ids] = 0.
        self.last_dof_acc[env_ids] = 0.

        self.cycle_t[env_ids] = 0.
        self.last_cycle_t[env_ids] = 0.

        # for i in range(len(self.last_proprioceptive_bufs) - 1, 0, -1):
        #     self.last_proprioceptive_bufs[i][env_ids] = 0.

        # fill extras
        # 에피소드 summary 및 extras 갱신
        # 각 reward 항복의 평균값을 extras에 기록 -> 모니터링 또눈 텐서보드에 사용
        # terrain_level 평군도 기록
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    # 에피소드가 종료될 때 적용되는 보상을 계산하는 함수
    # 환경이 재설정될 때(reset_idx) 각 환경에 대해 적절한 종료 리워드를 남기는 용도
    def termination_rew(self, env_ids):
        # 관절 제한 위반 여부를 계산하는 부분
        dof_limit_over = self.dof_limit_lower | self.dof_limit_upper | self.dof_vel_over
        
        # 에피소드가 얼마나 오래 지속되었는지를 기반으로 소향의 보상을 부여
        # progress_buf: 해당 환경에서 진행된 스텝 수 -> 오래 살아남으면 아주 약간 보상
        self.rew_buf[env_ids] += self.progress_buf[env_ids] * 0.0001

        # 시간 초과 종료가 아닌 경우 전체 리워드를 25% 감쇠하여 패널티를 줌
        # 시간 내로 정상 종료한 경우 보상을 유지
        self.rew_buf[env_ids] = torch.where(~self.time_out[env_ids], self.rew_buf[env_ids]*0.25, self.rew_buf[env_ids])

        # print("self.time_out : ", self.time_out.size())
    
    # 커리큘럼 학습환경에서 현재 위치 성과에 따라 지형 난이도를 동적으로 조정하는 함수
    def update_terrain_level(self, env_ids):
        # init_done이 아직 완료되지 않았거나, curriculum 모드가 꺼져 있으면 함수 종료
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        
        # self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.1)

        # self.terrain_levels[env_ids] = torch.where((distance >= self.terrain.env_length/2 * ~self.no_commands[env_ids]) | self.time_out[env_ids],
        #                                            self.terrain_levels[env_ids] + 1,
        #                                            self.terrain_levels[env_ids])
        
        # 에이전트 위치와 환경 시작 위치 사이의 2d 거리 계산, 얼마나 앞으로 이동했는지 측정
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        
        # 거리에 비례해 난이도 보너스 보상 추가
        self.rew_buf[env_ids] += self.terrain_levels[env_ids] * 0.01 * 0.25 * distance
        # 아직 에피소드 중간 이하인 env들을 필터링, 너무 빨리 실패했을 경우를 찾기 위함
        time_limit = env_ids[self.progress_buf[env_ids] <= self.max_episode_length/2]
        # non_time_out_envs = env_ids[~self.time_out[env_ids]]
        # 로봇이 지형의 절반 이상이동했으면 난이도 한 단계 증가
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        
        # 에피소드가 중간도 안 갔는데 끝나면 난이도 한 단계 감소
        self.terrain_levels[time_limit] -= 1
        # self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
        # self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows

        # 난이도가 바뀐 만큼 각 로봇의 환경 시작 위치도 업데이트
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],
                                                            self.terrain_types[env_ids]]
    # 인위적으로 무작위 외력를 부여
    def push_robots(self):
        # 선형 x,y 무작위
        self.root_states[:, [7,8]] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device) # lin vel x/y
        # 회전속도 xyz 무작위
        self.root_states[:, [10,11,12]] = torch_rand_float(-0.5, 0.5, (self.num_envs, 3), device=self.device)
        # 시뮬레이션 내 상태를 갱신
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    
    # 시뮬레이션을 ㅎ나 스텝 수행하기 전 action->목표 관절 위치->토크 계산-> 시뮬레이션 적용 과정을 수행
    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        for _ in range(self.decimation): # 10ms 2ms 2.5ms
            scaled_actions = self.action_scale * self.actions
            scaled_actions[:,[0,3,6,9]] *= 0.25      
            targets = self.default_dof_pos.clone()  # 기본 관절 각도로 초기화
            targets += scaled_actions
            self.targets = targets.clone()
            # torques = torch.clip(self.Kp*(targets - self.dof_pos) - self.Kd*self.dof_vel,-17.,17.)
            noise_kp = random.uniform(0.95, 1.05)
            noise_kd = random.uniform(0.95, 1.05)
            torques = torch.clip((self.Kp * noise_kp) * (targets - self.dof_pos) - (self.Kd * noise_kd) * self.dof_vel, -200., 200.)
            # print("==========================")
            # print("targets : ", targets)
            # print("dof_pos : ", self.dof_pos)
            # print("errror : ", targets - self.dof_pos)
            # print("==========================")

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape)
            
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            
    # 시뮬레이션이 한 번 끝난 후에 수행되는 작업들을 모아 둔 함수
    # 시뮬레이션 1스텝 뒤에 수행되는 관측+보상+리셋+주기 관리+카메라 등을 처리하는 함수
    def post_physics_step(self):
        # print("progress_buf : ", self.progress_buf[self.observe_envs])
        # 이전 base 위치 저장 후, 현재 상태(위치, 속도 등)를 새로 읽어옴
        self.last_base_pos = self.base_pos.clone()
        self.refresh_state()

        # 에피소드 프레임 수, 랜덤화 카운터, 전체 흐레임 수, 주기 시간 등 업데이트
        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        self.cycle_t += 0.01        


        # 명령이 없는 경우 → 주기 시간 0으로 초기화 (로봇이 정지 상태인 경우)
        self.cycle_t = torch.where(torch.norm(self.commands,dim=-1) == 0.,
                                    self.cycle_t * 0.,
                                    self.cycle_t
                                    )
        # 주기 시간이 cycle_time(=전체 주기)보다 커지면 0.01로 초기화 → 반복 시작
        self.cycle_t = torch.where(self.cycle_t > self.cycle_time,
                                                        self.cycle_t * 0. + 0.01 ,
                                                        self.cycle_t)
        
        # print("self.leg_height_points: ", self.leg_height_points.size())

        # 일정 주기마다 **외부 힘(perturbation)**을 로봇에 가함 → 강건성 훈련
        if (self.common_step_counter % self.push_interval == 0) and not self.test_mode:
            self.push_robots()

        # if self.common_step_counter % self.zero_interval == 0 and not self.test_mode:
        #     self.cmd_change_zero()
        
        # if self.common_step_counter % self.change_interval == 0 and not self.test_mode:
        #     self.cmd_change()

        # 일정 주기마다 로봇의 명령을 멈춰 정지 상태로 유지하는 freeze 수행
        if self.common_step_counter % self.freeze_interval == 0 and not self.test_mode:
            self.freeze()
            self.freeze_flag = True
            self.freeze_steps = random.randint(100, 150)

        # freeze 상태가 일정 프레임 이상 지속되면 다시 명령을 활성화
        all_env_ids = torch.arange(self.num_envs)
        if self.freeze_flag:
            self.freeze_cnt += 1
            if self.freeze_cnt >= self.freeze_steps:
                self.set_cmd(all_env_ids)
                
                self.freeze_flag=False
                self.freeze_cnt = 0

        # if self.common_step_counter % self.foot_freeze_interval == 0 and not self.test_mode:
        #     self.freeze_dof_vel(torch.arange(self.num_envs, device=self.device), 3)  # 3개의 DoF를 랜덤하게 선택하여 속도를 0으로 설정

        # print("commands : ",self.commands)
        # prepare quantities

        # self.base_pos = self.root_states[:, :3]
        # self.base_quat = self.root_states[:, 3:7]
        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        ## check contact forces 
        # for i in range(self.num_bodies):
        #     norm_value = torch.norm(self.contact_forces[0, i, :], dim=-1)
        #     print(f"Index {i}: {self.body_names[i]} {norm_value}")


        # 지형 높이 정보 수집 → base 기준, foot 기준 → 평균 저장
        self.measured_heights, self.measured_legs_heights = self.get_heights()
        self.mean_measured_heights = torch.mean(self.measured_heights, dim=-1)
        self.mean_measured_legs_heights = torch.mean(self.measured_legs_heights, dim=-1)
        # self.min_measured_legs_heights = torch.min(self.measured_legs_heights, dim=-1).values
        # self.mean_measured_legs_heights = torch.mean(self.measured_legs_heights, dim=-1)
        
        # print("===========================================================")
        # # print("self.mean_measured_heights: ", self.mean_measured_heights[self.observe_envs])
        # print("self.mlh size : ", self.measured_legs_heights.size())
        # print("self.measured_legs_heights: ",self.measured_legs_heights[self.observe_envs])
        # print("self.mean_measured_legs_heights: ", self.mean_measured_legs_heights[self.observe_envs])
        # print("===========================================================")


        # compute observations, rewards, resets, ...
        self.reset_buf[:] = self.check_termination()
        self.compute_reward()     

        # 종료 환경은 초기화(reset)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        
        # print("================================================")
        # print("progress_buf : ", self.progress_buf[self.observe_envs])
        # print("obs_t : ", self.obs_buf_t[self.observe_envs])
        # print("obs_buf : ", self.obs_buf[self.observe_envs])
        # print("================================================")
        self.plot_juggler()

        # 직전 timestep들의 기록 업데이트 → 관측에 시간차 정보 활용 가능
        self.last_actions2[:] = self.last_actions[:]
        self.last_actions =  self.actions[:]

        self.last_dof_vel2[:] = self.last_dof_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
        self.last_dof_acc[:] = self.dof_acc[:]
        self.last_cycle_t[:] = self.cycle_t[:]

        # for i in range(len(self.last_proprioceptive_bufs) - 1, 0, -1):
        #     self.last_proprioceptive_bufs[i][:] = self.last_proprioceptive_bufs[i - 1][:]

        # 첫 번째 버퍼에 현재 관측값 복사
        # self.last_proprioceptive_bufs[0][:] = self.obs_buf[:]
        
        # 시뮬레이터 뷰어의 카메라 갱신 (관측 환경이 바뀌면)
        if self.cam_mode != 0:        
            self.camera_update()

    # 시뮬레이션 환경에서 카메라 시점을 자동으로 갱신해주는 함수
    def camera_update(self):
        # 테스트 모드이고 그래픽 디바이스가 사용 가능한 경우에만 카메라 업데이트 수행
        if (self.test_mode) & (self.graphics_device_id != -1):
            offset = 1.5 #[m]
            # 현재 관찰 환경의 로봇 베이스 위치
            base_pos = self.root_states[self.observe_envs, :3]

            # 대각선 위에서 로봇을 바라보는 기본 뷰
            if self.cam_mode == 1:
                cam_pos = gymapi.Vec3(base_pos[0] + offset, base_pos[1] + offset, base_pos[2] + offset)
                cam_target = gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2])

            elif self.cam_mode == 2:
                # 1인칭 시점: 카메라 위치를 로봇 베이스 위치 + offset으로 설정
                cam_pos_relative = gymapi.Vec3(0, 0, offset/3)
                cam_pos_relative_tensor = torch.tensor([cam_pos_relative.x, cam_pos_relative.y, cam_pos_relative.z], device=self.device).unsqueeze(0)
                rotated_offset = my_quat_rotate(self.base_quat, cam_pos_relative_tensor)
                rotated_offset_numpy = rotated_offset.cpu().numpy().flatten()
                cam_pos = gymapi.Vec3(*(base_pos.cpu().numpy() + rotated_offset_numpy))
                
                cam_target_relative = gymapi.Vec3(2*offset, 0, 0)
                cam_target_relative_tensor = torch.tensor([cam_target_relative.x, cam_target_relative.y, cam_target_relative.z], device=self.device).unsqueeze(0)
                rotated_offset_target = my_quat_rotate(self.base_quat, cam_target_relative_tensor)
                rotated_offset_target_numpy = rotated_offset_target.cpu().numpy().flatten()
                cam_target = gymapi.Vec3(*(base_pos.cpu().numpy() + rotated_offset_target_numpy))

            elif self.cam_mode == 3:
                # 3인칭 시점: 로봇 뒤쪽 위에서 바라보는 시점
                cam_pos_relative = gymapi.Vec3(-2*offset, 0, offset)
                cam_pos_relative_tensor = torch.tensor([cam_pos_relative.x, cam_pos_relative.y, cam_pos_relative.z], device=self.device).unsqueeze(0)
                rotated_offset = my_quat_rotate(self.base_quat, cam_pos_relative_tensor)
                rotated_offset_numpy = rotated_offset.cpu().numpy().flatten()
                cam_pos = gymapi.Vec3(*(base_pos.cpu().numpy() + rotated_offset_numpy))
                
                cam_target_relative = gymapi.Vec3(2*offset, 0, 0)
                cam_target_relative_tensor = torch.tensor([cam_target_relative.x, 0, 0], device=self.device).unsqueeze(0)
                rotated_offset_target = my_quat_rotate(self.base_quat, cam_target_relative_tensor)
                rotated_offset_target_numpy = rotated_offset_target.cpu().numpy().flatten()
                cam_target = gymapi.Vec3(*(base_pos.cpu().numpy() + rotated_offset_target_numpy))

            # 카메라 부드러운 이동
            if self.smoothed_cam_pos is None:
                self.smoothed_cam_pos = cam_pos
            else:
                self.smoothed_cam_pos = gymapi.Vec3(
                    self.smoothed_cam_pos.x * (1 - self.smoothing_alpha) + cam_pos.x * self.smoothing_alpha,
                    self.smoothed_cam_pos.y * (1 - self.smoothing_alpha) + cam_pos.y * self.smoothing_alpha,
                    self.smoothed_cam_pos.z * (1 - self.smoothing_alpha) + cam_pos.z * self.smoothing_alpha
                )

            if self.smoothed_cam_target is None:
                self.smoothed_cam_target = cam_target
            else:
                self.smoothed_cam_target = gymapi.Vec3(
                    self.smoothed_cam_target.x * (1 - self.smoothing_alpha) + cam_target.x * self.smoothing_alpha,
                    self.smoothed_cam_target.y * (1 - self.smoothing_alpha) + cam_target.y * self.smoothing_alpha,
                    self.smoothed_cam_target.z * (1 - self.smoothing_alpha) + cam_target.z * self.smoothing_alpha
                )
            # 실제 카메라 업데이트
            self.gym.viewer_camera_look_at(self.viewer, None, self.smoothed_cam_pos, self.smoothed_cam_target)

            
    # 로봇의 모든 움직임 명령을 0으로 초기화(멈춤)시키는 함수
    def freeze(self):
        # x 방향 0
        self.commands_x[:] = torch.zeros(self.num_envs, device=self.device)
        # y 방향 0
        self.commands_y[:] = torch.zeros(self.num_envs, device=self.device)
        # yaw 회전 0
        self.commands_yaw[:] = torch.zeros(self.num_envs, device=self.device)

    # 시뮬레이터에서 현재 로봇의 상태를 업데이트하고 필요한 값들을 가굥하여 내부 변수로 저장하는 함수
    def refresh_state(self):
        # 로봇의 전체 위치 회전 속도 정보
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 충돌, 접촉 힘 정보
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # 개별 부위(발, 종아리)의 위치 속도 정보
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 로봇의 위치[x,y,z]
        self.base_pos = self.root_states[:, :3]
        # 로봇의 방향 회전 쿼터니언 [x,y,z,w]
        self.base_quat = self.root_states[:, 3:7]
        
        # 이전 속도
        self.last_base_vel = self.base_vel
        
        # 선속도, 각속도를 local 좌표계 기준으로 변환 (로봇 좌표계 기준)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        
        # 전체 속도를 하나의 텐서로 결합[vx, vy, vz, wx, wy, wz]
        self.base_vel = torch.cat([self.base_lin_vel, self.base_ang_vel], dim=1)

        # print("base_lin_vel : ", self.base_lin_vel[0])
        # print("base_ang_vel : ", self.base_ang_vel[0])
        # print("base_vel : ", self.base_vel[0])
        # print("last_base_vel : ", self.base_vel[0])

        noise = torch_rand_float(-0.05, 0.05, (self.num_envs,3), device=self.device).squeeze()
        # 중력 벡터를 local 프레임 기준으로 함
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec) 
        # print("self.projected_gravity : ", self.projected_gravity[0] )

        # 도메인 랜덤화 시 중력 벡터에 작은 노이즈 추가 일반화를 위해
        if self.randomize:
            self.projected_gravity += noise
        # print("self.projected_gravity noised : ", self.projected_gravity[0] )

    # 로봇의 주변 지형 높이를 감지하기 위해 일정한 패턴으로 포인트들을 생성하는 함수?
    # 이 포인트들은 보통 obs로 사용되어 로봇이 주변 지형을 인식할 수 있게 합니다 ->> 현재 그렇게 사용하고 있는지는 모르겠음
    def init_height_points(self):
        # 60cm x 25cm rectangle 
        # x,y 방향 격자를 만들기 위한 좌표값 설정
        # x 8개 , y 6개
        y = 0.05 * torch.tensor([-5, -3,-1,1, 3, 5], device=self.device, requires_grad=False)
        x = 0.05 * torch.tensor([-12, -9, -6, -3, 3, 6, 9, 12], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        # 전체 포인트 수 저장
        self.num_height_points = grid_x.numel()
        # 모든 환경 수에 대해 포인트 배열 생성
        # shape[num_envs, 48, 3], 각 포인트는 (x, y, z) 좌표를 가짐
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_leg_height_points(self):
        """각 다리 아래의 높이를 측정할 지점들을 초기화합니다.

        각 다리 밑의 작은 사각형 영역 (예: 10cm x 10cm) 내에 균등하게 분포된 점들을 생성합니다.
        이 점들은 지면과의 거리를 측정하거나 다리의 높이를 추정하는 데 사용될 수 있습니다.

        Returns:
            torch.Tensor: (num_envs, 4, num_leg_height_points, 3) 크기의 텐서.
                        각 환경, 각 다리 아래의 측정 지점들의 (x, y, z) 좌표를 담고 있습니다.
        """
        # 각 다리 아래의 작은 사각형 범위 (예: 10cm x 10cm)를 정의하기 위한 리스트
        leg_points = []
        # 각 다리의 중심 위치를 기준으로 하는 오프셋 (상대적인 위치)
        # FL: Front Left (앞 왼쪽), FR: Front Right (앞 오른쪽),
        # RL: Rear Left (뒤 왼쪽), RR: Rear Right (뒤 오른쪽)
        leg_offsets = [
            (-0. , 0.), # FL (2사분면) - 수정: 의미상 약간 더 자연스러운 위치
            ( 0. , 0.),  # FR (1사분면) - 수정: 의미상 약간 더 자연스러운 위치
            (-0. , 0.),# RL (3사분면) - 수정: 의미상 약간 더 자연스러운 위치
            ( 0. , 0.)  # RR (4사분면) - 수정: 의미상 약간 더 자연스러운 위치
        ]

        # 각 다리 밑의 격자(grid) 형태로 점들을 생성
        # -2, -1, 0, 1, 2를 0.025 (2.5cm)씩 곱하여 -5cm, -2.5cm, 0cm, 2.5cm, 5cm 간격의 x, y 좌표 생성
        leg_grid_x, leg_grid_y = torch.meshgrid(
            0.02 * torch.tensor([-5, -3, -1, 1, 3, 5], device=self.device, requires_grad=False),
            0.02 * torch.tensor([-5, -3, -1, 1, 3, 5], device=self.device, requires_grad=False)
        )
        # 생성된 x, y 좌표들을 평탄화(flatten)하여 각 점의 (x, y) 쌍을 만듦
        leg_points_grid = torch.stack([leg_grid_x.flatten(), leg_grid_y.flatten()], dim=-1) # (25, 2) 크기

        # 각 다리에 대해 생성된 격자 점들을 오프셋만큼 이동시키고 z 좌표를 0으로 설정
        for offset_x, offset_y in leg_offsets:
            # 각 다리의 중심 오프셋에 격자 점들의 x, y 좌표를 더하고 z 좌표 0을 추가
            leg_points.append(torch.tensor([offset_x, offset_y, 0.0], device=self.device) + torch.cat([leg_points_grid, torch.zeros_like(leg_points_grid[:, 0:1])], dim=-1))

        # 각 다리 아래의 생성된 점들의 개수를 저장 (여기서는 5x5 = 25개)
        self.num_leg_height_points = leg_points_grid.shape[0]
        # 생성된 각 다리의 점들 리스트를 텐서로 쌓음 (4개의 다리, 각 25개의 점, 각 점은 [x, y, z] 좌표)
        leg_points_tensor = torch.stack(leg_points, dim=0) # (4, 25, 3) 크기
        # 환경의 개수만큼 복사하고 차원을 추가하여 최종적인 형태 (num_envs, 4, 25, 3)를 만듦
        leg_points_tensor = leg_points_tensor.unsqueeze(0).repeat(self.num_envs, 1, 1, 1)

        return leg_points_tensor
    
    def init_leg_height_points_radial(self):
        """각 다리 아래의 높이를 측정할 지점들을 초기화합니다 (방사형)."""
        leg_points = []
        leg_offsets = [
            (-0. , 0.),
            ( 0. , 0.),
            (-0. , 0.),
            ( 0. , 0.)
        ]

        num_radii = 2
        num_angles = 12
        radii = torch.tensor([0.025, 0.05], device=self.device, requires_grad=False) # 2.5cm, 5cm
        angles = torch.linspace(0, 2 * torch.pi, num_angles, device=self.device, requires_grad=False)

        points_local = torch.zeros((num_radii * num_angles + 1, 3), device=self.device)
        points_local[0] = torch.tensor([0.0, 0.0, 0.0], device=self.device) # 중심점

        for i in range(num_radii):
            for j in range(num_angles):
                angle = angles[j]
                radius = radii[i]
                x = radius * torch.cos(angle)
                y = radius * torch.sin(angle)
                points_local[i * num_angles + j + 1] = torch.tensor([x, y, 0.0], device=self.device)

        self.num_leg_height_points = points_local.shape[0]

        for offset_x, offset_y in leg_offsets:
            leg_points.append(torch.tensor([offset_x, offset_y, 0.0], device=self.device) + points_local)

        leg_points_tensor = torch.stack(leg_points, dim=0)
        leg_points_tensor = leg_points_tensor.unsqueeze(0).repeat(self.num_envs, 1, 1, 1)

        return leg_points_tensor
    
    def init_leg_height_points_sector(self):
        """각 다리 아래의 높이를 측정할 지점들을 초기화합니다 (부채꼴 모양, +x 방향, 10개 포인트)."""
        leg_points = []
        leg_offsets = [
            (-0. , 0.),
            ( 0. , 0.),
            (-0. , 0.),
            ( 0. , 0.)
        ]

        num_points = 15      # 중심점을 제외한 부채꼴 내부 포인트 수 (총 10개)
        min_radius = 0.05   # 최소 반지름
        max_radius = 0.20   # 최대 반지름
        angle_extent = torch.pi * 3 / 4  # 부채꼴의 각도 범위 (90도)
        angle_offset = 0.0            # 부채꼴의 시작 각도 (현재 +x 방향)

        points_local_list = []
        points_local_list.append(torch.tensor([0.0, 0.0, 0.0], device=self.device)) # 중심점

        # 포인트 수를 기반으로 반지름 및 각도 간격 조정
        if num_points > 0:
            # 대략적인 반지름 및 각도 분할 수 계산
            num_radii_approx = 3
            num_angles_approx = (num_points + num_radii_approx - 1) // num_radii_approx

            radii = torch.linspace(min_radius, max_radius, num_radii_approx, device=self.device)
            angles = torch.linspace(-angle_extent / 2 + angle_offset, angle_extent / 2 + angle_offset, num_angles_approx, device=self.device)

            count = 0
            for i in range(num_radii_approx):
                radius = radii[i]
                for j in range(num_angles_approx):
                    if count < num_points:
                        angle = angles[j]
                        x = radius * torch.cos(angle)
                        y = radius * torch.sin(angle)
                        points_local_list.append(torch.tensor([x, y, 0.0], device=self.device))
                        count += 1
                    else:
                        break

        points_local = torch.stack(points_local_list, dim=0)
        self.num_leg_height_points = points_local.shape[0]

        for offset_x, offset_y in leg_offsets:
            leg_points.append(torch.tensor([offset_x, offset_y, 0.0], device=self.device) + points_local)

        leg_points_tensor = torch.stack(leg_points, dim=0)
        leg_points_tensor = leg_points_tensor.unsqueeze(0).repeat(self.num_envs, 1, 1, 1)

        return leg_points_tensor
    
    # 로봇이 위치한 지형의 높이를 측정하는 함수
    def get_heights(self, env_ids=None):
        # 평평한 지형이면 높이는 모두 0
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False), torch.zeros(self.num_envs, 4, self.num_leg_height_points, device=self.device, requires_grad=False)

        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids is not None:
            base_quat = self.base_quat[env_ids]
            root_states = self.root_states[env_ids]
            height_points = self.height_points[env_ids]
            leg_height_points = self.leg_height_points[env_ids]
            foot_quat = self.rigid_body_rot[env_ids][:, [4, 8, 12, 16], :]
            foot_states = self.rigid_body_pos[env_ids][:, [4, 8, 12, 16], :]
        else:
            base_quat = self.base_quat
            root_states = self.root_states
            height_points = self.height_points
            leg_height_points = self.leg_height_points
            foot_quat = self.rigid_body_rot[:, [4, 8, 12, 16], :]
            foot_states = self.rigid_body_pos[:, [4, 8, 12, 16], :]

        # 기존의 넓은 범위 높이 측정 (유지)
        points = quat_apply_yaw(base_quat.repeat(1, self.num_height_points), height_points) + (root_states[:, :3]).unsqueeze(1)
        world_x = points[:, :, 0] # 각 점의 월드 좌표계 x 좌표
        world_y = points[:, :, 1]  # 각 점의 월드 좌표계 y 좌표
        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)
        measured_heights = heights.view(root_states.shape[0], -1) * self.terrain.vertical_scale

        # 지면 기울기 계산(base 하단 삼각형 기준)
        # 지형 기울기나 자세 판단 등에 사용될 수 있음
        #p1, p2, p3 정의
        world_x -= self.terrain.border_size
        world_y -= self.terrain.border_size
        p1 = torch.stack([world_x[:, 0], world_y[:, 0], measured_heights[:, 0]], dim=-1)
        p2 = torch.stack([world_x[:, 6], world_y[:, 6], measured_heights[:, 6]], dim=-1)
        p3 = torch.stack([world_x[:, 11], world_y[:, 11], measured_heights[:, 11]], dim=-1)

        # print("p1  : ", p1)
        # print("p2  : ", p2)
        # print("p3  : ", p3)
        v1 = p2 - p1
        v2 = p3 - p1

        # 두 벡터의 외적 계산
        normal = torch.cross(v1, v2)

        # 법선 벡터 정규화
        normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True)

        
        
        # 각 발(Foot) 아래 높이 측정 (TIP 기준)
        num_envs = root_states.shape[0]
        num_legs = 4

        # leg_height_points (로컬 좌표) - 이미 (num_envs, 4, 25, 3) 크기임
        foot_points_local = leg_height_points

        # 발의 회전 적용 (전체 쿼터니언 사용)
        foot_points_rotated = quat_rotate(foot_quat.unsqueeze(2).repeat(1, 1, self.num_leg_height_points, 1).view(-1, 4),
                                          foot_points_local.view(-1, 3)).view(num_envs, num_legs, self.num_leg_height_points, 3)

        # 발의 위치 적용 (월드 좌표계)
        foot_points_world = foot_points_rotated + foot_states.unsqueeze(2).repeat(1, 1, self.num_leg_height_points, 1)

        # 월드 좌표를 terrain 맵 좌표로 변환
        foot_points_world += self.terrain.border_size
        foot_points_world_scaled = (foot_points_world / self.terrain.horizontal_scale).long()

        # terrain 맵 좌표를 사용하여 높이 샘플링
        fx = foot_points_world_scaled[:, :, :, 0].view(-1)
        fy = foot_points_world_scaled[:, :, :, 1].view(-1)

        fx = torch.clip(fx, 0, self.height_samples.shape[0] - 2)
        fy = torch.clip(fy, 0, self.height_samples.shape[1] - 2)

        foot_heights1 = self.height_samples[fx, fy]
        foot_heights2 = self.height_samples[fx + 1, fy + 1]
        foot_heights = torch.min(foot_heights1, foot_heights2)
        measured_foot_heights = foot_heights.view(num_envs, num_legs, self.num_leg_height_points) * self.terrain.vertical_scale


       # ========================================= Base 시각화 =========================================
        if self.test_mode:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom_base = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 0, 1)) # base 측정 지점 색상 (파란색)

            for i in range(self.num_envs):
                # Base 시각화 (기존 코드)
                base_position = self.root_states[i, :3]
                base_rotation = self.base_quat[i]
                height_points_local_base = self.height_points[i].cpu().numpy() # base의 로컬 측정 포인트
                measured_heights_cpu_base = measured_heights[i].cpu().numpy() # base에서 측정된 높이

                for j in range(height_points_local_base.shape[0]):
                    local_point = torch.tensor(height_points_local_base[j], device=self.device)
                    world_point = quat_apply_yaw(base_rotation, local_point) + base_position
                    z = measured_heights_cpu_base[j]

                    sphere_pose = gymapi.Transform(gymapi.Vec3(world_point[0].item(), world_point[1].item(), z.item()), r=None)
                    gymutil.draw_lines(sphere_geom_base, self.gym, self.viewer, self.envs[i], sphere_pose)




    # ========================================= Foot 시각화 =========================================
    # if self.test_mode:
    #     self.gym.clear_lines(self.viewer)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)
    #     sphere_geom_foot = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0)) # 발 측정 지점 색상 (빨간색)

    #     for i in range(self.num_envs):
    #         # Foot 시각화
    #         foot_base_positions = self.rigid_body_pos[i, [4, 8, 12, 16], :3]
    #         foot_rotations = self.rigid_body_rot[i, [4, 8, 12, 16], :]
    #         leg_height_points_local = self.leg_height_points[i].cpu().numpy()
    #         measured_foot_heights_cpu = measured_foot_heights[i].cpu().numpy()

    #         for k in range(num_legs):
    #             base_pos = foot_base_positions[k]
    #             rotation = foot_rotations[k]
    #             heights = measured_foot_heights_cpu[k]
    #             local_points = leg_height_points_local[k]

    #             for j in range(local_points.shape[0]):
    #                 local_point = torch.tensor(local_points[j], device=self.device)
    #                 world_point = quat_apply_yaw(rotation, local_point) + base_pos
    #                 z = heights[j]

    #                 sphere_pose = gymapi.Transform(gymapi.Vec3(world_point[0].item(), world_point[1].item(), z.item()), r=None)
    #                 gymutil.draw_lines(sphere_geom_foot, self.gym, self.viewer, self.envs[i], sphere_pose)
    # ========================================= Foot 시각화 =========================================

        return measured_heights, measured_foot_heights

        # num_envs = root_states.shape[0]
        # num_legs = 4

        # # leg_height_points (로컬 좌표, 부채꼴 형태)
        # foot_points_local = leg_height_points

        # # 선형 속도 벡터로부터 방향 각도 계산
        # velocity_angles = torch.atan2(self.commands_y, self.commands_x)
        # zeros = torch.zeros_like(velocity_angles)
        # velocity_rotations_yaw_only = quat_from_euler_xyz(zeros, zeros, velocity_angles)
        # velocity_rotations = velocity_rotations_yaw_only.unsqueeze(1).repeat(1, num_legs, 1) # (num_envs, 4, 4)

        # # Yaw 커맨드 값
        # yaw_command = self.commands_yaw if hasattr(self, 'commands_yaw') else torch.zeros_like(velocity_angles)

        # # 선형 속도 크기 계산
        # linear_velocity_magnitude = torch.sqrt(self.commands_x**2 + self.commands_y**2)

        # # Yaw 회전 오프셋 계산 및 적용 조건
        # front_yaw_offset = yaw_command * 1.
        # rear_yaw_offset = -yaw_command * 1.
        # yaw_offsets = torch.stack([front_yaw_offset, front_yaw_offset , rear_yaw_offset,  rear_yaw_offset], dim=1) # [num_envs, num_legs]
        # yaw_rotations = quat_from_euler_xyz(zeros.unsqueeze(1).repeat(1, num_legs), zeros.unsqueeze(1).repeat(1, num_legs), yaw_offsets) # [num_envs, num_legs, 4]

        # combined_quat = torch.zeros(num_envs, num_legs, 4, device=self.device)
        # identity_quat = torch.tensor([0., 0., 0., 1.], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_envs, num_legs, 1)

        # for i in range(num_envs):
        #     if linear_velocity_magnitude[i] > 1e-2: # 임계값 설정 (조정 필요)
        #         # 선형 속도가 충분히 크면 속도 방향 사용
        #         combined_quat[i] = quat_mul(velocity_rotations[i].view(-1, 4), foot_quat[i].view(-1, 4)).view(num_legs, 4)
        #     else:
        #         # 선형 속도가 작으면 Yaw 커맨드 적용 (velocity_rotations 영향 없앰)
        #         combined_quat[i] = quat_mul(yaw_rotations[i].view(-1, 4), foot_quat[i].view(-1, 4)).view(num_legs, 4)

        # # 부채꼴 점들을 발의 위치로 이동시키기 전에, 결합된 회전 적용
        # foot_points_rotated = quat_rotate(combined_quat.unsqueeze(2).repeat(1, 1, self.num_leg_height_points, 1).view(-1, 4),
        #                                 foot_points_local.view(-1, 3)).view(num_envs, num_legs, self.num_leg_height_points, 3)


        # # 발의 위치 적용 (월드 좌표계)
        # foot_points_world = foot_points_rotated + foot_states.unsqueeze(2).repeat(1, 1, self.num_leg_height_points, 1)

        # # 월드 좌표를 terrain 맵 좌표로 변환
        # foot_points_world += self.terrain.border_size
        # foot_points_world_scaled = (foot_points_world / self.terrain.horizontal_scale).long()

        # # terrain 맵 좌표를 사용하여 높이 샘플링
        # fx = foot_points_world_scaled[:, :, :, 0].view(-1)
        # fy = foot_points_world_scaled[:, :, :, 1].view(-1)

        # fx = torch.clip(fx, 0, self.height_samples.shape[0] - 2)
        # fy = torch.clip(fy, 0, self.height_samples.shape[1] - 2)

        # foot_heights1 = self.height_samples[fx, fy]
        # foot_heights2 = self.height_samples[fx + 1, fy + 1]
        # foot_heights = torch.min(foot_heights1, foot_heights2)
        # measured_foot_heights = foot_heights.view(num_envs, num_legs, self.num_leg_height_points) * self.terrain.vertical_scale

        # # 시각화 (test_mode 활성화 시)
        # if self.test_mode:
        #     self.gym.clear_lines(self.viewer)
        #     self.gym.refresh_rigid_body_state_tensor(self.sim)
        #     sphere_geom_foot = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(1, 0, 0))
        #     for i in range(self.num_envs):
        #         foot_base_positions = self.rigid_body_pos[i, [4, 8, 12, 16], :3]
        #         combined_rotation = combined_quat[i]
        #         leg_height_points_local = self.leg_height_points[i].cpu().numpy()
        #         measured_foot_heights_cpu = measured_foot_heights[i].cpu().numpy()
 
        #         for k in range(num_legs):
        #             base_pos = foot_base_positions[k]
        #             rotation = combined_rotation[k]
        #             heights = measured_foot_heights_cpu[k]
        #             local_points = leg_height_points_local[k]

        #             for j in range(local_points.shape[0]):
        #                 local_point = torch.tensor(local_points[j], device=self.device)
        #                 world_point = quat_apply_yaw(rotation, local_point) + base_pos
        #                 z = heights[j]

        #                 sphere_pose = gymapi.Transform(gymapi.Vec3(world_point[0].item(), world_point[1].item(), z.item()), r=None)
        #                 gymutil.draw_lines(sphere_geom_foot, self.gym, self.viewer, self.envs[i], sphere_pose)

        # return measured_heights, measured_foot_heights

    def normal_vector_to_quaternion(self, normal_vector):
        """법선 벡터를 쿼터니언으로 변환합니다."""
        # z축을 법선 벡터로 정렬하는 회전 행렬 생성
        z_axis = torch.tensor([0, 0, 1.0], device=normal_vector.device)
        # normal_vector의 차원에 맞게 z_axis 반복
        z_axis = z_axis.repeat(normal_vector.shape[0], 1)
        rotation_axis = torch.cross(z_axis, normal_vector)
        rotation_angle = torch.arccos(torch.sum(z_axis * normal_vector, dim=1) / (torch.norm(z_axis, dim=1) * torch.norm(normal_vector, dim=1)))

        # 회전 축이 정의되지 않은 경우(평행한 경우) 처리
        rotation_quaternion = torch.zeros((normal_vector.shape[0], 4), device=normal_vector.device)
        non_zero_axis = torch.linalg.norm(rotation_axis, dim=1) > 1e-6
        if torch.any(non_zero_axis):
            rotation_quaternion[non_zero_axis] = torch.tensor(Rotation.from_rotvec((rotation_axis[non_zero_axis] * rotation_angle[non_zero_axis].unsqueeze(1)).cpu().numpy()).as_quat(), device=normal_vector.device)
        rotation_quaternion[~non_zero_axis] = torch.tensor([0, 0, 0, 1.0], device=normal_vector.device) # 회전 없음

        return rotation_quaternion



# terrain generator
# gym 기반 강화학습 환경의 지형 생성기
# 로봇이 다양한 지형에서 학습할 수 있도록 random or curriculum 방식으로 높이맵 기반 지형을 생성
from isaacgym.terrain_utils import *

# 다양한 환경에 대해 높이 맵(Heightfield) 기반 지형 생성
# 생성된 지형을 trimesh로 변환 (시뮬레이터 사용용)
# 각 환경의 지형 기준 z 높이(origin z)를 계산해 저장
class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1     # 한 픽셀(해상도)의 실제 거리[m]
        self.vertical_scale = 0.005     # 픽셀 1단계의 높이 변화량[m]
        self.border_size = 20           # 지형 외곽 여유 공간
        self.num_per_env = 2            # 하나의 지형에 몇 개의 로봇 배치할지
        self.env_length = cfg["mapLength"]      #하나의 환경 크기 [m]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg["terrainProportions"]))]   # 각 terrain 종류의 비율 누적합

        # 전체 지형을 (행 x 열)로 구성
        # 각 지형당 할당되는 로봇 수 재계산
        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)

        # 각 지형 블록의 중심좌표(x, y, z)를 저장할 배열
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        # 지형 하나가 몇 픽셀로 구성되는지 계산
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        # 전체 지형 이밎의 가로/세로 픽셀 수 계산(여유 공간 포함)
        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border
        
        # 전체 지형 높이맵 초기화
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        # 커리큘럼 기반 지형 구성인지 여부에 따라 다른 생성 함수 호출
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
    
    # 다양한 지형에서 학습하도록 슬로프, 계단, 장애무등의 조합을 자동으로 배치하는 구조
    def randomized_terrain(self):
        # num_maps 개의 개별 지형을 반복하면서 무작위로 지형 타입을 결정,  해당 지형을 heightfield에 할당하며 각 지형의 중심위치(env_origin)를 계산하여 저장
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            # 해당 지형이 heightfield 맵에서 차지하는 픽셀 범위(x,y축) 계산
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            # 지형 타입을 무작위로 고르기 위한 확률값
            choice = np.random.uniform(0, 1)
            # 경사면
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.25, -0.15, 0, 0.15, 0.25]))
                    # random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.02, downsampled_scale=0.25)
                    # random_uniform_terrain_with_flat_start(terrain, min_height=-0.05, max_height=0.05, step=0.02, downsampled_scale=0.25,flat_start_size=1.)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.25, -0.15, 0, 0.15, 0.25]))
            # 계산지형
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.03, 0.03])
                pyramid_stairs_terrain(terrain, step_width=0.1, step_height=step_height, platform_size=3.)
            # 불연속 장애물
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            # 해당 지형의 중심 위치(x, y 좌표)를 실제 거리 기준으로 계산
            env_origin_x = (i + 0.5) * self.env_length 
            env_origin_y = (j + 0.5) * self.env_width

            # 지형의 중앙 인근 부분의 최대 높이를 구해서 z 좌표 계산 (로봇이 떠 있지 않도록 초기 높이 조정 목적)
            x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / self.horizontal_scale)

            # 해당 지형의 중심 위치를 저장
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    # 점진적으로 난이도가 증가하는 지형을 생성
    # 지형을 "난이도(level)"와 "지형 유형(terrain type)"에 따라 격자 형태로 배치
    # 난이도가 점진적으로 올라가며, 여러 유형의 지형이 혼합됨
    # self.height_field_raw에 지형의 height map을 기록
    # self.env_origins에 각 지형 환경의 중심 위치를 저장
    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                    width=self.width_per_env_pixels,
                                    length=self.width_per_env_pixels,
                                    vertical_scale=self.vertical_scale,
                                    horizontal_scale=self.horizontal_scale)

                # 난이도 및 유형 계산
                difficulty = i / num_levels    # 현재 행 번호에 따라 난이도 비율
                choice = j / num_terrains      # 열 번호를 기반으로 terrain 종류를 결정할 값

                # slope = 0.025 + difficulty * 0.25
                # step_height = 0.025 + 0.025 * difficulty
                # discrete_obstacles_height = 0.025 + difficulty * 0.025
                # stepping_stones_size = 2 - 1.9* difficulty

                # 난이도가 높을수록 더 험한 경사, 더 큰 장애물, 좁은 스톤 등 설정
                slope = 0.125 + difficulty * 0.25
                step_height = 0.025 + 0.075 * difficulty
                discrete_obstacles_height = 0.05 + difficulty * 0.1
                stepping_stones_size = 2 - 1.8* difficulty

                # slope = 0.025 + difficulty * 0.05
                # step_height = 0.025 + 0.05 * difficulty
                # discrete_obstacles_height = 0.025 + difficulty * 0.01
                # stepping_stones_size = 2 - 1.8* difficulty


                # 지형 생성 조건
                # 경사지형
                if choice < self.proportions[0]:
                    if choice < 0.1 :
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=1.5)
                
                
                elif choice < self.proportions[1]:
                    if choice < 0.3 :
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=1.5)
                    # random_uniform_terrain_with_flat_start(terrain, min_height=-0.1, max_height=0.1, step=0.02, downsampled_scale=0.4,flat_start_size=1.)
                
                
                elif choice < self.proportions[2]:
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.02, downsampled_scale=0.275)
                    # random_uniform_terrain_with_flat_start(terrain, min_height=-0.1, max_height=0.1, step=0.02, downsampled_scale=0.4,flat_start_size=1.)
                
                #장애물
                elif choice < self.proportions[3]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 0.5, 1., 100, platform_size=3.)
                
                #계단
                elif choice < self.proportions[4]:
                    if choice<0.88:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.35, step_height=step_height, platform_size=3.)
                    # random_uniform_terrain_with_flat_start(terrain, min_height=-0.025, max_height=0.025, step=0.05, downsampled_scale=0.2,flat_start_size=2.)
                
                # else:
                #     stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.5) * self.env_length 
                env_origin_y = (j + 0.5) * self.env_width 
                x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

#PyTorch JIT 컴파일러를 사용하여 성능을 최적화한 두 개의 함수입니다. 각각 쿼터니언의 Yaw만 적용, 그리고 각도를 -π~π 범위로 래핑(wrapping) 하는 기능
@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles