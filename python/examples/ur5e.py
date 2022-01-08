from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random

def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats

# acquire gym interface
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0/60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = True

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.001
sim_params.physx.rest_offset = 0.0
sim_params.physx.friction_offset_threshold = 0.001
sim_params.physx.friction_correlation_distance = 0.0005

# set torch device
device = 'cuda'

# create sim
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# set root directory for assets
asset_root = "../../assets"

# create the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
# plane_params.distance = 0
# plane_params.static_friction = 1
# plane_params.dynamic_friction = 1
# plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# create table asset and pose
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

# create box asset
box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
box_pose = gymapi.Transform()

# create UR5e with 2F85 gripper asset and pose
ur5e_2f85_asset_file = "urdf/ur5_robotiq85_gripper.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
ur5e_2f85_asset = gym.load_asset(sim, asset_root, ur5e_2f85_asset_file, asset_options)
ur5e_2f85_pose = gymapi.Transform()
ur5e_2f85_pose.p = gymapi.Vec3(0, 0, 0)

# configure UR5e dofs
ur5e_dof_props = gym.get_asset_dof_properties(ur5e_2f85_asset)
ur5e_lower_limits = ur5e_dof_props["lower"]
ur5e_upper_limits = ur5e_dof_props["upper"]
# ur5e_ranges = ur5e_upper_limits - ur5e_lower_limits
ur5e_mids = 0.3 * (ur5e_upper_limits + ur5e_lower_limits)

# use position drive for all dofs
ur5e_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
ur5e_dof_props["stiffness"][:6].fill(400.0)
ur5e_dof_props["damping"][:6].fill(40.0)
# grippers
ur5e_dof_props["stiffness"][6:].fill(800.0)
ur5e_dof_props["damping"][6:].fill(40.0)

# default dof position targets
ur5e_num_dofs = gym.get_asset_dof_count(ur5e_2f85_asset)
default_dof_pos = np.zeros(ur5e_num_dofs, dtype=np.float32)
# default_dof_pos[:6] = ur5e_upper_limits[:6]
# open grippers
# default_dof_pos[6:] = ur5e_upper_limits[6:]

# default dof states
default_dof_state = np.zeros(ur5e_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# get link index of robotiq 85 gripper base link, which we will use as end effector
gripper_link_dict = gym.get_asset_rigid_body_dict(ur5e_2f85_asset)
gripper_base_link_index = gripper_link_dict["robotiq_coupler"]

# configure env grid
num_envs = 64
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs = []
box_idxs = []
gripper_base_idxs = []
init_pos_list = []
init_rot_list = []

# create environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add box randomly for each environment
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    box_pose.p.z = table_dims.z + (0.5 * box_size)
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)

    # get gloabal index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    # add UR5e
    ur5e_2f85_handle = gym.create_actor(env, ur5e_2f85_asset, ur5e_2f85_pose, "ur5e", i, 2)

    # set dof properties of UR5e
    gym.set_actor_dof_properties(env, ur5e_2f85_handle, ur5e_dof_props)

    # set initial dof states of UR5e
    gym.set_actor_dof_states(env, ur5e_2f85_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets of UR5e
    gym.set_actor_dof_position_targets(env, ur5e_2f85_handle, default_dof_pos)

    # get gripper effector initial pose
    gripper_base_handle = gym.find_actor_rigid_body_handle(env, ur5e_2f85_handle, "robotiq_coupler")
    gripper_base_pose = gym.get_rigid_transform(env, gripper_base_handle)
    init_pos_list.append([gripper_base_pose.p.x, gripper_base_pose.p.y, gripper_base_pose.p.z])
    init_rot_list.append([gripper_base_pose.r.x, gripper_base_pose.r.y, gripper_base_pose.r.z, gripper_base_pose.r.w])

    # get global index of gripper base in rigid body state tensor
    gripper_base_idx = gym.find_actor_rigid_body_index(env, ur5e_2f85_handle, "robotiq_coupler", gymapi.DOMAIN_SIM)
    gripper_base_idxs.append(gripper_base_idx)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

#### Prepare Tensors ####
gym.prepare_sim(sim)

# initial effector position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# gripper orientation for grasping
down_q = torch.stack(num_envs * [torch.Tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downward axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# ur5e + robotiq gripper has shape (num_envs, 20, 6, 12)
_jacobian = gym.acquire_jacobian_tensor(sim, "ur5e")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to gripper end effector
j_eef = jacobian[:, gripper_base_link_index - 1, :]

# #get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 12, 1)

# Create a tensor noting whether the gripper should return to the initial position
gripper_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# simulation loop
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]

    gripper_base_pos = rb_states[gripper_base_idxs, :3]
    gripper_base_rot = rb_states[gripper_base_idxs, 3:7]

    to_box = box_pos - gripper_base_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # how far the hand should be from box for grasping
    grasp_offset = 0.12

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 10] + dof_pos[:, 11]
    gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * box_size)
    # print(gripped)

    yaw_q = cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = quat_axis(yaw_q, 0)
    gripper_yaw_dir = quat_axis(gripper_base_rot, 0)
    yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), gripper_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the gripper to start moving to the box
    to_init = init_pos - gripper_base_pos
    init_dist = torch.norm(to_init, dim=-1)
    gripper_restart = (gripper_restart & (init_dist > 0.02)).squeeze(-1)
    return_to_start = (gripper_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if gripper is above box, descend to grasp rest_offset
    # otherwise, seek a position above the box
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 2)).squeeze(-1)
    grasp_pos = box_pos.clone()
    grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 1.5)
    # print(above_box)

    # compute goal position and orientation
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

    # compute position and orientation error
    pos_err = goal_pos - gripper_base_pos
    orn_err = orientation_error(goal_rot, gripper_base_rot)
    dpose = torch.cat([pos_err,  orn_err], -1).unsqueeze(-1)

    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    d = 0.05 # damping term
    lmbda = torch.eye(6).to(device) * (d ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 12, 1)

    # update position targets
    pos_target = dof_pos + u

    # gripper actions depend on distance between gripper base and box
    close_gripper = (box_dist < grasp_offset + 0.03) | gripped
    print(close_gripper)
    # always open the gripper above a certain height, dropping the box and restarting
    gripper_restart = gripper_restart | (box_pos[:, 2] > 0.6)
    # print(gripper_restart)
    keep_going = torch.logical_not(gripper_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    # print(close_gripper)
    grip_acts = torch.where(close_gripper, torch.Tensor([[1.57, 1.57]] * num_envs).to(device), torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
    pos_target[:, 10:12] = grip_acts.unsqueeze(-1)

    # set new position targets
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

    #update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
