from typing import Union

import numpy as np

from gymnasium_robotics.envs.robot_env import MujocoPyRobotEnv, MujocoRobotEnv
from gymnasium_robotics.utils import rotations

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def get_base_fetch_env(RobotEnvClass: Union[MujocoPyRobotEnv, MujocoRobotEnv]):
    """Factory function that returns a BaseFetchEnv class that inherits
    from MujocoPyRobotEnv or MujocoRobotEnv depending on the mujoco python bindings.
    """

    class BaseFetchEnv(RobotEnvClass):
        """Superclass for all Fetch environments."""

        def __init__(
            self,
            gripper_extra_height,
            block_gripper,
            has_object: bool,
            target_in_the_air,
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            reward_type,
            **kwargs
        ):
            """Initializes a new Fetch environment.

            Args:
                model_path (string): path to the environments XML file
                n_substeps (int): number of substeps the simulation runs on every call to step
                gripper_extra_height (float): additional height above the table when positioning the gripper
                block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
                has_object (boolean): whether or not the environment has an object
                target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
                target_offset (float or array with 3 elements): offset of the target
                obj_range (float): range of a uniform distribution for sampling initial object positions
                target_range (float): range of a uniform distribution for sampling a target
                distance_threshold (float): the threshold after which a goal is considered achieved
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            """

            self.gripper_extra_height = gripper_extra_height
            self.block_gripper = block_gripper
            self.has_object = has_object
            self.target_in_the_air = target_in_the_air
            self.target_offset = target_offset
            self.obj_range = obj_range
            self.target_range = target_range
            self.distance_threshold = distance_threshold
            self.reward_type = reward_type

            super().__init__(n_actions=4, **kwargs)

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, info):
            # Compute distance between goal and the achieved goal.
            d = goal_distance(achieved_goal, goal)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (4,)
            action = (
                action.copy()
            )  # ensure that we don't change the action outside of this scope
            pos_ctrl, gripper_ctrl = action[:3], action[3]

            pos_ctrl *= 0.05  # limit maximum change in position
            rot_ctrl = [
                1.0,
                0.0,
                1.0,
                0.0,
            ]  # fixed rotation of the end effector, expressed as a quaternion
            gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
            assert gripper_ctrl.shape == (2,)
            if self.block_gripper:
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

            return action

        def _get_obs(self):
            try:
                (
                    grip_pos,
                    object_pos,
                    object_rel_pos,
                    gripper_state,
                    object_rot,
                    object_velp,
                    object_velr,
                    grip_velp,
                    gripper_vel,
                ) = self.generate_mujoco_observations()
            except ValueError:
                return self._get_obs_many_object()

            full_goal = False
            if not self.has_object:
                achieved_goal = grip_pos.copy()
            elif full_goal is True:
                achieved_goal = np.concatenate([grip_pos.copy(), np.squeeze(object_pos.copy())])
            else:
                achieved_goal = np.squeeze(object_pos.copy())

            obs = np.concatenate(
                [
                    grip_pos,
                    object_pos.ravel(),
                    object_rel_pos.ravel(),
                    gripper_state,
                    object_rot.ravel(),
                    object_velp.ravel(),
                    object_velr.ravel(),
                    grip_velp,
                    gripper_vel,
                ]
            )

            return {
                "observation": obs.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy(),
            }
        
        def _get_obs_many_object(self): 
            '''
            Used in the case of many objects on table (the custom environment)
            Called from MujocoFetchRandomEnv
            '''
            object_positions = self.generate_mujoco_observations()
            desired_goal = self.goal.copy()
            obs = {'observation': None, 'achieved_goal': None, 'desired_goal': desired_goal}
            obs['observation'] = np.concatenate([position.ravel() for object, position in object_positions.items()])
            obs['achieved_goal'] = np.concatenate([position.ravel() for object, position in object_positions.items()])
            return obs

        def generate_mujoco_observations(self):

            raise NotImplementedError

        def _get_gripper_xpos(self):

            raise NotImplementedError

        def _sample_goal(self):
            if self.has_object:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
            return goal.copy()

        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)

    return BaseFetchEnv


class MujocoPyFetchEnv(get_base_fetch_env(MujocoPyRobotEnv)):
    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
            self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
            self.sim.forward()

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.sim, action)
        self._utils.mocap_set_action(self.sim, action)

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt

        robot_qpos, robot_qvel = self._utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos("object0")
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
            # velocities
            object_velp = self.sim.data.get_site_xvelp("object0") * dt
            object_velr = self.sim.data.get_site_xvelr("object0") * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )

    def _get_gripper_xpos(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        return self.sim.data.body_xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _viewer_setup(self):
        lookat = self._get_gripper_xpos()
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def fix_block_reset_position(self, fixed_block_position):
        self.fixed_block_pos = fixed_block_position

    def update_goal(self, updated_goal_position):
        self.goal = updated_goal_position

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        self.sim.forward()
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self._utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self.sim.data.get_site_xpos("robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos("object0")[2]


class MujocoFetchEnv(get_base_fetch_env(MujocoRobotEnv)):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)
        self.fixed_block_pos = None

    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )



    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        #print(self.goal)
        self.model.site_pos[site_id] = self.goal - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)


    def set_fixed_block_pos(self, fixed_pos):
        self.fixed_block_pos = fixed_pos

    def update_goal(self, updated_goal_position):
        self.goal = updated_goal_position

    def _reset_sim(self, fixed_reset_pos=None):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:

            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            if self.fixed_block_pos is not None:
                object_xpos = self.initial_gripper_xpos[:2] + self.fixed_block_pos
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]


OBJECT_NAMES = ['object0', 'object1', 'object2']
class MujocoFetchRandomEnv(get_base_fetch_env(MujocoRobotEnv)):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)
        self.fixed_block_pos = None

    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        if self.has_object:
            object_pos = {name: None for name in OBJECT_NAMES}
            for obj in OBJECT_NAMES:
                object_pos[obj] = self._utils.get_site_xpos(self.model, self.data, obj)
        return object_pos

        # # ORIGINAL CODE BELOW 
        # if self.has_object:
        #     object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        #     # rotations
        #     object_rot = rotations.mat2euler(
        #         self._utils.get_site_xmat(self.model, self.data, "object0")
        #     )
        #     # velocities
        #     object_velp = (
        #         self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
        #     )
        #     object_velr = (
        #         self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
        #     )
        #     # gripper state
        #     object_rel_pos = object_pos - grip_pos
        #     object_velp -= grip_velp
        # else:
        #     object_pos = (
        #         object_rot
        #     ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        # gripper_state = robot_qpos[-2:]

        # gripper_vel = (
        #     robot_qvel[-2:] * dt
        # )  # change to a scalar if the gripper is made symmetric

        # return (
        #     grip_pos,
        #     object_pos,
        #     object_rel_pos,
        #     gripper_state,
        #     object_rot,
        #     object_velp,
        #     object_velr,
        #     grip_velp,
        #     gripper_vel,
        # )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        #print(self.goal)
        self.model.site_pos[site_id] = self.goal - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)


    def set_fixed_block_pos(self, fixed_pos):
        self.fixed_block_pos = fixed_pos

    def update_goal(self, updated_goal_position):
        self.goal = updated_goal_position

    def _reset_sim(self, fixed_reset_pos=None):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:

            # sets object pos to be at the gripper pos and randomize until it is not near it
            # object_xpos = self.initial_gripper_xpos[:2]
            # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            #     object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
            #         -self.obj_range, self.obj_range, size=2
            #     )
            
            if self.fixed_block_pos is not None:
                '''If we fix the block position, set object_xpos to be our fixed position'''
                object_xpos = self.initial_gripper_xpos[:2] + self.fixed_block_pos
            

            for object in OBJECT_NAMES:
                object_qpos = self._utils.get_joint_qpos(
                    self.model, self.data, f"{object}:joint"
                )
                assert object_qpos.shape == (7,)
                posx = 1.2 + np.random.random() * 0.3
                posy = 0.5 + np.random.random() * 0.5
                while not 1.2 < posx < 1.5 or not 0.5 < posy < 1.0:
                    posx = 1.2 + np.random.random() * 0.3
                    posy = 0.5 + np.random.random() * 0.5
                object_qpos[:2] = [posx, posy]
                self._utils.set_joint_qpos(
                    self.model, self.data, f"{object}:joint", object_qpos
                )

        self._mujoco.mj_forward(self.model, self.data)
        return True
    
    def render(self):
        self._render_callback()
        if self.render_mode == 'human':
            array = self.mujoco_renderer.render('rgb_array')
            return self.mujoco_renderer.render(self.render_mode), array
        else:
            return self.mujoco_renderer.render(self.render_mode)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]
