import os
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.ur5e.ur5e as rbt
import robot_sim.end_effectors.gripper.robotiq140.robotiq140 as hnd
import robot_sim.robots.robot_interface as ri
from panda3d.core import CollisionNode, CollisionBox, Point3
import point_cloud_segmentation_sin as pcl_seg
class UR5EConveyorBelt(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="ur5e_conveyorbelt", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # base plate
        self.base_stand = jl.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     homeconf=np.zeros(4),
                                     name='base_stand')
        self.base_stand.jnts[1]['loc_pos'] = np.array([.9, -1.5, -0.06])
        self.base_stand.jnts[2]['loc_pos'] = np.array([0, 1.23, 0])
        self.base_stand.jnts[3]['loc_pos'] = np.array([0, 0, 0])
        self.base_stand.jnts[4]['loc_pos'] = np.array([-.9, .27, 0.06])
        self.base_stand.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur5e_base.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.base_stand.lnks[0]['rgba'] = [.35, .35, .35, 1]
        # self.base_stand.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "conveyor2.stl")
        # self.base_stand.lnks[1]['rgba'] = [.35, .55, .35, 1]
        # self.base_stand.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "camera_stand2.stl")
        # self.base_stand.lnks[2]['rgba'] = [.55, .55, .55, 1]
        # self.base_stand.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "cameras2.stl")
        # self.base_stand.lnks[3]['rgba'] = [.55, .55, .55, 1]
        # self.base_stand.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "platform2.stl")
        # self.base_stand.lnks[4]['rgba'] = [.35, .35, .9, 1]
        self.base_stand.reinitialize()
        # arm
        arm_homeconf = np.zeros(6)
        # arm_homeconf[0] = math.pi / 2
        arm_homeconf[1] = -math.pi * 2 / 3
        arm_homeconf[2] = math.pi / 3
        arm_homeconf[3] = -math.pi / 2
        arm_homeconf[4] = -math.pi / 2
        self.arm = rbt.UR5E(pos=self.base_stand.jnts[-1]['gl_posq'],
                            rotmat=self.base_stand.jnts[-1]['gl_rotmatq'],
                            homeconf=arm_homeconf,
                            name='arm', enable_cc=False)
        # gripper
        self.hnd = hnd.Robotiq140(pos=self.arm.jnts[-1]['gl_posq'],
                                  rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                  name='hnd_s', enable_cc=False)
        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.jaw_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-0.1, 0.0, 0.14 - 0.82),
                                              x=.35 + radius, y=.3 + radius, z=.14 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, -.3),
                                              x=.112 + radius, y=.112 + radius, z=.3 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    def enable_cc(self):
        # TODO when pose is changed, oih info goes wrong
        super().enable_cc()
        self.cc.add_cdlnks(self.base_stand, [0])
        self.cc.add_cdlnks(self.arm, [1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.lft_outer, [0, 1, 2, 3, 4])
        self.cc.add_cdlnks(self.hnd.rgt_outer, [1, 2, 3, 4])
        activelist = [#self.base_stand.lnks[0],
                      # self.base_stand.lnks[1],
                      # self.base_stand.lnks[2],
                      # self.base_stand.lnks[3],
                      # self.base_stand.lnks[4],
                      self.arm.lnks[1],
                      self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.lft_outer.lnks[0],
                      self.hnd.lft_outer.lnks[1],
                      self.hnd.lft_outer.lnks[2],
                      self.hnd.lft_outer.lnks[3],
                      self.hnd.lft_outer.lnks[4],
                      self.hnd.rgt_outer.lnks[1],
                      self.hnd.rgt_outer.lnks[2],
                      self.hnd.rgt_outer.lnks[3],
                      self.hnd.rgt_outer.lnks[4]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.base_stand.lnks[0],
                    # self.base_stand.lnks[1],
                    # self.base_stand.lnks[2],
                    # self.base_stand.lnks[3],
                    # self.base_stand.lnks[4],
                    self.arm.lnks[1]]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_outer.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.lft_outer.lnks[3],
                    self.hnd.lft_outer.lnks[4],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[3],
                    self.hnd.rgt_outer.lnks[4]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[2]]
        intolist = [self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft_outer.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.lft_outer.lnks[3],
                    self.hnd.lft_outer.lnks[4],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[3],
                    self.hnd.rgt_outer.lnks[4]]
        self.cc.set_cdpair(fromlist, intolist)
        fromlist = [self.arm.lnks[3]]
        intolist = [self.arm.lnks[6],
                    self.hnd.lft_outer.lnks[0],
                    self.hnd.lft_outer.lnks[1],
                    self.hnd.lft_outer.lnks[2],
                    self.hnd.lft_outer.lnks[3],
                    self.hnd.lft_outer.lnks[4],
                    self.hnd.rgt_outer.lnks[1],
                    self.hnd.rgt_outer.lnks[2],
                    self.hnd.rgt_outer.lnks[3],
                    self.hnd.rgt_outer.lnks[4]]
        self.cc.set_cdpair(fromlist, intolist)
        for oih_info in self.oih_infos:
            objcm = oih_info['collision_model']
            self.hold('arm', objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_stand.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_stand.jnts[-1]['gl_posq'], rotmat=self.base_stand.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fk(self, component_name='arm', jnt_values=np.zeros(6)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not supported!")

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self, hnd_name='hnd_s', jawwidth=0.0):
        self.hnd.jaw_to(jawwidth)

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.base_stand.lnks[0],
                    # self.base_stand.lnks[1],
                    # self.base_stand.lnks[2],
                    # self.base_stand.lnks[3],
                    # self.base_stand.lnks[4],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def release(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm7_shuidi_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.base_stand.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                       tcp_loc_pos=tcp_loc_pos,
                                       tcp_loc_rotmat=tcp_loc_rotmat,
                                       toggle_tcpcs=False,
                                       toggle_jntscs=toggle_jntscs,
                                       toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_shuidi_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.base_stand.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                                      tcp_loc_pos=tcp_loc_pos,
                                      tcp_loc_rotmat=tcp_loc_rotmat,
                                      toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
        self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.hnd.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel


import time
import random
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
from manipulation.pick_place_planner import PickPlacePlanner
base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, .0])


def check_collision(coord, sim=False):
    print("Checking Collision")

    gm.gen_frame().attach_to(base)
    component_name = 'arm'
    robot_s = UR5EConveyorBelt(enable_cc=True)
    inijnts = np.array([-75.62, -54.86, -142.27, 18.05, 76.53, -2.38]) / 180 * math.pi
    robot_s.fk(component_name, inijnts)
    robot_s.jaw_to(component_name, .14)
    robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=False).attach_to(base)
    # Check original robot pos and rotmat
    print(robot_s.arm.jnts[-1]['gl_posq'] + robot_s.arm.jnts[-1]['gl_rotmatq'].dot(robot_s.hnd.jaw_center_pos))
    print(robot_s.arm.jnts[-1]['gl_rotmatq'].dot(robot_s.hnd.jaw_center_rotmat))

    # Draw the object model
    # object = cm.gen_box([0.1, 0.1, 0.1])
    # object_copy = object.copy()
    # start_obj_pos = [-0.5, 0, 0.2]
    # start_obj_rotmat = np.identity(3)
    # object_copy.set_pos(start_obj_pos)
    # object_copy.set_rotmat(start_obj_rotmat)
    # object_copy.set_rgba([.9, .75, .35, .5])
    # object_copy.attach_to(base)

    front_wall = cm.CollisionModel('wrs/robot_sim/robots/ur5e_conveyorbelt/front_wall.stl')
    front_wall.set_rgba([1.0, 1.0, 1.0, .0])
    front_wall.attach_to(base)

    side_wall = cm.CollisionModel('wrs/robot_sim/robots/ur5e_conveyorbelt/side_wall.stl')
    side_wall.set_rgba([1.0, 1.0, 1.0, .0])
    side_wall.attach_to(base)

    conveyor = cm.CollisionModel('wrs/robot_sim/robots/ur5e_conveyorbelt/conveyor.stl', expand_radius=0.001)
    conveyor.set_rgba([.35, .55, .35, 1])
    conveyor.attach_to(base)

    # table = cm.CollisionModel('table.stl', expand_radius=0.0001)
    # table.set_rgba([.16, .16, .65, 1])
    # table.attach_to(base)   

    objects = []
    for i in range(len(coord)):
        posx = (coord[i][1] + coord[i][3]) / 2 - 0.8
        posy = (coord[i][2] + coord[i][0]) / 2 - 0.95
        posz = coord[i][4] / 2
        homomat = np.array([[1, 0, 0, posx],
                        [0, 1, 0, posy],
                        [0, 0, 1, 0.18],
                        [0, 0, 0, 1]]) # 3x1: bounding box center of mass
        l_x = coord[i][3] - coord[i][1]
        l_y = coord[i][2] - coord[i][0]
        l_z = coord[i][4]
        gm.gen_box([l_x, l_y, l_z], homomat, [0, 0, 1, 1]).attach_to(base) # length width hight # rgbd
        box1 = cm.gen_box([l_x, l_y, l_z], homomat, [0, 0, 1, 1])
        objects.append(box1)
    print("objects", objects)
    # Plan pick-and-place
    # planner = PickPlacePlanner(robot_s)
    # pos = [0, 0, 0] # grasp pose (relative to object)
    # rotmat = rm.rotmat_from_axangle([1, 1, 0], -math.pi/3*2)
    # gm.gen_frame(start_obj_pos, start_obj_rotmat).attach_to(base)
    # wid = 0.14
    # start_jaw_center_pos = start_obj_rotmat.dot(pos) + start_obj_pos
    # start_jaw_center_rotmat = start_obj_rotmat.dot(rotmat)
    # start_jaw_center_rotmat = rm.rotmat_from_axangle([1,0,0], math.pi/2)
    start_jaw_center_rotmat1 = np.array([[ 1,  0,  0],
                                        [ 0, -1,  0],
                                        [ 0,  0, -1]])
    start_jaw_center_rotmat2 = np.array([[ 0, -1,  0],
                                        [-1,  0,  0],
                                        [ 0,  0, -1]])
    start_jaw_center_rotmats = [start_jaw_center_rotmat1, start_jaw_center_rotmat2]

    # TODO: Camera to Robot Frame Transformation
    # start_jaw_center_pos = sampled points on midline for every gap of gripepr width
    # cam_pos = np.array([0, -0.06, 0.05])
    # cam_rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi * 1)
    # pos = np.array([-0.5, 0, 0.18])  
    # start_jaw_center_pos = cam_rotmat.dot(pos) + cam_pos
    # Midpoint of rectangle: ( (x1 + x2) / 2, (y1 + y2) / 2 )
    pos_x = (coord[0][1] + coord[0][3]) / 2 - 0.8
    pos_y = (coord[0][2] + coord[0][0]) / 2 - 0.95
    pos_z = coord[0][4]
    start_jaw_center_pos = np.array([pos_x, pos_y, pos_z])
    
    # print("From point_cloud_segmentation_sin: ", pcl_seg.getinfo())
    success = []
    for start_jaw_center_rotmat in start_jaw_center_rotmats:
        start = robot_s.ik(component_name, start_jaw_center_pos, start_jaw_center_rotmat, inijnts)
        if start is None:
            raise ValueError("Cannot generate the grasp!")
        robot_s.fk(component_name, start)
        robot_s.gen_meshmodel().attach_to(base)
        # if robot_s.is_collided([box1, box2]):
        #     # raise ValueError("Collided!")
        #     print("Collided!")
        # whether start exists && robot not collided
        success.append(int(not start is None) * int(not robot_s.is_collided(objects)))
    print("success: ", success)
    # base.run()
    
    # start_tcp_pos = robot_s.arm.jnts[-1]['gl_posq']
    # start_tcp_rotmat = robot_s.arm.jnts[-1]['gl_rotmatq']
    if sim:
        base.run()
    return not all(num == 0 for num in success) and len(success) > 0

### Planner Start
from copy import deepcopy
from heapq import heappush, heappop # min-heap

# Width of the gripper's tip
gWidth = 0.019 # m
# Maximum opening length
gOpen = 0.139
# Table height
table_h = 1

# Helper functions
def inInterval(num, interval):
    return num > interval[0] and num < interval[1]

def near(n1, n2):
    threshold = 0.01
    interval = [n2 - threshold, n2 + threshold]
    return inInterval(n1, interval)


class Box:
    def __init__(self, name, coordinates, length, width, height):
        self.name = name # number
        self.x0 = coordinates[0] # [x0, y0, x1, y1]
        self.y0 = coordinates[1]
        self.x1 = coordinates[2]
        self.y1 = coordinates[3]
        self.coordinates = coordinates
        self.length = length
        self.width = width
        self.height = height

        # For push: next to the midpoint on the side
        self.front = None
        # me
        # box
        self.back = None
        self.left = None
        self.right = None
        self.surroundings = []
        self.orientation = None # H/V

    
    def reset_neighbors(self):
        self.front = None
        self.back = None
        self.left = None
        self.right = None
        self.surroundings = []

    def update_surrounding(self, otherBoxes)->None:
        # Do this after all boxes' coordinates are recorded
        mid_f = ((self.x1 + self.x0) / 2, self.y1)
        mid_b = ((self.x1 + self.x0) / 2, self.y0)
        mid_l = (self.x0, (self.y1 + self.y0) / 2)
        mid_r = (self.x1, (self.y1 + self.y0) / 2)
        for box in otherBoxes:
            if inInterval(mid_f[0], [box.x0, box.x1]) and near(mid_f[1], box.y0):
                self.front = box
                self.surroundings.append(box)
            elif inInterval(mid_b[0], [box.x0, box.x1]) and near(mid_b[1], box.y1):
                self.back = box
                self.surroundings.append(box)
            elif near(mid_l[0], box.x1) and inInterval(mid_l[1], [box.y0, box.y1]):
                self.left = box
                self.surroundings.append(box)
            elif inInterval(mid_r[0], [box.x0, box.x1]) and inInterval(mid_r[1], [box.y0, box.y1]):
                self.right = box
                self.surroundings.append(box)

    
    def adjacent_box(self)->list:
        res = []
        if self.front:
            res.append(self.front)
        if self.back:
            res.append(self.back)
        if self.left:
            res.append(self.left)
        if self.right:
            res.append(self.right)
        return res

    def can_grasp(self, allBoxes):
        '''
        Can directly grasp the object.
        Requirements:
         - There exists a pair of parallel sides with no other bounding box overlaping for the gripper's width
         - The parallel sides are apart from each other for a distance < the gripper's maximum
        '''
        simulation_coord = [self.coordinates]
        heights = np.array([self.height])
        print("simulation_coord", simulation_coord)
        if self.front:
            heights = np.append(heights, self.front.height)
            simulation_coord = np.append(simulation_coord, [self.front.coordinates], axis=0)
        if self.back:
            heights = np.append(heights, self.back.height)
            simulation_coord = np.append(simulation_coord, [self.back.coordinates], axis=0)
        if self.left:
            heights = np.append(heights, self.left.height)
            simulation_coord = np.append(simulation_coord, [self.left.coordinates], axis=0)
        if self.right:
            heights = np.append(heights, self.right.height)
            simulation_coord = np.append(simulation_coord, [self.right.coordinates], axis=0)
        heights = heights.reshape(len(heights), 1)
        res = np.append(simulation_coord, heights, axis=1)
        print("neighbor_coord", res)
        
        return check_collision(res)


    def pushable_sides(self): # TODO： PUsh IS WRONG
        # Return pushable sides else None
        # Requirements: No obstacles around the midpoint next to the side
        # We do not consider insert-push (push from the side where there is an obstacle)
        res = []
        if self.front == None and self.back == None:
            res.append("front")
            res.append("back")
        if self.left == None and self.right == None:
            res.append("left")
            res.append("right")
        return res
    
    
    def can_push(self):
        res = self.pushable_sides()
        return True if len(res)!=0 else False


    def check_topple(self, side, otherBoxes):
        new_coord = np.append(self.coordinates, np.array(self.height))
        # print("new_coord", new_coord)
        if side == "front":
            new_coord[1] = self.y1 # y0
            new_coord[3] = self.y1 + self.height # y1
            new_coord[4] = self.width
        elif side == "back":
            new_coord[1] = self.y0 - self.height # y0
            new_coord[3] = self.y0
            new_coord[4] = self.width
        elif side == "left":
            new_coord[0] = self.x0 - self.height
            new_coord[2] = self.x1
            new_coord[4] = self.length
        elif side == "right":
            new_coord[0] = self.x1
            new_coord[2] = self.x1 + self.height
            new_coord[4] = self.length

        sim_coord = [new_coord]
        for otherbox in otherBoxes:
            line = np.append(otherbox.coordinates, np.array(otherbox.height))
            sim_coord = np.append(sim_coord, [line], axis=0)
        return check_collision(sim_coord)


    def toppleable_sides(self, allBoxes):
        res = []
        if self.height > self.width and self.height > self.length:
            # Whether there are other boxes blocking the falling motion of toppled box
            otherBoxes = allBoxes[:]
            for box_x in otherBoxes:
                if box_x.name == self.name:
                    otherBoxes.remove(box_x)
                    break
            # Check "topple to the front" etc
            for side in ["front", "back", "left", "right"]:
                if self.check_topple(side, otherBoxes):
                    res.append(side)
        return res


    def can_topple(self, allBoxes):
        res = self.toppleable_sides(allBoxes)
        return True if len(res) != 0 else False
    
    def best_skill(self, allBoxes)->int:
        # return cost of skill
        if self.can_grasp(allBoxes):
            return 1
        elif self.height > self.width and self.height > self.length and self.can_topple(allBoxes):
            # if the box is tall, assume we can only topple
            return 3
        elif self.can_push():
            return 2
        return 0

class State:
    def __init__(self, target, allBoxes, parent=None):
        self.gone_box = target # "wish you were gone box"
        self.allBoxes = allBoxes
        self.parent = parent
        self.removed = None # the box we actually remove in this state not adjacent to 
        self.skill = None

    def update_whole_environment(self):
        if self.allBoxes is not None:
            for box in self.allBoxes:
                otherBoxes = self.allBoxes[:]
                otherBoxes.remove(box)
                if len(otherBoxes) > 0:
                    box.reset_neighbors()
                    box.update_surrounding(otherBoxes)


class Solvers:
    def __init__(self, startState):
        self.startState = startState
        self.frontierList = []       # contains state



    def check_hypothesis(self, box1, box2, curState):
        """
        If when box2 is gone, can box1 be grasp/push-and-grasp/topple-and-grasp? If yes, return the skill to remove box1,
        otherwise return 0.
        """
        box1_name = box1.name
        box2_name = box2.name
        newState = deepcopy(curState) # also deepcopy boxes in the state
        newAllBoxes = curState.allBoxes[:]
        for box_x in newAllBoxes:
            if box_x.name == box2_name:
                newAllBoxes.remove(box_x)
                break
        newState.allBoxes = newAllBoxes
        newState.update_whole_environment()
        # print("newAllBoxes", len(newAllBoxes))
        for box_x in newState.allBoxes:
            if box_x.name == box1_name:
                return box_x.best_skill(newState.allBoxes)
        raise ValueError("Error: in check_hypothesis Box 1 was not found!")

    
    def trace_sol(self, state):
        def skill_translator(skill):
            if skill == 1:
                return "Direct Grasp"
            elif skill == 2:
                return "Push and Grasp"
            elif skill == 3:
                return "Topple and Grasp"
            else:
                return "No Skill"
            
        res = []
        while state:
            if state.removed:
                res.append((state.self.removed.name, skill_translator(state.skill)))
                state = state.parent
            else:
                res.append((state.gone_box.name, skill_translator(state.skill)))
                state = state.parent
        print("Found solution: ", res)
        return res
    

    def planner(self, target):
        """
        parent_state.parent = None; parent_state.gone_box = target;
        Add (depth=0, cost=0, hash(parent_state), parent_state) to the frontier (min-heap sorted by depth, to break tie sort by cost);
        i = 0;
        For (depth==i, cost1, id, state1) in the frontier, 
        For all adj_box in adjacent_box(state1.gone_box), if when adj_box is gone, state1.gone_box can be grasp/push-and-grasp/topple-and-grasp,
        state2.parent = state1; state2.gone_box = adj_box;
        If adj_box can also be grasp/push-and-grasp/topple-and-grasp,
            Go to trace_sol;
        Else
            cost2 = 2 if topplable, 3 if pushable, inf if none
            Add (depth=i+1, cost2, hash(state2), state2) to the frontier;
        Repeat Line 4 for i+=1;
        def trace_sol(state):
        res = []
        while state.parent:
        skill-to-remove-gone_box = grasp/push-and-grasp/topple-and-grasp
        res.append((state.gone_box, skill-to-remove-gone_box))
        state = state.parent
        return res
        """
        parent_state = self.startState
        parent_state.parent = None
        parent_state.gone_box = target
        parent_state.skill = 0
        heappush(self.frontierList, (1, 0, hash(parent_state), parent_state))
        print("frontier", self.frontierList)
        i = 1
        while self.frontierList:
            if self.frontierList[0][0] == i: # if the first one in queue is the correct depth
                cur_depth, cost1, id, state1 = heappop(self.frontierList)
                if state1.skill != 0:
                    return self.trace_sol(state2)
                print("len",len(state1.gone_box.adjacent_box()))
                for adj_box in state1.gone_box.adjacent_box():
                    gone_box_skill = self.check_hypothesis(state1.gone_box, adj_box, state1)
                    if gone_box_skill != 0:
                        state1.skill = gone_box_skill
                        state2 = deepcopy(state1)
                        state2.parent = state1
                        state2.gone_box = adj_box
                        state2.removed = None
                        state2.skill = adj_box.best_skill(state2.allBoxes)
                        print("Gone box:", state2.gone_box.name, "state skill:", state2.skill)
                        # removing adj_box needs the removal of other boxes
                        cost2 = cost1 + gone_box_skill
                        heappush(self.frontierList, (i+1, cost2, hash(state2), state2))
                        print("frontier", self.frontierList)
                    else: # the removal of adj_box does not make gone_box graspable
                        state2 = deepcopy(state1)
                        state2.parent = state1
                        state2.gone_box = state1.gone_box
                        newAllBoxes = state1.allBoxes[:]
                        for box_x in newAllBoxes:
                            if box_x.name == adj_box.name:
                                newAllBoxes.remove(box_x)
                                break
                        state2.allBoxes = newAllBoxes
                        state2.update_whole_environment()
                        state2.removed = adj_box
                        state2.skill = adj_box.best_skill(state2.allBoxes)
                        # assert state2.skill != 0
                        cost2 = cost1 + state2.skill
                        heappush(self.frontierList, (i+1, cost2, hash(state2), state2))
            else:
                i += 1


# Create boxes
boxes = []
coord = np.array([[310.1452, 324.39203, 578.3942, 498.38672],
         [314.69073, 147.6358, 579.8356, 323.22113],
         [572.48645, 345.97537, 775.3653, 453.26862],
         [574.8529, 181.5, 781.0047, 289.5],
         [570.4848, 450.48865, 775.04504, 558.99945],
         [773.69305, 327.93094, 773.69305+108, 327.9+206],
        #  [570.4848, 558.99945, 775.04504, 559+268]
         ]) * 0.001
heights = np.array([0.135, 0.135, 0.35, 0.35, 0.35, 0.35]).reshape(6, 1)
# print(heights)
for i in range(len(coord)):
    l_y = (coord[i][3] - coord[i][1])
    l_x = (coord[i][2] - coord[i][0])
    print("l_x", l_x, l_y)
    boxes.append(Box(i+1, coord[i], l_x, l_y, heights[i][0]))

"""
l_x 0.26824899999999996 0.17399469000000006
l_x 0.26514487000000003 0.17558533000000004
l_x 0.20287885000000005 0.10729325
l_x 0.2061518 0.10799999999999998
l_x 0.20456023999999995 0.10851080000000007
l_x 0.10799999999999998 0.20596906000000004
"""

# For visualization
sim_coord_eg = np.append(coord, heights, axis=1)

sim_coord_eg1 = np.array([[0.5704848 , 0.55899945, 0.77504504, 0.95899945, 0.1085108 ],
       [0.3101452 , 0.32439203, 0.5783942 , 0.49838672, 0.135],
       [0.31469073, 0.1476358 , 0.5798356 , 0.32322113, 0.135     ],
       [0.57248645, 0.34597537, 0.7753653 , 0.45326862, 0.4       ],
       [0.5748529, 0.1815   , 0.7810047, 0.2895   , 0.4      ],
       [0.77369305, 0.32793094, 0.88169305, 0.5339    , 0.4       ]])
# check_collision(sim_coord_eg, True)

state = State(boxes[0], boxes)
state.update_whole_environment()
solvers = Solvers(state)
solvers.planner(boxes[0])
# check_collision()