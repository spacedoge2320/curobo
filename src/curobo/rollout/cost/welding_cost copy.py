from typing import List, Optional
from dataclasses import dataclass

# Third Party
import torch

# CuRobo
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .cost_base import CostBase, CostConfig

import uuid


@get_torch_jit_decorator()
def working_angle_error_fn(x_vectors:torch.Tensor, start_point: torch.Tensor, end_point:torch.Tensor, tool_axis_vec:torch.Tensor):
    # Rotated x-vector calculation
    weldline_vec = (end_point - start_point) / (torch.norm(end_point - start_point)+0.000001)
    
    # Calculate the deviation from 45deg angle, by projecting the rotated x-vector on to the normal of the weldline and tool axis
    normal = torch.cross(weldline_vec, tool_axis_vec, dim=-1)
    normal = normal / torch.norm(normal)
    projection_on_normal = torch.matmul(x_vectors, normal) 
    
    yaw_error = torch.abs(projection_on_normal)

    return yaw_error


@get_torch_jit_decorator()
def working_angle_consistency_error_fn(x_vectors:torch.Tensor, start_point: torch.Tensor, end_point:torch.Tensor, tool_axis_vec:torch.Tensor, cost_mask:torch.Tensor):
    # Rotated x-vector calculation
    weldline_vec = (end_point - start_point) / (torch.norm(end_point - start_point)+0.000001)
    
    # Calculate the deviation from 45deg angle, by projecting the rotated x-vector on to the normal of the weldline and tool axis
    normal = torch.cross(weldline_vec, tool_axis_vec, dim=-1)
    normal = normal / torch.norm(normal)
    projection_on_normal = torch.matmul(x_vectors, normal) 

    #print(x_vectors)
    projection_on_normal_weldline = projection_on_normal*cost_mask

    projection_on_normal_weldline_count_nonzero = torch.sum(cost_mask)+0.000001
    projection_on_normal_weldline_sum= torch.sum(projection_on_normal_weldline)+0.000001

    #print(projection_on_normal_weldline_sum/projection_on_normal_weldline_count_nonzero)

    consistency_error = torch.abs(projection_on_normal - projection_on_normal_weldline_sum/projection_on_normal_weldline_count_nonzero)

    # Apply error for abs(projection on normal) > 0.08715574274
    bound_error = torch.relu(torch.abs(projection_on_normal) - 0.08715574274)

    #print(consistency_error, bound_error)

    return consistency_error, bound_error

@get_torch_jit_decorator()
def working_angle_bound_error_fn(x_vectors:torch.Tensor, start_point: torch.Tensor, end_point:torch.Tensor, tool_axis_vec:torch.Tensor):
    # Rotated x-vector calculation
    weldline_vec = (end_point - start_point) / torch.norm(end_point - start_point)
    
    # Calculate the deviation from 45deg angle, by projecting the rotated x-vector on to the normal of the weldline and tool axis
    normal = torch.cross(weldline_vec, tool_axis_vec, dim=-1)
    normal = normal / torch.norm(normal)
    projection_on_normal = torch.matmul(x_vectors, normal) 
    
    yaw_error = torch.abs(projection_on_normal)

    return yaw_error


@get_torch_jit_decorator()
def travel_angle_error_fn(x_vectors:torch.Tensor, start_point: torch.Tensor, end_point:torch.Tensor, start_end_angle:torch.Tensor,  start_matrix: torch.Tensor, end_matrix: torch.Tensor):
    # Rotated x-vector calculation
    weldline_vec = (end_point - start_point) / torch.norm(end_point - start_point)
    
    angles = torch.asin(torch.matmul(x_vectors, weldline_vec))
    
    #start_angle = torch.sum(angles*start_matrix, dim=1, keepdim=True)
    #end_angle = torch.sum(angles*end_matrix, dim=1, keepdim=True)
    
    start_angle_diff = torch.abs(angles - torch.deg2rad(start_end_angle[0]))*start_matrix
    end_angle_diff = torch.abs(angles - torch.deg2rad(start_end_angle[1]))*end_matrix

    vector_diff = x_vectors - torch.roll(x_vectors, shifts=-1, dims=1)

    decrease_penalty = -torch.relu(-torch.matmul(vector_diff, weldline_vec))

    return start_angle_diff, end_angle_diff, decrease_penalty


@get_torch_jit_decorator()
def waypoint_angle_error_fn(x_vectors: torch.Tensor,start_point: torch.Tensor, end_point:torch.Tensor, angle: torch.Tensor, waypoint_matrix: torch.Tensor):

    weldline_vec = (end_point - start_point) / torch.norm(end_point - start_point)
    
    angles = torch.asin(torch.matmul(x_vectors, weldline_vec))

    angle_diff = torch.abs(angles - angle)
    
    angle_cost = angle_diff * waypoint_matrix
    
    return angle_cost
    
    
    
@get_torch_jit_decorator()
def calculate_tool_vectors(ee_quat_batch:torch.Tensor, xvectors:torch.Tensor, one_tensor:torch.Tensor):

    #print(ee_quat_batch)
    #print(xvectors)
    vectors = xvectors.repeat(ee_quat_batch.shape[0], ee_quat_batch.shape[1], 1)
    
    quaternions = ee_quat_batch
    
    vector_quats = torch.cat([one_tensor.repeat(ee_quat_batch.shape[0],ee_quat_batch.shape[1], 1), vectors], dim=2)
    q_conjugates = torch.cat([quaternions[:,:,:1], -quaternions[:,:,1:]], dim=2)

    #print(quaternions)
    
    q1 = quaternions
    q2 = vector_quats
    
    w1, x1, y1, z1 = q1[:,:, 0], q1[:,:, 1], q1[:,:, 2], q1[:,:, 3]
    w2, x2, y2, z2 = q2[:,:, 0], q2[:,:, 1], q2[:,:, 2], q2[:,:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    q1 = torch.stack([w, x, y, z], dim=2)
    q2 = q_conjugates
    
    w1, x1, y1, z1 = q1[:,:, 0], q1[:,:, 1], q1[:,:, 2], q1[:,:, 3]
    w2, x2, y2, z2 = q2[:,:, 0], q2[:,:, 1], q2[:,:, 2], q2[:,:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    v_rotated = torch.stack([w, x, y, z], dim=2)
    
    x_vectors = v_rotated[:, :,1:]

    return x_vectors

@get_torch_jit_decorator()
def weldline_distance_error_fn(ee_pos_batch: torch.Tensor, start_point: torch.Tensor, end_point: torch.Tensor):
    
    start_relative_pos = ee_pos_batch[:, :, 0:3] - start_point
    
    dir_vec = (end_point - start_point) / torch.norm(end_point - start_point)
    
    projection = torch.matmul(start_relative_pos, dir_vec)
    projection_positive = torch.relu(projection)
    projection_capped = torch.norm(end_point - start_point) -torch.relu(torch.norm(end_point - start_point) - projection_positive)
    
    closest_point_on_line = start_point + projection_capped.unsqueeze(-1) * dir_vec.unsqueeze(0)
    
    distance = torch.norm(ee_pos_batch[:, :,0:3] - closest_point_on_line, dim=-1)
    
    
    vector_from_start = ee_pos_batch - start_point
    
    per_step_vector =  vector_from_start - torch.roll(vector_from_start, shifts=1, dims=1)
    
    per_step_acceleration = per_step_vector - torch.roll(per_step_vector, shifts=1, dims=1)

    squared_acc = torch.norm(per_step_acceleration, dim=-1)


    return distance, squared_acc
    


@get_torch_jit_decorator()
def constant_movement_error_fn(ee_pos_batch: torch.Tensor, start_matrix: torch.Tensor, end_matrix: torch.Tensor):
    
    masked_positions_start = ee_pos_batch * start_matrix.unsqueeze(-1)
    traj_start_point = torch.sum(masked_positions_start, dim=1, keepdim=True)

    masked_positions_end = ee_pos_batch * end_matrix.unsqueeze(-1)
    traj_end_point = torch.sum(masked_positions_end, dim=1, keepdim=True)
    
    #print(end_matrix[0].cpu().numpy())
    
    #print(traj_start_point.cpu().numpy(), traj_end_point.cpu().numpy())
    
    b, h, _ = ee_pos_batch.shape
    t = torch.arange(h, device=ee_pos_batch.device, dtype=ee_pos_batch.dtype)
    start_idx = torch.sum(t*start_matrix[0], dim=0, keepdim=False)+2
    end_idx = torch.sum(t*end_matrix[0], dim=0, keepdim=False) -2
    
    t = (t - start_idx)/(end_idx - start_idx)
    t = torch.relu(t)
    t = 1 - torch.relu(1-t)
    
    #print(t.cpu().numpy())
    
    #print(t.shape, traj_start_point.shape, traj_end_point.shape)
    
    interpolated_points = traj_start_point + t.unsqueeze(0).unsqueeze(-1) * (traj_end_point - traj_start_point)
    


    
    distances = torch.norm(ee_pos_batch - interpolated_points, dim=-1)
    
    
    return distances, traj_start_point, traj_end_point



@get_torch_jit_decorator()
def waypoint_distance_error_fn(ee_pos_batch: torch.Tensor, waypoint: torch.Tensor, waypoint_matrix: torch.Tensor):
    
    waypoint_relative_pos = ee_pos_batch - waypoint

    waypoint_distance = (torch.norm(waypoint_relative_pos, dim=-1)) * waypoint_matrix
    
    return waypoint_distance




@dataclass
class WeldingCostMetric:
    
    tool_axis_vec: torch.Tensor = torch.tensor([1.0, 0.0, 0.0])
    welding_start_point: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])  
    welding_end_point: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])
    welding_start_end_angle: torch.Tensor = torch.tensor([0.0, 0.0])
    welding_start_end_range: torch.Tensor = torch.tensor([0.0, 1.0])
    welding_start_end_position_weight: torch.Tensor = torch.tensor([[0.0, 0.0]]) 
    welding_start_end_angle_weight: torch.Tensor = torch.tensor([[0.0, 0.0]]) 
    
    welding_tool_contact_weight: torch.Tensor = torch.tensor([0.0])
    welding_tool_working_angle_weight: torch.Tensor = torch.tensor([0.0])
    welding_tool_travel_angle_weight: torch.Tensor = torch.tensor([0.0])
    welding_movement_consistency_weight: torch.Tensor = torch.tensor([0.0, 0.0])
    
    waypoint_position: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])
    waypoint_weight: torch.Tensor = torch.tensor([0.0])
    waypoint_ratio: torch.Tensor = torch.tensor([0.0])

    
    @classmethod
    def reset_metric(cls):
        return WeldingCostMetric( 
            tool_axis_vec=torch.tensor([1.0, 0.0, 0.0]),
            welding_start_end_angle=torch.tensor([0.0, 0.0]),
            welding_start_point=torch.tensor([0.0, 0.0, 0.0]),
            welding_end_point=torch.tensor([0.0, 0.0, 1.0]),
            welding_start_end_range=torch.tensor([0.0, 1.0]),
            welding_start_end_position_weight=torch.tensor([[0.0, 0.0]]),  
            welding_start_end_angle_weight=torch.tensor([[0.0, 0.0]]),
             
            welding_tool_contact_weight=torch.tensor([0.0]),
            welding_tool_working_angle_weight=torch.tensor([0.0]),
            welding_movement_consistency_weight=torch.tensor([0.0, 0.0]),
            welding_tool_travel_angle_weight=torch.tensor([0.0]),
            waypoint_position=torch.tensor([0.0, 0.0, 0.0]),
            waypoint_weight=torch.tensor([0.0]),
            waypoint_ratio=torch.tensor([0.0]),
        )


class WeldingCost(CostBase):
    def __init__(self, config: CostConfig):
        self.uid = uuid.uuid4().hex
        CostBase.__init__(self, config)
        self.vel_idxs = torch.arange(
            self.dof, 2 * self.dof, dtype=torch.long, device=self.tensor_args.device
        )
        with torch.no_grad():
            self.I = torch.eye(self.dof, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            
            self.tool_axis_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            
            self.welding_start_point = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_end_point = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_start_end_angle = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_start_end_range = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_start_end_position_weight = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_start_end_angle_weight = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_tool_travel_angle_weight = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_tool_contact_weight = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_tool_working_angle_weight = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_movement_consistency_weight = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            
            self.waypoint_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.waypoint_weight = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.waypoint_ratio = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            
              # Changed from endpoint_goal_weight
            
            self.x_vectors = torch.tensor([1, 0, 0], dtype=torch.float32).to(self.tensor_args.device)
            self.one_tensor = torch.tensor([0.0], dtype=torch.float32).to(self.tensor_args.device)
            self.cost_mask = torch.tensor([0.0], dtype=torch.float32).to(self.tensor_args.device)
            
            if self.welding_start_end_position_weight is not None:
                if self.welding_start_end_position_weight[1] == 0.0:
                    self.end_error_weight = 0.0
                else:
                    self.end_error_weight = 1.0
                    
                    
    def return_metric(self):
        return WeldingCostMetric(
            tool_axis_vec=self.tool_axis_vec,
            welding_start_point=self.welding_start_point,
            welding_end_point=self.welding_end_point,
            welding_start_end_range=self.welding_start_end_range,
            welding_start_end_angle=self.welding_start_end_angle,
            welding_start_end_position_weight=self.welding_start_end_position_weight,
            welding_start_end_angle_weight=self.welding_start_end_angle_weight,
            welding_tool_contact_weight=self.welding_tool_contact_weight,
            welding_tool_working_angle_weight=self.welding_tool_working_angle_weight,
            welding_tool_travel_angle_weight=self.welding_tool_travel_angle_weight,
            welding_movement_consistency_weight=self.welding_movement_consistency_weight,
            waypoint_position=self.waypoint_position,
            waypoint_weight=self.waypoint_weight,
            waypoint_ratio=self.waypoint_ratio,
        )
        
    def update_metric(self, metric: WeldingCostMetric):
        self.counter = 0
        with torch.no_grad():
            if metric.tool_axis_vec is not None:
                self.tool_axis_vec[:] = metric.tool_axis_vec
                
            if metric.welding_start_point is not None:
                self.welding_start_point[:] = metric.welding_start_point
            if metric.welding_end_point is not None:
                self.welding_end_point[:] = metric.welding_end_point
            if metric.welding_start_end_range is not None:
                self.welding_start_end_range[:] = metric.welding_start_end_range
            if metric.welding_start_end_angle is not None:
                self.welding_start_end_angle[:] = metric.welding_start_end_angle
            if metric.welding_start_end_position_weight is not None:
                self.welding_start_end_position_weight[:] = metric.welding_start_end_position_weight
            if metric.welding_start_end_angle_weight is not None:
                self.welding_start_end_angle_weight[:] = metric.welding_start_end_angle_weight
                
            if metric.welding_tool_contact_weight is not None:
                self.welding_tool_contact_weight[:] = metric.welding_tool_contact_weight
            if metric.welding_tool_working_angle_weight is not None:
                self.welding_tool_working_angle_weight[:] = metric.welding_tool_working_angle_weight
            if metric.welding_movement_consistency_weight is not None:
                self.welding_movement_consistency_weight[:] = metric.welding_movement_consistency_weight

            if metric.welding_tool_travel_angle_weight is not None:
                self.welding_tool_travel_angle_weight[:] = metric.welding_tool_travel_angle_weight
                
            if metric.waypoint_position is not None:
                self.waypoint_position[:] = metric.waypoint_position
            if metric.waypoint_weight is not None:
                self.waypoint_weight[:] = metric.waypoint_weight
            if metric.waypoint_ratio is not None:
                self.waypoint_ratio[:] = metric.waypoint_ratio
                
            if self.welding_start_end_position_weight is not None:
                if self.welding_start_end_position_weight[1] == 0.0:
                    self.end_error_weight = 0.0
                else:
                    self.end_error_weight = 1.0



    def reset_metric(self):
        with torch.no_grad():
            self.tool_axis_vec[:] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            
            self.welding_start_point[:] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_end_point[:] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_start_end_range[:] = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_start_end_angle[:] = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_start_end_position_weight[:] = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_start_end_angle_weight[:] = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_tool_travel_angle_weight[:] = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_tool_contact_weight[:] = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_tool_working_angle_weight[:] = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.welding_movement_consistency_weight[:] = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            
            self.waypoint_position[:] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.waypoint_weight[:] = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            self.waypoint_ratio[:] = torch.tensor([0.0], dtype=torch.float32, device=self.tensor_args.device)
            
    def enable(self, enabled=True):
        self.enabled = enabled
        
    def logcosh(self, x, a=1):
        return (x/a + torch.log(1+torch.exp(-2*(x/a))) - 0.69314718056)*a
    
    def logcosh_custom(self, x, arctan_inside, arctan_outside, logcosh_inside, logcosh_outside):
        return arctan_outside*torch.square(torch.arctan(arctan_inside*x)) + logcosh_outside*(x*logcosh_inside + torch.log(1+torch.exp(-2*x*logcosh_inside)) - 0.69314718056)
        
        
    def forward_start_end_position(self, ee_pos_batch, ee_quat_batch):
        b, h, _ = ee_pos_batch.shape
        # Create a range tensor for the horizon
        horizon_range = torch.arange(h, device=self.tensor_args.device).unsqueeze(0).expand(b, -1)
        
        # Calculate what portion to apply the welding cost to.
        # This should be applied to the actual welding portion
        start_threshold = self.welding_start_end_range[0] * (h - 1) 
        end_threshold = self.welding_start_end_range[1] * (h - 1) 
        waypoint_threshold = self.waypoint_ratio * (h - 1) 
        
        cost_mask = (horizon_range >= start_threshold).float() * (horizon_range <= end_threshold).float()
        cost_mask_except_last = (horizon_range >= start_threshold).float() * (horizon_range <= end_threshold - 1).float()
        
        start_matrix = (horizon_range >= start_threshold).float() * (horizon_range < start_threshold + 1).float()
        end_matrix = (horizon_range <= end_threshold).float() * (horizon_range > end_threshold - 1).float()
        waypoint_matrix = (horizon_range <= waypoint_threshold).float() * (horizon_range > waypoint_threshold - 1).float()
        
        # Calculate where to apply weldtip collision cost to
        # This should be applied to approach or the return trajectory
        weldtip_cost_matrix = (horizon_range <= start_threshold -2).float() + (horizon_range >= end_threshold + 2).float()
        
        # Calculate costs that are applied to each point
        # Flatten the ee_quat_batch and ee_pos_batch for costs that are applied to each point
        
        tool_x_vectors = calculate_tool_vectors(ee_quat_batch, self.x_vectors, self.one_tensor)

        working_angle_error = working_angle_error_fn(tool_x_vectors, self.welding_start_point, self.welding_end_point, self.tool_axis_vec)
        
        start_angle_error, end_angle_error, travel_angle_error = travel_angle_error_fn(tool_x_vectors, self.welding_start_point, self.welding_end_point, self.welding_start_end_angle, start_matrix, end_matrix)
        
        weldline_distance_error, acc_squared_error = weldline_distance_error_fn(ee_pos_batch, self.welding_start_point, self.welding_end_point)

        constant_movement_error, start_point, end_point = constant_movement_error_fn(ee_pos_batch, start_matrix, end_matrix)

        start_point_distance_error = waypoint_distance_error_fn(ee_pos_batch, self.welding_start_point.repeat(b,h,1), start_matrix)

        start_point_pause_error1 = waypoint_distance_error_fn(ee_pos_batch, start_point.repeat(1,h,1), torch.roll(start_matrix, 1, dims=1))
        start_point_pause_error2 = waypoint_distance_error_fn(ee_pos_batch, start_point.repeat(1,h,1), torch.roll(start_matrix, -1, dims=1))

        end_point_distance_error = waypoint_distance_error_fn(ee_pos_batch, self.welding_end_point.repeat(b,h,1), end_matrix)



        working_angle_cost = self.logcosh(working_angle_error, 0.01)* self.welding_tool_working_angle_weight

        travel_angle_cost = self.logcosh(travel_angle_error, 0.01)* self.welding_tool_travel_angle_weight

        start_travel_angle_cost = self.logcosh(start_angle_error, 0.01)*self.welding_start_end_angle_weight[0]
        
        end_travel_angle_cost = self.logcosh(end_angle_error, 0.01)*self.welding_start_end_angle_weight[1]
        
        weldline_distance_cost =  self.logcosh(weldline_distance_error, 0.0005)*self.welding_tool_contact_weight
        
        constant_movement_cost = self.logcosh(constant_movement_error,0.0005)* cost_mask_except_last* self.welding_movement_consistency_weight[0]

        acc_squared_cost = self.logcosh(acc_squared_error, 0.000001)* self.welding_movement_consistency_weight[0]

        start_position_cost = self.logcosh(start_point_distance_error, 0.001)*self.welding_start_end_position_weight[0]
        
        start_pause_cost = self.logcosh(start_point_pause_error1 + start_point_pause_error2, 0.001)* self.welding_start_end_position_weight[0]
        
        end_position_cost = self.logcosh(end_point_distance_error, 0.001)* self.welding_start_end_position_weight[1]
        

        # Final sum of costs
        cost = torch.sum(torch.stack([start_position_cost, start_pause_cost, end_position_cost], dim=0), dim=[0]) * cost_mask
        
        return cost

        

    def forward(self, ee_pos_batch, ee_quat_batch):

        #print(ee_pos_batch.shape)
        b, h, _ = ee_pos_batch.shape
        # Create a range tensor for the horizon
        horizon_range = torch.arange(h, device=self.tensor_args.device).unsqueeze(0).expand(b, -1)
        
        # Calculate what portion to apply the welding cost to.
        # This should be applied to the actual welding portion
        start_threshold = self.welding_start_end_range[0] * (h - 1) 
        end_threshold = self.welding_start_end_range[1] * (h - 1) 
        waypoint_threshold = self.waypoint_ratio * (h - 1) 
        
        cost_mask = (horizon_range >= start_threshold).float() * (horizon_range <= end_threshold).float()
        cost_mask_except_last = (horizon_range >= start_threshold).float() * (horizon_range <= end_threshold - 1).float()
        
        start_matrix = (horizon_range >= start_threshold).float() * (horizon_range < start_threshold + 1).float()
        end_matrix = (horizon_range <= end_threshold).float() * (horizon_range > end_threshold - 1).float()
        waypoint_matrix = (horizon_range <= waypoint_threshold).float() * (horizon_range > waypoint_threshold - 1).float()
        
        # Calculate where to apply weldtip collision cost to
        # This should be applied to approach or the return trajectory
        weldtip_cost_matrix = (horizon_range <= start_threshold -2).float() + (horizon_range >= end_threshold + 2).float()
        
        # Calculate costs that are applied to each point
        # Flatten the ee_quat_batch and ee_pos_batch for costs that are applied to each point
        #nan_count = torch.isnan(ee_quat_batch).sum().item()
        #self.counter += 1
        #print(self.counter)
        #print(ee_quat_batch.shape)
        #print(f"Number of NaN values in ee_quat_batch: {nan_count/4/44}")
        tool_x_vectors = calculate_tool_vectors(ee_quat_batch, self.x_vectors, self.one_tensor)

        working_angle_consistency_error, working_angle_bound_error = working_angle_consistency_error_fn(tool_x_vectors, self.welding_start_point, self.welding_end_point, self.tool_axis_vec, cost_mask)

        working_angle_error = working_angle_error_fn(tool_x_vectors, self.welding_start_point, self.welding_end_point, self.tool_axis_vec)
        
        start_angle_error, end_angle_error, travel_angle_error = travel_angle_error_fn(tool_x_vectors, self.welding_start_point, self.welding_end_point, self.welding_start_end_angle, start_matrix, end_matrix)
        
        weldline_distance_error, acc_squared_error = weldline_distance_error_fn(ee_pos_batch, self.welding_start_point, self.welding_end_point)

        constant_movement_error, start_point, end_point = constant_movement_error_fn(ee_pos_batch, start_matrix, end_matrix)

        start_point_distance_error = waypoint_distance_error_fn(ee_pos_batch, self.welding_start_point.repeat(b,h,1), start_matrix)

        start_point_pause_error1 = waypoint_distance_error_fn(ee_pos_batch, start_point.repeat(1,h,1), torch.roll(start_matrix, 1, dims=1))
        start_point_pause_error2 = waypoint_distance_error_fn(ee_pos_batch, start_point.repeat(1,h,1), torch.roll(start_matrix, -1, dims=1))

        end_point_distance_error = waypoint_distance_error_fn(ee_pos_batch, self.welding_end_point.repeat(b,h,1), end_matrix)



        #working_angle_cost = torch.square(self.logcosh((working_angle_consistency_error)))
        working_angle_cost = torch.square(self.logcosh((working_angle_error*100))*self.welding_tool_working_angle_weight)

        travel_angle_cost = self.logcosh(travel_angle_error, 0.01)* self.welding_tool_travel_angle_weight

        start_travel_angle_cost = self.logcosh(start_angle_error, 0.01)*self.welding_start_end_angle_weight[0]

        end_travel_angle_cost = self.logcosh(end_angle_error, 0.01)*self.welding_start_end_angle_weight[1]
        
        weldline_distance_cost =  torch.square(self.logcosh(weldline_distance_error, 0.001)*self.welding_tool_contact_weight)
        
        constant_movement_cost = self.logcosh(constant_movement_error, 0.01)* cost_mask_except_last* self.welding_movement_consistency_weight[0]

        acc_squared_cost = self.logcosh(acc_squared_error, 0.01)* self.welding_movement_consistency_weight[0]

        start_position_cost = self.logcosh(start_point_distance_error, 0.001)*self.welding_start_end_position_weight[0]
        
        start_pause_cost = self.logcosh((start_point_pause_error1 + start_point_pause_error2), 0.01)* self.welding_start_end_position_weight[0]
        
        end_position_cost = self.logcosh(end_point_distance_error, 0.01)* self.welding_start_end_position_weight[1]
        

        # Final sum of costs
        cost = torch.sum(torch.stack([working_angle_cost, travel_angle_cost, start_travel_angle_cost, end_travel_angle_cost, weldline_distance_cost, constant_movement_cost, start_position_cost, start_pause_cost, end_position_cost], dim=0), dim=[0]) * cost_mask
        
        return cost, weldtip_cost_matrix