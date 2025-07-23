# ====================================================================
# 文件: mapf_solver_v20_final.py
#
#
# v20 核心修复:
#   1. 严格的求解器分离:
#
#      - 单个智能体(其规划路径无冲突)的路径被直接采纳，不再进入
#        错误的死锁解决流程。
#   2. 明确“单独被困”智能体的处理:
#      - 单独被困的智能体现在被 AgentMemory 正确标记为 "ESCAPING"。
#      - 主循环逻辑会为这些 "ESCAPING" 状态的智能体自动调用
#        强大的A*逃逸规划器，从而解决“卡在凹槽”的问题。

#   这是经过反复调试后最稳定、逻辑最清晰的版本。
# ========================================
# 

import torch
import numpy as np
import time
import random
import logging
from typing import Dict, List, Tuple, Optional, Set, Deque, Any, cast
import networkx as nx
from collections import defaultdict, deque
import heapq
import itertools
import numba

# --- 外部依赖 ---
from unet_model_new import UNetPotentialField  
from action_sequence_decoder_v12 import decode_action_sequence_refined
from local_cbs_solver_robust_v12_optimized import solve_local_cbs_robust, detect_all_conflicts_spacetime

# --- 日志与常量 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ACTION_DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}; ACTION_STAY = 0
UNKNOWN_CELL, FREE_CELL, OBSTACLE_CELL = 2, 0, 1

# ====================================================================
# 0. 核心数据结构与通用函数
# ====================================================================
@numba.jit(nopython=True, cache=True)
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def get_map_hash(grid_map: np.ndarray) -> int: return hash(grid_map.tobytes())



# ====================================================================
# 0. 核心数据结构与通用函数
# ====================================================================
# 保留Numba装饰器以获得性能
@numba.jit(nopython=True, cache=True)
def intelligent_wall_follower(
    start_pos: Tuple[int, int],
    prev_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    obstacle_map: np.ndarray,
    max_steps: int = 20
) -> List[Tuple[int, int]]:
    """
    【V22.7 终极兼容版】智能绕墙逃逸算法。
    修复了Numba无法处理 Optional[Tuple] 索引的问题。
    """
    h, w = obstacle_map.shape
    path = [(0, 0)]; path.pop()

    while True:
        neighbors = {
            'up': (start_pos[0] - 1, start_pos[1]),
            'right': (start_pos[0], start_pos[1] + 1),
            'down': (start_pos[0] + 1, start_pos[1]),
            'left': (start_pos[0], start_pos[1] - 1)
        }
        obs_pattern = {'up': False, 'right': False, 'down': False, 'left': False}
        for direction, pos in neighbors.items():
            if not (0 <= pos[0] < h and 0 <= pos[1] < w and not obstacle_map[pos[0], pos[1]]):
                obs_pattern[direction] = True

        goal_vec = (goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
        prefer_left = True
        initial_move_dir = 'up'

        if obs_pattern['up'] and obs_pattern['down'] and obs_pattern['left'] and obs_pattern['right']:
            move_back = (start_pos[0] + (start_pos[0] - prev_pos[0]), start_pos[1] + (start_pos[1] - prev_pos[1]))
            if 0 <= move_back[0] < h and 0 <= move_back[1] < w and not obstacle_map[move_back[0], move_back[1]]:
                 path.append(move_back)
            break

        elif obs_pattern['up'] and obs_pattern['down']:
            if not obs_pattern['right']: initial_move_dir = 'right'; prefer_left = False
            elif not obs_pattern['left']: initial_move_dir = 'left'; prefer_left = True
            else: break

        elif obs_pattern['left'] and obs_pattern['right']:
            if not obs_pattern['up']: initial_move_dir = 'up'
            elif not obs_pattern['down']: initial_move_dir = 'down'
            else: break

        elif obs_pattern['up'] and obs_pattern['left']: initial_move_dir = 'right'; prefer_left = False
        elif obs_pattern['up'] and obs_pattern['right']: initial_move_dir = 'left'; prefer_left = True
        elif obs_pattern['down'] and obs_pattern['left']: initial_move_dir = 'right'; prefer_left = False
        elif obs_pattern['down'] and obs_pattern['right']: initial_move_dir = 'left'; prefer_left = True
        
        else:
            should_wall_follow = False
            if obs_pattern['up'] and goal_vec[0] <= 0: should_wall_follow = True
            elif obs_pattern['down'] and goal_vec[0] >= 0: should_wall_follow = True
            elif obs_pattern['left'] and goal_vec[1] <= 0: should_wall_follow = True
            elif obs_pattern['right'] and goal_vec[1] >= 0: should_wall_follow = True

            if should_wall_follow:
                if obs_pattern['up']: initial_move_dir = 'right' if goal_vec[1] > 0 else 'left'
                elif obs_pattern['down']: initial_move_dir = 'right' if goal_vec[1] > 0 else 'left'
                elif obs_pattern['left']: initial_move_dir = 'up' if goal_vec[0] < 0 else 'down'
                elif obs_pattern['right']: initial_move_dir = 'up' if goal_vec[0] < 0 else 'down'
            else:
                best_neighbor = start_pos
                min_dist = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])
                # Numba JIT requires explicit key iteration
                neighbor_keys = ('up', 'right', 'down', 'left')
                for key in neighbor_keys:
                    pos_val = neighbors[key]
                    r, c = pos_val
                    if 0 <= r < h and 0 <= c < w and not obstacle_map[r, c]:
                        dist = abs(r - goal_pos[0]) + abs(c - goal_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_neighbor = pos_val
                if best_neighbor != start_pos:
                    path.append(best_neighbor)
                break
        
        # --- V22.7 核心修复 ---
        # 1. 从 neighbors 字典中安全地获取坐标
        # .get() 在 nopython 模式下返回 Optional 类型，需要小心处理
        # 我们在这里不使用.get()，而是直接索引，因为我们知道key一定存在
        first_move_pos = neighbors[initial_move_dir]

        # 2. 显式地分离 None 检查和索引操作
        if first_move_pos is None: # 理论上不会发生，但这是安全的写法
             break 
        
        # 3. 在确认不是 None 之后，再进行索引和边界检查
        r, c = first_move_pos
        if not (0 <= r < h and 0 <= c < w and not obstacle_map[r, c]):
             break
        # --- 修复结束 ---
        
        path.append(first_move_pos)
        current_pos = first_move_pos
        
        dr, dc = current_pos[0] - start_pos[0], current_pos[1] - start_pos[1]
        current_dir_idx = 0
        if dr == -1 and dc == 0: current_dir_idx = 0
        elif dr == 0 and dc == 1: current_dir_idx = 1
        elif dr == 1 and dc == 0: current_dir_idx = 2
        elif dr == 0 and dc == -1: current_dir_idx = 3

        dirs = ((-1, 0), (0, 1), (1, 0), (0, -1))
        turn_order = ((-1, 0, 1, 2)) if prefer_left else ((1, 0, -1, 2))

        for _ in range(max_steps - 1):
            moved = False
            for turn in turn_order:
                next_dir_idx = (current_dir_idx + turn + 4) % 4
                d_dr, d_dc = dirs[next_dir_idx]
                next_pos = (current_pos[0] + d_dr, current_pos[1] + d_dc)
                if not (0 <= next_pos[0] < h and 0 <= next_pos[1] < w and not obstacle_map[next_pos[0], next_pos[1]]):
                    continue
                path.append(next_pos)
                current_pos = next_pos
                current_dir_idx = next_dir_idx
                moved = True
                break
            if not moved: break
        
        break

    return path



@numba.jit(nopython=True, cache=True)
def follow_wall(start_pos: Tuple[int, int], prev_pos: Tuple[int, int], 
                obstacle_map: np.ndarray, max_steps: int = 20, prefer_left: bool = True) -> List[Tuple[int, int]]:
    """
    【V22 新增】使用左手或右手规则沿着墙壁移动，以逃离局部最小值。
    这是当智能体使用A*等方法仍无法脱困时的最终手段。

    Args:
        start_pos (Tuple[int, int]): 智能体当前位置。
        prev_pos (Tuple[int, int]): 智能体上一步的位置，用于确定初始朝向。
        obstacle_map (np.ndarray): 障碍物地图 (True=障碍, False=可通行)。
        max_steps (int): 执行此策略的最大步数。
        prefer_left (bool): True表示使用左手规则，False表示右手规则。

    Returns:
        List[Tuple[int, int]]: 一条沿着墙壁移动的路径坐标列表 (不包含起点)。
    """
    h, w = obstacle_map.shape
    path = []
    current_pos = start_pos
    
    # 确定初始方向 (从 prev_pos -> start_pos)
    dr, dc = start_pos[0] - prev_pos[0], start_pos[1] - prev_pos[1]
    
    # 方向映射: 0:Up(-1,0), 1:Right(0,1), 2:Down(1,0), 3:Left(0,-1)
    # Numba JIT 要求字典的键和值类型一致，这里我们用if/elif代替
    current_dir_idx = 0 # 默认朝上
    if dr == -1 and dc == 0: current_dir_idx = 0
    elif dr == 0 and dc == 1: current_dir_idx = 1
    elif dr == 1 and dc == 0: current_dir_idx = 2
    elif dr == 0 and dc == -1: current_dir_idx = 3

    # 定义方向向量和转向顺序
    dirs = ((-1, 0), (0, 1), (1, 0), (0, -1)) # N, E, S, W
    
    # 左手规则: 尝试左转(-1)，直行(0)，右转(1)，回头(2)
    # 右手规则: 尝试右转(1)，直行(0)，左转(-1)，回头(2)
    turn_order = (-1, 0, 1, 2) if prefer_left else (1, 0, -1, 2)

    for _ in range(max_steps):
        moved = False
        # 按照转向顺序尝试移动
        for turn in turn_order:
            # 计算下一个朝向的索引 (e.g., 当前朝北(0), 左转-1 -> 朝西(3))
            next_dir_idx = (current_dir_idx + turn + 4) % 4
            d_dr, d_dc = dirs[next_dir_idx]
            next_pos = (current_pos[0] + d_dr, current_pos[1] + d_dc)

            # 检查下一个位置是否有效 (在边界内且不是障碍)
            if not (0 <= next_pos[0] < h and 0 <= next_pos[1] < w and not obstacle_map[next_pos[0], next_pos[1]]):
                continue
            
            # 移动成功
            path.append(next_pos)
            current_pos = next_pos
            current_dir_idx = next_dir_idx
            moved = True
            break # 找到移动方向，跳出内层循环
        
        if not moved: # 如果被完全困住，四个方向都无法移动
            break

    return path

def solve_by_push_and_rotate(group_list: List[int], cbs_map_for_solver: np.ndarray, agents_global_positions: List[Tuple[int, int]], sim_params: Dict) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """ Push and Rotate 算法的简化实现，用于解开小规模紧密死锁。 """
    verbose = sim_params['verbose']
    if verbose: logging.warning(f"Attempting Layer 3 Defense: Push-and-Rotate for group {group_list}.")
    
    pos_map = {aid: tuple(agents_global_positions[aid]) for aid in group_list}
    h, w = cbs_map_for_solver.shape
    
    q = deque([(p, [p]) for p in pos_map.values()])
    visited = set(pos_map.values())
    empty_pos = None
    
    while q:
        curr, path_to_curr = q.popleft()
        if curr not in pos_map.values():
            empty_pos = curr
            break
        if len(path_to_curr) > 20: continue
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (curr[0] + dr, curr[1] + dc)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and cbs_map_for_solver[neighbor] == 0 and neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, path_to_curr + [neighbor]))
                
    if not empty_pos:
        logging.error(f"Push-and-Rotate FAILED for group {group_list}: could not find a nearby empty cell.")
        return None

    solution_paths = {aid: [pos_map[aid]] for aid in group_list}
    try:
        current_empty = empty_pos
        moved_agents = set()
        for _ in range(len(group_list)):
            closest_agent_id = -1
            min_dist = float('inf')
            for aid in group_list:
                if aid not in moved_agents:
                    dist = heuristic(solution_paths[aid][-1], current_empty)
                    if dist < min_dist:
                        min_dist = dist
                        closest_agent_id = aid
            
            if closest_agent_id == -1: break
            agent_start_pos = solution_paths[closest_agent_id][-1]
            temp_obs_map = cbs_map_for_solver.copy()
            for other_aid in group_list:
                if other_aid != closest_agent_id and other_aid not in moved_agents:
                    temp_obs_map[solution_paths[other_aid][-1]] = 1
            
            path_to_empty = run_single_agent_astar(agent_start_pos, current_empty, cbs_map_for_solver, 50, None, None, sim_params)

            if not path_to_empty:
                logging.error(f"Push-and-Rotate FAILED for group {group_list}: agent {closest_agent_id} could not path to empty cell.")
                return None

            solution_paths[closest_agent_id].extend(path_to_empty)
            current_empty = agent_start_pos
            moved_agents.add(closest_agent_id)

        max_len = max(len(p) for p in solution_paths.values()) if solution_paths else 0
        for aid in group_list:
            path = solution_paths[aid]
            path.extend([path[-1]] * (max_len - len(path)))
            
        logging.info(f"Push-and-Rotate succeeded for group {group_list}.")
        return solution_paths

    except Exception as e:
        logging.error(f"Push-and-Rotate encountered an exception for group {group_list}: {e}")
        return None




class Constraint:
    def __init__(self, agent_id, loc, timestep, is_edge_constraint=False, prev_loc=None):
        self.agent_id, self.location, self.timestep, self.is_edge_constraint, self.prev_location = agent_id, loc, timestep, is_edge_constraint, prev_loc
    def __eq__(self, other): return isinstance(other, Constraint) and self.__dict__ == other.__dict__
    def __hash__(self): return hash((self.agent_id, self.location, self.timestep, self.is_edge_constraint, self.prev_location))

class Conflict:
    VERTEX, EDGE = 1, 2
    def __init__(self, conflict_type, agent1_id, agent2_id, loc1, timestep, loc2=None):
        self.type, self.agent1_id, self.agent2_id, self.location1, self.timestep, self.location2 = conflict_type, agent1_id, agent2_id, loc1, timestep, loc2


# ====================================================================
# 1. 记忆与高层规划模块 (稳定版本)
# ====================================================================

class AgentMemory:
    """ 智能体记忆模块，与 v17 逻辑一致 """
    def __init__(self, agent_id: int, map_shape: Tuple[int, int], sim_params: Dict):
        self.agent_id = agent_id; self.map_shape = map_shape; self.params = sim_params
        self.position_history: Deque[Tuple[int, int]] = deque(maxlen=self.params['pattering_history_len'])
        self.dynamic_cost_map = np.zeros(map_shape, dtype=np.float32)
        self.pattering_status = "NORMAL" # NORMAL, PATTERING, ESCAPING
        self.last_progress_check = {'step': 0, 'heuristic_dist': float('inf')}
        self.consecutive_no_progress_steps = 0
        self.escape_plan_failed_count=0


    def update_after_step(self, new_pos: Tuple[int, int], current_step: int, goal_pos: Tuple[int, int]):
        self.position_history.append(new_pos)
        #self.dynamic_cost_map *= self.params['dynamic_cost_decay']
        self.dynamic_cost_map == self.params['dynamic_cost_decay']

        self.dynamic_cost_map[self.dynamic_cost_map < 0.1] = 0
        current_dist = heuristic(new_pos, goal_pos)
        if current_step - self.last_progress_check['step'] >= self.params['pattering_progress_check_interval']:
            if current_dist >= self.last_progress_check['heuristic_dist'] - self.params['pattering_heuristic_improvement_threshold']:
                self.consecutive_no_progress_steps += 1
            else:
                self.consecutive_no_progress_steps = 0
                if self.pattering_status != "NORMAL": logging.debug(f"Agent {self.agent_id} made progress, resetting status to NORMAL.")
                self.pattering_status = "NORMAL"
            self.last_progress_check = {'step': current_step, 'heuristic_dist': current_dist}

    # 修改 check_and_handle_pattering 方法
    def check_and_handle_pattering(self, goal_pos: Tuple[int, int]):
        if self.pattering_status == "FORCED_GROUP_SOLVE": return
        if len(self.position_history) < self.params['pattering_history_len']: return
        
        no_progress = self.consecutive_no_progress_steps >= self.params['pattering_no_progress_steps_threshold']
        low_exploration = len(set(self.position_history)) <= self.params['pattering_unique_pos_threshold']
        
        if no_progress and low_exploration:
            # --- V22 增强 S1.1 开始 ---
            # 引入分层逃逸逻辑
            if self.pattering_status == "NORMAL":
                logging.warning(f"LDAM: Agent {self.agent_id} detected PATTERING. Status -> ESCAPING.")
                self.pattering_status = "ESCAPING" # 首先尝试 A* 逃逸
                self._penalize_line_of_sight(goal_pos)
            elif self.pattering_status == "ESCAPING":
                # 如果 A* 逃逸多次失败 (基于 escape_plan_failed_count)，则升级为更强的绕墙策略
                if self.escape_plan_failed_count > self.params.get('wall_follow_trigger_fails', 3):
                    # --- V22.3 增强 S1.1.1 开始: 触发惩罚 ---
                    logging.critical(f"LDAM: Agent {self.agent_id} A* escape failed. Triggering Wall-Following and PENALIZING TRAP ZONE.")
                    
                    # 在升级状态前，立即惩罚当前被困的区域！
                    self._penalize_trap_zone()
                    
                    self.pattering_status = "ESCAPING_BY_WALL_FOLLOW"
                    self.escape_plan_failed_count = 0 
                    # --- V22.3 增强 S1.1.1 结束 ---
            # --- V22 增强 S1.1 结束 ---
            
    # 在 AgentMemory 中也需要 reset_pattering_status_after_success 方法
    # 确保这个方法存在并且能重置状态
    def reset_pattering_status_after_success(self, path_found: bool):
        if path_found: 
            if self.pattering_status != "NORMAL":
                logging.info(f"Agent {self.agent_id} successfully escaped. Status -> NORMAL.")
            self.pattering_status = "NORMAL"
            self.consecutive_no_progress_steps = 0
            self.escape_plan_failed_count = 0
        elif self.pattering_status != "NORMAL": 
            self.escape_plan_failed_count += 1


    
    def _penalize_line_of_sight(self, goal_pos: Tuple[int, int]):
        penalty = self.params.get('line_of_sight_penalty', 50.0)
        x0, y0 = self.position_history[-1]; x1, y1 = goal_pos
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if 0 <= x0 < self.map_shape[0] and 0 <= y0 < self.map_shape[1]: self.dynamic_cost_map[x0, y0] += penalty
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 >= dy: err += dy; x0 += sx
            if e2 <= dx: err += dx; y0 += sy
                
    def _penalize_trap_zone(self):
        """
        【V22.3 新增】惩罚导致智能体陷入徘徊的“陷阱区域”。
        当绕墙逃逸被触发时调用此函数。它会给智能体最近的行动轨迹
        (即导致它被困的路径) 施加一个高的动态成本。
        """
        penalty_value = self.params.get('trap_penalty_value', 100.0)
        
        # 我们惩罚导致其被困的整个历史轨迹区域，这比只惩罚一个点更有效
        if not self.position_history:
            return
            
        logging.warning(f"Agent {self.agent_id}: Penalizing trap zone based on recent history.")
        
        # 将最近历史上所有独特的点都加入高成本
        unique_trap_positions = set(self.position_history)
        
        for r, c in unique_trap_positions:
            if 0 <= r < self.map_shape[0] and 0 <= c < self.map_shape[1]:
                # 显著增加此处的动态成本
                self.dynamic_cost_map[r, c] += penalty_value
# ====================================================================
# 2. 高层规划与记忆模块 (High-Level Planning & Memory Modules)
# ====================================================================

class BaseRegionGraph:
    """提取了 ARG 和 VCARG 的通用逻辑的基类。"""
    def __init__(self, map_shape: Tuple[int, int], config: Dict[str, Any], persistent_map: np.ndarray, region_size: int):
        self.map_h, self.map_w = map_shape
        self.config = config
        self.region_size = region_size
        if self.region_size <= 0:
             self.region_size = 16
             logging.warning(f"Region size was <= 0, reset to 16.")
        self.regions_h = (self.map_h + self.region_size - 1) // self.region_size
        self.regions_w = (self.map_w + self.region_size - 1) // self.region_size
        self.graph = nx.Graph()
        self.region_centers: Dict[int, Tuple[int, int]] = {}
        self.last_map_hash = None
        self._build_base_graph(persistent_map)

    def _get_region_id(self, pos: Tuple[int, int]) -> int:
        return (pos[0] // self.region_size) * self.regions_w + (pos[1] // self.region_size)

    def _build_base_graph(self, persistent_map: np.ndarray):
        for r_idx in range(self.regions_h):
            for c_idx in range(self.regions_w):
                region_id = r_idx * self.regions_w + c_idx
                center_r = r_idx * self.region_size + self.region_size // 2
                center_c = c_idx * self.region_size + self.region_size // 2
                self.region_centers[region_id] = (min(center_r, self.map_h - 1), min(center_c, self.map_w - 1))
                self.graph.add_node(region_id, penalty=0.0)
        self.update_graph_with_obstacles(persistent_map)

    def _check_boundary_passable(self, r_idx: int, c_idx: int, dr: int, dc: int, persistent_map: np.ndarray) -> bool:
        """【v16.2 核心改进】通过扫描边界来判断区域连通性，而非检查单个中点。"""
        passable_cells = 0
        num_checked = 0
        # 检查水平方向的邻居 (dc=1)
        if dc == 1:
            gateway_c = (c_idx + 1) * self.region_size
            if gateway_c >= self.map_w: return False
            start_r, end_r = r_idx * self.region_size, (r_idx + 1) * self.region_size
            # 在边界上每隔2格进行一次采样
            for r in range(start_r, end_r, 2):
                if 0 <= r < self.map_h:
                    num_checked += 1
                    # 检查边界两侧是否都是可通过的（非障碍物）
                    if persistent_map[r, gateway_c-1] != OBSTACLE_CELL and persistent_map[r, gateway_c] != OBSTACLE_CELL:
                        passable_cells += 1
        # 检查垂直方向的邻居 (dr=1)
        elif dr == 1:
            gateway_r = (r_idx + 1) * self.region_size
            if gateway_r >= self.map_h: return False
            start_c, end_c = c_idx * self.region_size, (c_idx + 1) * self.region_size
            for c in range(start_c, end_c, 2):
                if 0 <= c < self.map_w:
                    num_checked += 1
                    if persistent_map[gateway_r-1, c] != OBSTACLE_CELL and persistent_map[gateway_r, c] != OBSTACLE_CELL:
                        passable_cells += 1
        
        # 至少需要2个可通过的采样点，或者超过20%的采样点可通过
        return passable_cells >= 2 or (num_checked > 0 and (passable_cells / num_checked) > 0.2)

    def update_graph_with_obstacles(self, persistent_map: np.ndarray):
        current_map_hash = get_map_hash(persistent_map)
        if current_map_hash == self.last_map_hash: return
        if self.config.get('verbose', False): logging.debug(f"({type(self).__name__}): Map changed, updating region graph.")
        self.last_map_hash = current_map_hash
        for r_idx in range(self.regions_h):
            for c_idx in range(self.regions_w):
                rid1 = self._get_region_id((r_idx * self.region_size, c_idx * self.region_size))
                for dr, dc in [(1, 0), (0, 1)]:
                    nr_idx, nc_idx = r_idx + dr, c_idx + dc
                    if not (0 <= nr_idx < self.regions_h and 0 <= nc_idx < self.regions_w): continue
                    rid2 = self._get_region_id((nr_idx * self.region_size, nc_idx * self.region_size))
                    is_passable = self._check_boundary_passable(r_idx, c_idx, dr, dc, persistent_map)
                    self._update_edge_status(rid1, rid2, is_passable, r_idx, c_idx, nr_idx, nc_idx, persistent_map)

    def _update_edge_status(self, u, v, is_passable, r1, c1, r2, c2, p_map):
        raise NotImplementedError # 子类必须实现此方法

    def find_high_level_path(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> Optional[List[int]]:
        start_region, goal_region = self._get_region_id(start_pos), self._get_region_id(goal_pos)
        if start_region == goal_region or not self.graph.has_node(start_region) or not self.graph.has_node(goal_region):
            return None
        try:
            path = nx.astar_path(self.graph, start_region, goal_region,
                                 heuristic=lambda u, v: heuristic(self.region_centers[u], self.region_centers[v]),
                                 weight='weight')
            return path
        except nx.NetworkXNoPath:
            if self.config.get('verbose'): logging.warning(f"({type(self).__name__}): No path found from region {start_region} to {goal_region}")
            return None
    
    def get_subgoal_from_path(self, region_path: List[int]) -> Optional[Tuple[int, int]]:
        if not region_path or len(region_path) < 2: return None
        return self.region_centers.get(region_path[1])

class VeryCoarseRegionGraph(BaseRegionGraph):
    def __init__(self, map_shape: Tuple[int, int], config: Dict[str, Any], persistent_map: np.ndarray):
        region_size = config.get('coarse_region_size', 64)
        super().__init__(map_shape, config, persistent_map, region_size)

    def _update_edge_status(self, u, v, is_passable, r1, c1, r2, c2, p_map):
        has_edge = self.graph.has_edge(u, v)
        if is_passable and not has_edge:
            dist = heuristic(self.region_centers[u], self.region_centers[v])
            self.graph.add_edge(u, v, weight=dist)
        elif not is_passable and has_edge:
            self.graph.remove_edge(u, v)

class AdaptiveRegionGraph(BaseRegionGraph):
    def __init__(self, map_shape: Tuple[int, int], config: Dict[str, Any], persistent_map: np.ndarray):
        region_size = config.get('region_size', 16)
        super().__init__(map_shape, config, persistent_map, region_size)
    
    def _get_region_freeness(self, r_idx: int, c_idx: int, persistent_map: np.ndarray) -> float:
        start_r, start_c = r_idx * self.region_size, c_idx * self.region_size
        end_r, end_c = min(start_r + self.region_size, self.map_h), min(start_c + self.region_size, self.map_w)
        region_slice = persistent_map[start_r:end_r, start_c:end_c]
        if region_slice.size == 0: return 0.0
        return np.sum((region_slice == FREE_CELL) | (region_slice == UNKNOWN_CELL)) / region_slice.size

    def _update_edge_status(self, u, v, is_passable, r1, c1, r2, c2, p_map):
        has_edge = self.graph.has_edge(u, v)
        if is_passable and not has_edge:
            dist = heuristic(self.region_centers[u], self.region_centers[v])
            self.graph.add_edge(u, v, weight=dist)
        elif not is_passable and has_edge:
            self.graph.remove_edge(u, v)
        if self.graph.has_edge(u,v):
            freeness1 = self._get_region_freeness(r1, c1, p_map)
            freeness2 = self._get_region_freeness(r2, c2, p_map)
            weight_multiplier = 1.0 / (freeness1 * freeness2 + 1e-6)
            dist = heuristic(self.region_centers[u], self.region_centers[v])
            self.graph[u][v]['weight'] = dist * weight_multiplier
    
    def update_region_penalty(self, pos: Tuple[int, int], penalty: float):
        region_id = self._get_region_id(pos)
        if self.graph.has_node(region_id): self.graph.nodes[region_id]['penalty'] += penalty


# ====================================================================
# 2. 底层规划器与辅助函数
# ====================================================================
def calculate_dynamic_priorities(
    active_ids: List[int],
    agent_positions: List[Tuple[int, int]],
    agent_goals: List[Tuple[int, int]],
    agent_memories: Dict[int, AgentMemory],
    sim_params: Dict[str, Any]
) -> Dict[int, float]:
    """
    【V22 新增】为活跃的智能体计算动态优先级。
    优先级越高，路权越高。

    优先级由三个因素决定：
    1.  进度: 离目标越近，优先级越高。
    2.  等待时间: 陷入停滞越久，优先级越高 (防止饿死)。
    3.  拥堵程度: 周围越拥挤，优先级越低 (鼓励让行)。
    
    Returns:
        Dict[int, float]: 一个映射 {agent_id: priority_score} 的字典。
    """
    priorities: Dict[int, float] = {}
    
    # 从 sim_params 中获取权重，如果未定义则使用默认值
    w_progress = sim_params.get('prio_w_progress', 1.0)
    w_wait = sim_params.get('prio_w_wait', 0.5)
    w_congestion = sim_params.get('prio_w_congestion', 0.8)
    congestion_radius = sim_params.get('prio_congestion_radius', 5)

    active_agent_positions = {i: agent_positions[i] for i in active_ids}

    for aid in active_ids:
        pos = agent_positions[aid]
        goal = agent_goals[aid]
        mem = agent_memories[aid]

        # 1. 进度度量 (越高越好)
        dist_to_goal = heuristic(pos, goal)
        progress_score = w_progress / (1.0 + dist_to_goal)

        # 2. 等待时间度量 (越高越好)
        waiting_score = w_wait * mem.consecutive_no_progress_steps

        # 3. 拥堵度量 (越低越好 -> 惩罚项)
        congestion_count = 0
        for other_aid, other_pos in active_agent_positions.items():
            if aid == other_aid:
                continue
            if heuristic(pos, other_pos) <= congestion_radius:
                congestion_count += 1
        congestion_penalty = w_congestion * congestion_count
        
        # 综合优先级
        priorities[aid] = progress_score + waiting_score - congestion_penalty
        
    return priorities
    
# ... (此处包含 run_single_agent_astar 等函数的完整实现)
def get_spatial_features_from_obs(agent_obs_dict, local_window_size):
    num_spatial_channels = 4; h = w = local_window_size
    spatial_features = np.zeros((num_spatial_channels, h, w), dtype=np.float32)
    center_idx = h // 2
    spatial_features[0, :, :] = agent_obs_dict.get("obstacles", np.ones((h, w))).astype(np.float32)
    spatial_features[1, :, :] = agent_obs_dict.get("agents", np.zeros((h, w))).astype(np.float32)
    spatial_features[2, center_idx, center_idx] = 1.0
    spatial_features[3, :, :] = agent_obs_dict.get("target", np.zeros((h,w))).astype(np.float32)
    return torch.from_numpy(spatial_features)

def get_non_spatial_features(current_global_pos, global_goal_pos):
    dy = global_goal_pos[0] - current_global_pos[0]; dx = global_goal_pos[1] - current_global_pos[1]
    norm = np.sqrt(dx**2 + dy**2) if (dx**2 + dy**2) > 0 else 1.0
    norm_dy, norm_dx = dy / norm, dx / norm
    return torch.from_numpy(np.array([norm_dy, norm_dx], dtype=np.float32))

def global_to_local_coords(global_pos_target, agent_global_pos, obs_radius, window_size):
    gr_target, gc_target = global_pos_target; gr_agent, gc_agent = agent_global_pos
    lr = gr_target - (gr_agent - obs_radius); lc = gc_target - (gc_agent - obs_radius)
    if 0 <= lr < window_size and 0 <= lc < window_size: return (int(lr), int(lc))
    return None

def path_coords_to_actions(path_coords: List[Tuple[int, int]], start_pos: Tuple[int, int]) -> List[int]:
    actions = []
    current_path = [start_pos] + path_coords
    for i in range(len(current_path) - 1):
        dr = current_path[i+1][0] - current_path[i][0]
        dc = current_path[i+1][1] - current_path[i][1]
        action = next((act for act, (ddr, ddc) in ACTION_DELTAS.items() if ddr == dr and ddc == dc), ACTION_STAY)
        actions.append(action)
    return actions if actions else [ACTION_STAY]

@numba.jit(nopython=True, cache=True)
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_map_hash(grid_map: np.ndarray) -> int:
    return hash(grid_map.tobytes())

_astar_path_cache: Dict[Tuple, Optional[List[Tuple[int, int]]]] = {}
_ASTAR_CACHE_MAX_SIZE = 2048

def run_single_agent_astar(start_pos_global: Tuple[int, int], goal_pos_global: Tuple[int, int], grid_map: np.ndarray, max_path_len: int, dynamic_cost_map: Optional[np.ndarray], other_agent_positions: Optional[Set[Tuple[int, int]]], sim_params: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
    global _astar_path_cache
    map_hash = get_map_hash(grid_map)
    dyn_map_hash = get_map_hash(dynamic_cost_map) if dynamic_cost_map is not None else None
    occ_hash = hash(tuple(sorted(list(other_agent_positions)))) if other_agent_positions and len(other_agent_positions) < 64 else None

    cache_key = (start_pos_global, goal_pos_global, map_hash, dyn_map_hash, max_path_len, occ_hash)
    if cache_key in _astar_path_cache: return _astar_path_cache[cache_key]

    h, w = grid_map.shape
    if not (0 <= start_pos_global[0] < h and 0 <= start_pos_global[1] < w) or grid_map[start_pos_global] == OBSTACLE_CELL:
        return None

    final_dyn_cost = dynamic_cost_map.copy() if dynamic_cost_map is not None else np.zeros((h, w), dtype=np.float32)
    unknown_soft_cost = sim_params.get('unknown_soft_cost', 4.5)
    unknown_mask = (grid_map == UNKNOWN_CELL)
    final_dyn_cost[unknown_mask] += unknown_soft_cost

    planning_obstacle_map = (grid_map == OBSTACLE_CELL)
    if other_agent_positions:
        for r, c in other_agent_positions:
            if 0 <= r < h and 0 <= c < w: planning_obstacle_map[r, c] = 1

    open_set = [(heuristic(start_pos_global, goal_pos_global), 0, start_pos_global)]
    came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}
    g_score: Dict[Tuple[int,int], float] = {start_pos_global: 0}
    path = None
    node_limit = max_path_len * 8 
    nodes_expanded = 0

    while open_set:
        _, g, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal_pos_global:
            path_deque = deque()
            temp = current
            while temp in came_from:
                path_deque.appendleft(temp)
                temp = came_from[temp]
            path = list(path_deque)
            break
        
        if len(path or []) + g > max_path_len * 1.5 or nodes_expanded > node_limit: break

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w and not planning_obstacle_map[neighbor]): continue

            move_cost = 1 + final_dyn_cost[neighbor]
            new_g = g + move_cost
            if new_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor], g_score[neighbor] = current, new_g
                f_score = new_g + heuristic(neighbor, goal_pos_global)
                heapq.heappush(open_set, (f_score, new_g, neighbor))

    if len(_astar_path_cache) > _ASTAR_CACHE_MAX_SIZE:
        keys_to_del = random.sample(list(_astar_path_cache.keys()), len(_astar_path_cache) // 2)
        for key in keys_to_del: 
            if key in _astar_path_cache: del _astar_path_cache[key]
            
    _astar_path_cache[cache_key] = path
    return path


def _build_conflict_groups(initial_paths: Dict[int, List[Tuple[int, int]]], active_agents: List[int], max_timestep: int, agent_memories: Dict[int, AgentMemory], all_pos: List[Tuple[int, int]], sim_params: Dict) -> List[Set[int]]:
    if not active_agents: return []
    
    paths_for_detection = {aid: p for aid, p in initial_paths.items() if aid in active_agents}
    # This external function must return Conflict objects with `type` and `location1` attributes
    conflicts = detect_all_conflicts_spacetime(paths_for_detection, max_timestep)

    congestion_penalty = sim_params.get('congestion_penalty', 0.0)
    if congestion_penalty > 0 and conflicts:
        for conflict in conflicts:
            if conflict.type == Conflict.VERTEX and hasattr(conflict, 'location1') and conflict.location1:
                pos = conflict.location1
                if conflict.agent1_id in agent_memories:
                    agent_memories[conflict.agent1_id].dynamic_cost_map[pos] += congestion_penalty
                if conflict.agent2_id in agent_memories:
                    agent_memories[conflict.agent2_id].dynamic_cost_map[pos] += congestion_penalty

    G = nx.Graph()
    G.add_nodes_from(active_agents)
    for conflict in conflicts:
        if G.has_node(conflict.agent1_id) and G.has_node(conflict.agent2_id):
            G.add_edge(conflict.agent1_id, conflict.agent2_id)

    forced_agents = {aid for aid, mem in agent_memories.items() if mem.pattering_status == "FORCED_GROUP_SOLVE" and aid in active_agents}
    
    components = [set(c) for c in nx.connected_components(G)]
    if not forced_agents:
        return [c for c in components if len(c) > 0]

    final_groups = []
    merged_forced_group = set(forced_agents)
    related_components, unrelated_components = [], []
    for comp in components:
        if not comp.isdisjoint(merged_forced_group): related_components.append(comp)
        else: unrelated_components.append(comp)
    for comp in related_components: merged_forced_group.update(comp)
    final_groups.extend(unrelated_components)
    final_groups.append(merged_forced_group)
    if sim_params['verbose']: logging.info(f"Final conflict groups (with forced merge): {[list(g) for g in final_groups]}")

    return [g for g in final_groups if len(g) > 0]


def _find_retreat_goal(agent_id: int, current_pos: Tuple[int, int], persistent_known_map: np.ndarray, all_agent_positions: List[Tuple[int, int]], claimed_subgoals: Set[Tuple[int, int]], max_search_dist: int) -> Optional[Tuple[int, int]]:
    h, w = persistent_known_map.shape
    q: Deque[Tuple[Tuple[int,int], int]] = deque([(current_pos, 0)])
    visited = {current_pos}
    other_agent_locs = {p for i, p in enumerate(all_agent_positions) if i != agent_id}

    while q:
        pos, dist = q.popleft()
        if dist > 0:
            is_valid_retreat = (persistent_known_map[pos] == FREE_CELL and pos not in other_agent_locs and pos not in claimed_subgoals)
            if is_valid_retreat: return pos
        if dist >= max_search_dist: continue
        
        for dr, dc in random.sample([(-1,0), (1,0), (0,-1), (0,1)], 4):
            neighbor = (pos[0] + dr, pos[1] + dc)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and neighbor not in visited:
                visited.add(neighbor)
                if persistent_known_map[neighbor] != OBSTACLE_CELL:
                    q.append((neighbor, dist + 1))
    return None

def _find_frontier_subgoal(start_pos: Tuple[int, int], persistent_map: np.ndarray, claimed_subgoals: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    h, w = persistent_map.shape
    if persistent_map[start_pos] != FREE_CELL: return None
    
    q: Deque[Tuple[int,int]] = deque([start_pos])
    visited = {start_pos}
    frontiers = []

    while q:
        r, c = q.popleft()
        is_frontier = False
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if persistent_map[nr, nc] == UNKNOWN_CELL:
                    is_frontier = True
                elif persistent_map[nr, nc] == FREE_CELL and (nr, nc) not in visited:
                    visited.add((nr, nc)); q.append((nr, nc))
        if is_frontier:
            frontiers.append((r,c))

    unclaimed_frontiers = [p for p in frontiers if p not in claimed_subgoals]
    if not unclaimed_frontiers: return None
    return min(unclaimed_frontiers, key=lambda p: heuristic(p, start_pos))


                             
def get_spatial_features_v19_compatible(agent_obs_dict, window_size):
    """ 创建与您现有 unet 兼容的 4 通道空间输入 """
    # 0: obstacles, 1: other_agents, 2: self_pos, 3: target_hotspot
    num_spatial_channels = 4
    h = w = window_size
    spatial_features = np.zeros((num_spatial_channels, h, w), dtype=np.float32)
    center_idx = h // 2
    
    spatial_features[0, :, :] = agent_obs_dict.get("obstacles", np.ones((h, w))).astype(np.float32)
    spatial_features[1, :, :] = agent_obs_dict.get("agents", np.zeros((h, w))).astype(np.float32)
    spatial_features[2, center_idx, center_idx] = 1.0
    # target_hotspot channel is now intelligently populated before this function is called
    spatial_features[3, :, :] = agent_obs_dict.get("target", np.zeros((h,w))).astype(np.float32)
                
    return torch.from_numpy(spatial_features)


# ====================================================================
# 3. 死锁干预与级联防御策略 (稳定版本)
# ====================================================================#


def solve_by_coordinated_retreat(group_list: List[int], group_agents_data: List[Dict], persistent_known_map: np.ndarray, cbs_map_for_solver: np.ndarray, agents_global_positions: List[Tuple[int, int]], agents_global_goals: List[Tuple[int, int]], sim_params: Dict, dynamic_priorities: Dict[int, float]) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    if len(group_list) < 2: return None
    verbose = sim_params['verbose']
    
    # agents_with_dist = sorted([{'id': aid, 'dist': heuristic(tuple(agents_global_positions[aid]), tuple(agents_global_goals[aid]))} for aid in group_list], key=lambda x: x['dist'], reverse=True)
    # 新的逻辑：基于动态优先级排序，分数越低（路权越低）的越靠前
    agents_with_prio = sorted(
        [{'id': aid, 'prio': dynamic_priorities.get(aid, 0)} for aid in group_list],
        key=lambda x: x['prio'] # 默认升序，低优先级的在前
    )
    # --- V22 增强 S2.1 结束 ---
    
    num_yielders = max(1, len(group_list) // 2)
    yielder_ids, mover_ids = {d['id'] for d in agents_with_prio[:num_yielders]}, {d['id'] for d in agents_with_prio[num_yielders:]}
    if verbose: logging.info(f"Coordinated Retreat: Yielders={yielder_ids}, Movers={mover_ids}")

    yielder_planning_map = cbs_map_for_solver.copy(); 
    for mover_id in mover_ids: yielder_planning_map[tuple(agents_global_positions[mover_id])] = 1

    yielder_paths: Dict[int, List[Tuple[int,int]]] = {}
    dynamic_constraints: List[Constraint] = []
    claimed_goals: Set[Tuple[int,int]] = set()
    
    for aid in yielder_ids:
        pos = tuple(agents_global_positions[aid])
        adj_free_dirs = sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)] if 0 <= pos[0]+dr < cbs_map_for_solver.shape[0] and 0 <= pos[1]+dc < cbs_map_for_solver.shape[1] and cbs_map_for_solver[pos[0]+dr, pos[1]+dc] == 0)
        search_depth = 3 + 2 * adj_free_dirs
        
        retreat_goal = _find_retreat_goal(aid, pos, persistent_known_map, agents_global_positions, claimed_goals, search_depth)
        if not retreat_goal: logging.warning(f"Retreat failed: Yielder {aid} no goal found."); return None
        claimed_goals.add(retreat_goal)
        
        path_coords = run_single_agent_astar(pos, retreat_goal, (yielder_planning_map != FREE_CELL), sim_params['local_plan_horizon_base'], None, None, sim_params)
        if not path_coords: logging.warning(f"Retreat failed: Yielder {aid} no path."); return None
        
        full_path = [pos] + path_coords
        yielder_paths[aid] = full_path
        for t, p in enumerate(full_path):
            if t > 0:
                dynamic_constraints.append(Constraint(aid, p, t))
                dynamic_constraints.append(Constraint(aid, p, t, is_edge_constraint=True, prev_loc=full_path[t-1]))

    movers_data = [d for d in group_agents_data if d['id'] in mover_ids]
    if not movers_data: return yielder_paths

    mover_paths = solve_local_cbs_robust(
        agents_data=movers_data, obstacles_local_map=cbs_map_for_solver,
        max_plan_len=sim_params['local_plan_horizon_base'] - 1, max_cbs_iterations=int(sim_params['cbs_max_iterations'] * sim_params['sub_cbs_max_iterations_multiplier']),
        initial_constraints_list=dynamic_constraints,
        agents_true_global_goals_abs={i: tuple(g) for i, g in enumerate(agents_global_goals)},
        persistent_map_bundle={'persistent_known_map': persistent_known_map, 'map_global_origin_r': 0, 'map_global_origin_c': 0, 'FREE_CELL': FREE_CELL},
        verbose_cbs_solver=verbose)

    if not mover_paths: logging.warning("Retreat failed: CBS for movers failed."); return None
    logging.info(f"Coordinated Retreat strategy succeeded!")
    return {**yielder_paths, **mover_paths}


def solve_by_forced_shuffle(group_list, cbs_map_for_solver, agents_global_positions, verbose=False):
    if verbose: logging.critical(f"Attempting Final Defense: Forced Shuffle for group {group_list}.")
    solution_paths, shuffled_group = {}, random.sample(group_list, len(group_list))
    temp_obstacle_map = cbs_map_for_solver.copy()
    for agent_id in shuffled_group:
        current_pos = tuple(agents_global_positions[agent_id])
        possible_moves = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]; random.shuffle(possible_moves)
        move_found = False
        for dr, dc in possible_moves:
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)
            if (0 <= next_pos[0] < temp_obstacle_map.shape[0] and 0 <= next_pos[1] < temp_obstacle_map.shape[1] and temp_obstacle_map[next_pos] == 0):
                solution_paths[agent_id], temp_obstacle_map[next_pos], move_found = [current_pos, next_pos], 1, True; break
        if not move_found: solution_paths[agent_id] = [current_pos, current_pos]
    if verbose: logging.warning("Forced Shuffle complete.")
    return solution_paths

def solve_by_group_evaporation(group_list: List[int], cbs_map: np.ndarray, agents_pos: List[Tuple[int, int]], sim_params: Dict) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """【v17 新策略】大冲突组“蒸发”策略。"""
    logging.warning(f"Attempting Group Evaporation for large stuck group of size {len(group_list)}.")
    h, w = cbs_map.shape
    group_pos = {aid: agents_pos[aid] for aid in group_list}
    
    # 1. 计算冲突组的中心
    centroid_r = int(np.mean([pos[0] for pos in group_pos.values()]))
    centroid_c = int(np.mean([pos[1] for pos in group_pos.values()]))

    # 2. 通过BFS从中心向外寻找足够多的、分散的“逃生点”
    q = deque([(r, c) for r, c in group_pos.values() if 0 <= r < h and 0 <= c < w])
    visited = set(q)
    escape_pods = []
    
    while q and len(escape_pods) < len(group_list) * 1.5:
        r, c = q.popleft()
        dist_from_centroid = heuristic((r,c), (centroid_r, centroid_c))
        
        # 逃生点需要在一定距离之外，且不在任何智能体当前位置
        if dist_from_centroid > sim_params.get('evaporation_min_dist', 8) and (r,c) not in agents_pos:
            escape_pods.append((r,c))
            
        if dist_from_centroid > sim_params.get('evaporation_max_dist', 30): continue # 限制搜索范围

        for dr, dc in random.sample([(-1,0), (1,0), (0,-1), (0,1)], 4):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not cbs_map[nr, nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    
    if len(escape_pods) < len(group_list):
        logging.error(f"Evaporation failed: Not enough escape pods found ({len(escape_pods)}/{len(group_list)}).")
        return None

    # 3. 为每个智能体分配一个最近的逃生点
    solution_paths = {}
    claimed_pods = set()
    for aid in sorted(group_list, key=lambda i: heuristic(group_pos[i], (centroid_r, centroid_c))):
        best_pod = None
        min_dist = float('inf')
        for pod in escape_pods:
            if pod not in claimed_pods:
                dist = heuristic(group_pos[aid], pod)
                if dist < min_dist:
                    min_dist = dist
                    best_pod = pod
        
        if best_pod:
            claimed_pods.add(best_pod)
            path = run_single_agent_astar(group_pos[aid], best_pod, cbs_map, 50, None, None, sim_params)
            if path:
                solution_paths[aid] = [group_pos[aid]] + path
            else: # 如果找不到路径，就让它待在原地
                solution_paths[aid] = [group_pos[aid]]

    logging.info(f"Group Evaporation successful, assigning {len(solution_paths)} agents to escape pods.")
    return solution_paths

def _solve_single_group_with_defense_cascade(
    group_list: List[int], group_agents_data: List[Dict], cbs_map_for_solver: np.ndarray,
    agents_global_positions: List[Tuple[int, int]], agents_global_goals: List[Tuple[int, int]],
    consecutive_cbs_fails_count: Dict[frozenset, int], sim_params: Dict, dynamic_priorities: Dict[int, float] 
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    group_frozenset = frozenset(group_list)
    fails_count = consecutive_cbs_fails_count.get(group_frozenset, 0)
    
    # --- v18 Defense Cascade ---
    cbs_iterations = sim_params['cbs_max_iterations']
    plan_len = sim_params['n_exec_steps']
    is_high_priority = False
    
    # 策略 0: 高优先级长视界 CBS (用于顽固的小团体)
    if len(group_list) <= sim_params['hp_cbs_group_size_threshold'] and fails_count >= sim_params['hp_cbs_trigger_fails']:
        logging.warning(f"Group {group_list} is persistently stuck. Triggering High-Priority Long-Horizon CBS.")
        cbs_iterations = sim_params['hp_cbs_max_iterations']
        plan_len = sim_params['hp_cbs_plan_len']
        is_high_priority = True

    # 策略 1: 标准 CBS (或高优先级CBS)
    solution = solve_local_cbs_robust(group_agents_data, cbs_map_for_solver, max_plan_len=plan_len, max_cbs_iterations=cbs_iterations,
        initial_constraints_list=[], agents_true_global_goals_abs={i: g for i, g in enumerate(agents_global_goals)},
        persistent_map_bundle={'persistent_known_map': sim_params['persistent_known_map']}, verbose_cbs_solver=False
    )
    if solution:
        consecutive_cbs_fails_count[group_frozenset] = 0
        return solution
    
    consecutive_cbs_fails_count[group_frozenset] = fails_count + 1
    
    # --- V22.2 增强 S3.1 开始: 动态防御策略选择 ---
    group_positions = [agents_global_positions[aid] for aid in group_list]
    
    # 1. 分析冲突组几何特征
    if len(group_positions) > 1:
        mean_r = np.mean([p[0] for p in group_positions])
        mean_c = np.mean([p[1] for p in group_positions])
        centroid = (mean_r, mean_c)
        
        # 计算 "散布度"：所有点到质心的平均距离
        spread = np.mean([heuristic(p, centroid) for p in group_positions])
    else:
        spread = 0

    logging.warning(f"Group {group_list} CBS failed. Fails: {fails_count+1}. Analyzing geometry: spread={spread:.2f}")

    # 2. 根据特征选择策略
    # 策略A: 紧密聚集的小团体 (高密度死锁)
    if spread <= sim_params.get('strategy_spread_threshold_tight', 2.5) and \
       len(group_list) <= sim_params.get('push_rotate_group_size_threshold', 6):
        logging.warning("Strategy -> Tight-Knot. Attempting Push-and-Rotate.")
        solution = solve_by_push_and_rotate(group_list, cbs_map_for_solver, agents_global_positions, sim_params)
        if solution: return solution

    # 策略B: 中等规模或分散的冲突组
    logging.warning("Strategy -> Default. Attempting Coordinated Retreat.")
    solution = solve_by_coordinated_retreat(
        group_list, group_agents_data, sim_params['persistent_known_map'], 
        cbs_map_for_solver, agents_global_positions, agents_global_goals, sim_params,
        dynamic_priorities=dynamic_priorities # 传递优先级字典
    )    
    if solution: return solution

    # 策略C: 非常大或用尽其他方法的组
    if len(group_list) > sim_params.get('evaporation_group_size_threshold', 10):
        logging.warning("Strategy -> Large Group. Attempting Group Evaporation.")
        solution = solve_by_group_evaporation(group_list, cbs_map_for_solver, agents_global_positions, sim_params)
        if solution: return solution
        
    # --- V22.2 增强 S3.1 结束 ---
    
    # 最后手段: 强制洗牌
    return solve_by_forced_shuffle(group_list, cbs_map_for_solver, agents_global_positions, sim_params.get('verbose', False))





# ====================================================================
# 0. 新增的核心模块 (NEW & ENHANCED Core Modules)
# ====================================================================

class CorridorScheduler:
    """【V22.2 新增】一个简单的调度器，用于管理狭窄通道，避免对冲。"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.corridors: List[List[Tuple[int, int]]] = []
        self.corridor_indices: Dict[Tuple[int, int], int] = {}  # cell -> corridor_idx
        self.last_map_hash = None

    def _is_corridor_cell(self, r: int, c: int, grid: np.ndarray) -> bool:
        """检查一个单元格是否是走廊的一部分（自由邻居数量<=2）。"""
        if grid[r, c] != FREE_CELL:
            return False
        h, w = grid.shape
        free_dirs = 0
        # 只检查上下和左右，如果一个方向通，另一个方向堵，则很可能是通道
        # e.g., (up is free, down is blocked) or (up is blocked, down is free)
        vertical_free = (grid[min(h-1, r+1), c] == FREE_CELL) + (grid[max(0,r-1), c] == FREE_CELL)
        horizontal_free = (grid[r, min(w-1, c+1)] == FREE_CELL) + (grid[r, max(0,c-1)] == FREE_CELL)
        
        # 定义：通道是那些只在一个轴向上主要连通的单元格
        return (vertical_free >= 1 and horizontal_free <= 1) or \
               (horizontal_free >= 1 and vertical_free <= 1)

    def build_corridors(self, persistent_map: np.ndarray):
        """从持久化地图中构建所有符合条件的走廊段。"""
        current_map_hash = get_map_hash(persistent_map)
        if current_map_hash == self.last_map_hash:
            return
        
        if self.config.get('verbose', False):
            logging.info("CorridorScheduler: Map changed, rebuilding corridors...")
            
        self.last_map_hash = current_map_hash
        self.corridors = []
        self.corridor_indices = {}
        
        grid = persistent_map
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        for r in range(h):
            for c in range(w):
                if grid[r, c] == FREE_CELL and not visited[r, c] and self._is_corridor_cell(r, c, grid):
                    corridor: List[Tuple[int, int]] = []
                    q = deque([(r, c)])
                    visited[r, c] = True
                    while q:
                        curr_r, curr_c = q.popleft()
                        corridor.append((curr_r, curr_c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and \
                               grid[nr, nc] == FREE_CELL and not visited[nr, nc] and \
                               self._is_corridor_cell(nr, nc, grid):
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    
                    if len(corridor) >= self.config.get('corridor_min_len', 3):
                        corridor_idx = len(self.corridors)
                        self.corridors.append(corridor)
                        for pos in corridor:
                            self.corridor_indices[pos] = corridor_idx
        
        if self.config.get('verbose', False) and self.corridors:
            logging.info(f"CorridorScheduler: Built {len(self.corridors)} corridors.")

    def schedule_flow(self, proposed_paths: Dict[int, List[Tuple[int, int]]], 
                      dynamic_priorities: Dict[int, float], current_step: int) -> Dict[int, List[Tuple[int, int]]]:
        """为通过走廊的智能体进行贪心调度，返回成功调度的路径。"""
        if not self.corridors:
            return {}

        scheduled_paths: Dict[int, List[Tuple[int, int]]] = {}
        # (corridor_idx, time_phase) -> flow_direction (e.g., 'N' or 'S')
        phase_reservations: Dict[Tuple[int, int], str] = {}
        
        # 识别哪些agent想要进入通道
        agents_intending_corridor: List[Dict] = []
        for aid, path in proposed_paths.items():
            for t, pos in enumerate(path):
                if pos in self.corridor_indices:
                    agents_intending_corridor.append({'id': aid, 'path': path, 'entry_time': t, 
                                                      'entry_pos': pos, 'prio': dynamic_priorities.get(aid, 0)})
                    break
        
        if not agents_intending_corridor:
            return {}
            
        # 按优先级从高到低排序
        sorted_agents = sorted(agents_intending_corridor, key=lambda x: x['prio'], reverse=True)

        phase_len = self.config.get('corridor_phase_len', 10)

        for agent_data in sorted_agents:
            aid = agent_data['id']
            path = agent_data['path']
            corridor_idx = self.corridor_indices[agent_data['entry_pos']]
            
            # 确定智能体在通道内的流动方向
            entry_pos = agent_data['entry_pos']
            exit_pos = path[-1]
            for pos in reversed(path):
                if pos in self.corridor_indices:
                    exit_pos = pos
                    break
            
            flow_dir = 'E' # Default
            if abs(exit_pos[0] - entry_pos[0]) > abs(exit_pos[1] - entry_pos[1]):
                flow_dir = 'S' if exit_pos[0] > entry_pos[0] else 'N'
            else:
                flow_dir = 'E' if exit_pos[1] > entry_pos[1] else 'W'
            
            path_is_valid = True
            for t, pos in enumerate(path):
                if pos in self.corridor_indices and self.corridor_indices[pos] == corridor_idx:
                    time_phase = ((current_step + t) // phase_len)
                    
                    # 检查此时间片是否已被预定
                    if (corridor_idx, time_phase) in phase_reservations:
                        # 如果预定方向与当前agent方向不符，则路径无效
                        if phase_reservations[(corridor_idx, time_phase)] != flow_dir:
                            path_is_valid = False
                            break
                    else:
                        # 预定此时间片
                        phase_reservations[(corridor_idx, time_phase)] = flow_dir
            
            if path_is_valid:
                scheduled_paths[aid] = path
        
        if self.config.get('verbose', False) and scheduled_paths:
            logging.info(f"CorridorScheduler: Scheduled {len(scheduled_paths)} agents through corridors.")
            
        return scheduled_paths


def run_mapf_simulation(env, unet_model, device, max_episode_steps=1024, config: Optional[Dict] = None, **kwargs):
    # --- 1. 初始化 ---
    unet_model.eval(); obs_list, _ = env.reset(); num_agents = env.grid_config.num_agents
    all_pos, all_goals = list(env.get_agents_xy()), list(env.get_targets_xy())
    map_h, map_w = env.unwrapped.grid.get_obstacles().shape
    p_map = np.full((map_h, map_w), UNKNOWN_CELL, dtype=np.int8)
    obs_radius, window_size = env.grid_config.obs_radius, env.grid_config.obs_radius * 2 + 1
    
    default_config = {
        # --- V22 已有参数 ---
        'wall_follow_trigger_fails': 1,
        'wall_follow_max_steps': 20,
        'prio_w_progress': 1.0, 'prio_w_wait': 0.5,
        'prio_w_congestion': 1, 'prio_congestion_radius': 5,
        
        # --- V22.2 新增参数 ---
        # 通道调度器参数
        'corridor_min_len': 3,             # 识别为通道的最小长度
        'corridor_phase_len': 10,          # 通道交通灯的相位长度(步数)
        'trap_penalty_value': 250.0,       # 陷阱区域的基础惩罚值，应该设得很高

        # 动态策略选择参数
        'strategy_spread_threshold_tight': 2.5, # 判断为"紧密聚集"的散布度阈值
        
        # v18 继承的参数
        'hp_cbs_trigger_fails': 3, 'hp_cbs_group_size_threshold': 8, 'hp_cbs_max_iterations': 400, 'hp_cbs_plan_len': 30,
        'line_of_sight_penalty': 60.0, 'evaporation_group_size_threshold': 10, 'evaporation_min_dist': 8, 'evaporation_max_dist': 30,
        'push_rotate_group_size_threshold': 6, 'unknown_soft_cost': 12.0, 'pattering_no_progress_steps_threshold': 3,
        'pattering_history_len': 15, 'pattering_unique_pos_threshold': 14, 'pattering_progress_check_interval': 2,
        'pattering_heuristic_improvement_threshold': 1, 'escape_plan_max_len': 150, 'congestion_penalty': 15.0,
        'dynamic_cost_decay': 0.95, 'local_plan_horizon_base': 80, 'region_size': 16, 'use_arg_planner': True,
        'coarse_region_size': 64, 'use_very_coarse_arg': True, 'n_exec_steps': 5, 'cbs_max_iterations': 120,
        'cbs_time_limit_s': 300, 'max_consecutive_cbs_fails_for_intervention': 2, 'batch_unet_size': 64, 'verbose': False
    }
    if config: default_config.update(config)
    sim_params = default_config; sim_params.update(kwargs)
    sim_params.update({'persistent_known_map': p_map, 'FREE_CELL': FREE_CELL, 'OBSTACLE_CELL': OBSTACLE_CELL, 'UNKNOWN_CELL': UNKNOWN_CELL, 'map_h': map_h, 'map_w': map_w})
    
    # --- 2. 状态与规划器设置 ---
    def update_p_map(obs_list, agent_positions):
        for idx, (r, c) in enumerate(agent_positions):
            if idx < len(obs_list) and (fov_obs := obs_list[idx].get("obstacles")) is not None:
                tl_r, tl_c = r - obs_radius, c - obs_radius
                for f_r in range(window_size):
                    for f_c in range(window_size):
                        g_r, g_c = tl_r + f_r, tl_c + f_c
                        if 0 <= g_r < map_h and 0 <= g_c < map_w:
                            cell_val = fov_obs[f_r, f_c]
                            if cell_val == 1: p_map[g_r, g_c] = OBSTACLE_CELL
                            else: p_map[g_r, g_c] = FREE_CELL
    update_p_map(obs_list, all_pos)
    
    varg_planner = VeryCoarseRegionGraph((map_h, map_w), sim_params, p_map) if sim_params.get('use_very_coarse_arg', False) else None
    arg_planner = AdaptiveRegionGraph((map_h, map_w), sim_params, p_map) if sim_params.get('use_arg_planner', False) else None
    
    agent_memories = {i: AgentMemory(i, (map_h, map_w), sim_params) for i in range(num_agents)}
    
    corridor_scheduler = CorridorScheduler(sim_params) # <<<<< 新增

    consecutive_cbs_fails: Dict[frozenset, int] = defaultdict(int)
    active, paths = [True]*num_agents, {i: [tuple(all_pos[i])] for i in range(num_agents)}
    steps, success, errors, start_time = 0, True, [], time.time()

    # --- 3. 主仿真循环 ---
    while any(active) and steps < max_episode_steps:
        if time.time() - start_time > sim_params['cbs_time_limit_s']:
            success=False; errors.append(f"TIMEOUT@{steps}"); break
        active_ids = [i for i, a in enumerate(active) if a];
        if not active_ids: break
                # --- V22 增强 S2.1 开始: 计算动态优先级 ---
        dynamic_priorities = calculate_dynamic_priorities(
            active_ids, all_pos, all_goals, agent_memories, sim_params
        )
        # --- 3.1 路径提议 (v22 协同智能逻辑) ---
        for aid in active_ids: agent_memories[aid].check_and_handle_pattering(tuple(all_goals[aid]))

        proposed_paths: Dict[int, List[Tuple[int, int]]] = {}
        agents_for_unet, s_feats, ns_feats = [], [], []

        for aid in active_ids:
            pos, goal = tuple(all_pos[aid]), tuple(all_goals[aid]); mem = agent_memories[aid]
            
            # --- V22 增强 S1.1 开始: 智能规划器选择 ---
            # Case 0: 终极逃逸策略 - 绕墙走
            if mem.pattering_status == "ESCAPING_BY_WALL_FOLLOW":
                logging.warning(f"Agent {aid} is using Intelligent Wall-Following escape.")
                if len(mem.position_history) > 1:
                    prev_pos = mem.position_history[-2]
                    obstacle_map = (p_map == OBSTACLE_CELL)
                    
                    # --- V22.2 增强 S1.1 开始: 调用新版智能绕墙 ---
                    escape_path = intelligent_wall_follower(
                        start_pos=pos,
                        prev_pos=prev_pos,
                        goal_pos=goal, # <<<<<< 传递最终目标
                        obstacle_map=obstacle_map,
                        max_steps=sim_params.get('wall_follow_max_steps', 20)
                    )
                    # --- V22.2 增强 S1.1 结束 ---
                    
                    if escape_path:
                        proposed_paths[aid] = [pos] + escape_path
                        mem.reset_pattering_status_after_success(True)
                    else:
                        proposed_paths[aid] = [pos]
                        mem.reset_pattering_status_after_success(False)
                else:
                    proposed_paths[aid] = [pos]
                    mem.reset_pattering_status_after_success(False)
                continue

        for aid in active_ids:
            pos, goal = tuple(all_pos[aid]), tuple(all_goals[aid]); mem = agent_memories[aid]
            
            # --- 智能规划器选择 ---
            # Case 1: 智能体被困，启动专家 A* 逃逸
            if mem.pattering_status == "ESCAPING":
                other_agents_pos = {p for i, p in enumerate(all_pos) if i != aid and active[i]}
                path = run_single_agent_astar(pos, goal, p_map, sim_params['escape_plan_max_len'], mem.dynamic_cost_map, other_agents_pos, sim_params)
                proposed_paths[aid] = [pos] + path if path else [pos]
                mem.reset_pattering_status_after_success(path is not None and len(path) > 0)
                continue

            # Case 2 (默认): 使用 U-Net，但为其提供最好的导航信息
            target_for_unet = goal
            temp_obs = obs_list[aid]
            target_map = np.zeros((window_size, window_size), dtype=np.float32)

            goal_is_local = heuristic(pos, goal) <= obs_radius
            
            if not goal_is_local and arg_planner:
                # 目标在视野外，使用 ARG+A* 生成引导路径 "breadcrumb trail"
                subgoal = None
                if varg_planner:
                    coarse_path = varg_planner.find_high_level_path(pos, goal)
                    if coarse_path: subgoal = varg_planner.get_subgoal_from_path(coarse_path)
                if not subgoal:
                    fine_path = arg_planner.find_high_level_path(pos, goal)
                    if fine_path: subgoal = arg_planner.get_subgoal_from_path(fine_path)
                
                if subgoal:
                    target_for_unet = subgoal
                    # 在已知地图上快速规划引导路径
                    guidance_path = run_single_agent_astar(pos, subgoal, (p_map==OBSTACLE_CELL), obs_radius*3, None, None, sim_params)
                    if guidance_path:
                        # 将引导路径绘制到 target_map 上，形成梯度
                        for i, p in enumerate(guidance_path):
                            p_local = global_to_local_coords(p, pos, obs_radius, window_size)
                            if p_local:
                                value = 1.0 - (i / (len(guidance_path) * 1.5 + 1)) # 梯度值
                                target_map[p_local] = max(target_map[p_local], value)
            
            # 如果没有生成引导路径(例如目标在本地，或ARG失败)，则使用传统的单点目标
            if np.sum(target_map) == 0:
                target_local_coords = global_to_local_coords(target_for_unet, pos, obs_radius, window_size)
                if target_local_coords: target_map[target_local_coords] = 1.0
            
            temp_obs['target'] = target_map
            
            # --- 为 U-Net 准备输入 ---
            agents_for_unet.append(aid)
            s_feats.append(get_spatial_features_v19_compatible(temp_obs, window_size))
            ns_feats.append(get_non_spatial_features(pos, target_for_unet))

        # --- 批量运行 U-Net ---
        if agents_for_unet:
            batch_size = sim_params['batch_unet_size']
            for i in range(0, len(agents_for_unet), batch_size):
                batch_ids = agents_for_unet[i:i+batch_size];
                s_batch = torch.stack(s_feats[i:i+batch_size]).to(device); ns_batch = torch.stack(ns_feats[i:i+batch_size]).to(device)
                with torch.no_grad(): potentials = unet_model(s_batch, ns_batch).squeeze(1).cpu().numpy()
                for j, aid in enumerate(batch_ids):
                    pos = tuple(all_pos[aid]); obs_map = obs_list[aid].get("obstacles").astype(bool)
                    target_for_decode = global_to_local_coords(all_goals[aid], pos, obs_radius, window_size)
                    _, path_local = decode_action_sequence_refined(potentials[j], obs_map, (obs_radius, obs_radius), sim_params['n_exec_steps'] * 2, target_for_decode, None)
                    proposed_paths[aid] = [pos] + [(pos[0]-obs_radius+r, pos[1]-obs_radius+c) for r, c in path_local[1:]]


        final_paths: Dict[int, List[Tuple[int, int]]] = {}

        # --- V22.2 增强 S1.2 开始: 中观协调 - 通道调度 ---
        corridor_scheduler.build_corridors(p_map)
        # 使用动态优先级来辅助调度
        scheduled_paths = corridor_scheduler.schedule_flow(proposed_paths, dynamic_priorities, steps)
        
        # 路径被调度器接受的agent，其路径被最终确定
        final_paths.update(scheduled_paths)
        
        # 从后续冲突解决中移除这些已处理的agent
        agents_for_conflict_resolution = [aid for aid in active_ids if aid not in scheduled_paths]
        # --- V22.2 增强 S1.2 结束 ---


        # --- 3.2 冲突解决 ---
        # --- 3.2 冲突解决与路径最终确认 (v21.1 关键修复) ---
        final_paths: Dict[int, List[Tuple[int, int]]] = {}
        
        progressive_proposals = {}
        for aid in agents_for_conflict_resolution: # <<<<< 修改
            path = proposed_paths.get(aid)
            pos = tuple(all_pos[aid])
            goal = tuple(all_goals[aid]) # 使用 aid 索引
            if path and (len(set(path)) > 1 or pos == goal):
                progressive_proposals[aid] = path
        
        conflict_components = _build_conflict_groups(progressive_proposals, agents_for_conflict_resolution, sim_params['n_exec_steps'], agent_memories, all_pos, sim_params)
        all_conflicting_agents = set().union(*conflict_components)
        handled_agents = set()

        for group_set in conflict_components:
            if len(group_set) < 2: continue
            group = sorted(list(group_set))
            g_data = [{'id': aid, 'start_local': tuple(all_pos[aid]), 'goal_local': proposed_paths.get(aid, [tuple(all_pos[aid])])[-1]} for aid in group]
            # solution = _solve_single_group_with_defense_cascade(group, g_data, (p_map != FREE_CELL), all_pos, all_goals, consecutive_cbs_fails, sim_params)
            solution = _solve_single_group_with_defense_cascade(
                group, g_data, (p_map != FREE_CELL), all_pos, all_goals, 
                consecutive_cbs_fails, sim_params,
                dynamic_priorities=dynamic_priorities # <<--- 在这里传递优先级
            )            
            if solution: final_paths.update(solution)
            else: final_paths.update({aid: [tuple(all_pos[aid])] for aid in group})
            handled_agents.update(group)

        for aid in active_ids:
            if aid in handled_agents: continue
            
            proposal = proposed_paths.get(aid)
            pos = tuple(all_pos[aid])
            
            # 【v21.1 BUG FIX】在这里使用 aid 来索引 all_goals
            goal = tuple(all_goals[aid]) 

            if proposal and (len(set(proposal)) > 1 or pos == goal):
                final_paths[aid] = proposal
            else:
                logging.warning(f"Agent {aid} is conflict-free but not progressive. Forcing A* escape.")
                other_agents_pos = {p for i, p in enumerate(all_pos) if i != aid and active[i]}
                escape_path = run_single_agent_astar(pos, goal, p_map, sim_params['escape_plan_max_len'], agent_memories[aid].dynamic_cost_map, other_agents_pos, sim_params)
                
                if escape_path:
                    final_paths[aid] = [pos] + escape_path
                else:
                    logging.error(f"A* escape failed for solo agent {aid}. Agent will wait.")
                    final_paths[aid] = [pos]
            
            handled_agents.add(aid)

        # --- 3.3 路径执行 ---
        # (此部分与 v18 完全一致)
        max_path_len = 0 if not final_paths else max(len(p) for p in final_paths.values()) if final_paths else 0
        n_exec = min(sim_params['n_exec_steps'], max_path_len - 1 if max_path_len > 0 else 0)
        if n_exec <= 0: n_exec = 1
        action_sequences = {}
        for aid in active_ids:
            path = final_paths.get(aid, [tuple(all_pos[aid])])
            actions = path_coords_to_actions(path[1:], path[0])
            actions.extend([ACTION_STAY] * (n_exec - len(actions)))
            action_sequences[aid] = actions
        for k in range(n_exec):
            if not any(active): break
            step_actions = [action_sequences.get(i, [ACTION_STAY]*n_exec)[k] if active[i] else ACTION_STAY for i in range(num_agents)]
            obs_list, _, term, trunc, _ = env.step(step_actions)
            steps += 1; all_pos = list(env.get_agents_xy())
            had_truncation = False
            for i in range(num_agents):
                agent_memories[i].update_after_step(tuple(all_pos[i]), steps, tuple(all_goals[i]))
                if active[i]:
                    paths[i].append(tuple(all_pos[i]))
                    if term[i]: active[i] = False
                    if trunc[i]: active[i]=False; success=False; had_truncation=True; errors.append(f"A{i}_TRUNC@S{steps}")
            update_p_map(obs_list, all_pos)
            if arg_planner: arg_planner.update_graph_with_obstacles(p_map)
            if varg_planner: varg_planner.update_graph_with_obstacles(p_map)
            if not any(active) or steps >= max_episode_steps or had_truncation: break
        if not success: break
            
    # --- 4. 最终结果 ---
    finished = sum(1 for i in range(num_agents) if tuple(all_pos[i]) == tuple(all_goals[i]))
    return {
        "success": success and not any(active), "makespan": steps, 
        "sum_of_costs": sum(len(p)-1 for p in paths.values()), 
        "executed_paths_global": paths, "error_summary": "; ".join(errors) or "No errors.",
        "num_agents_reached_target": finished
    }
