# ====================================================================
# 文件: mapf_solver_v19_final.py
#
# v20 核心修复:
#   1. 严格的求解器分离:
#      - 复杂的级联防御策略(_solve_single_group_with_defense_cascade)
#        处理2个及以上智能体的真实冲突组。
#      - 单个智能体(其规划路径无冲突)的路径被直接采纳，不再进入
#        错误的死锁解决流程。
#   2. 明确“单独被困”智能体的处理:
#      - 单独被困的智能体现在被 AgentMemory 正确标记为 "ESCAPING"。
#      - 主循环逻辑会为这些 "ESCAPING" 状态的智能体自动调用
#        强大的A*逃逸规划器，从而解决“卡在凹槽”的问题。

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
# 0. 新增的核心模块 (NEW & ENHANCED Core Modules)
# ====================================================================
# ====================================================================
# 【新增模块】 启发管理器 (HeuristicManager)
# ====================================================================
class HeuristicManager:
    """
    动态计算和缓存基于当前已知地图的真实距离启发图。
    这是实现智能绕路的核心模块。
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.true_distance_cache: Dict[Tuple[Tuple[int, int], int], np.ndarray] = {}
        self.cache_max_size = self.config.get('heuristic_cache_max_size', 256)

    def _run_reverse_bfs(self, goal_pos: Tuple[int, int], p_map: np.ndarray) -> np.ndarray:
        """
        在给定的已知地图上，从目标点反向运行BFS，计算到所有点的真实距离。
        未知区域和障碍物都被视为不可通行。
        """
        h, w = p_map.shape
        heuristic_map = np.full((h, w), float('inf'), dtype=np.float32)

        if not (0 <= goal_pos[0] < h and 0 <= goal_pos[1] < w and p_map[goal_pos] == FREE_CELL):
            return heuristic_map # 目标点无效或不在自由空间

        q: Deque[Tuple[Tuple[int, int], int]] = deque([(goal_pos, 0)])
        heuristic_map[goal_pos] = 0

        while q:
            (r, c), dist = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and
                        p_map[nr, nc] == FREE_CELL and
                        heuristic_map[nr, nc] == float('inf')):
                    heuristic_map[nr, nc] = dist + 1
                    q.append(((nr, nc), dist + 1))
        return heuristic_map

    def get_true_distance_heuristic(self, goal_pos: Tuple[int, int], p_map: np.ndarray) -> np.ndarray:
        """
        获取一个目标点在当前已知地图上的真实距离启发图。
        优先从缓存读取，否则进行计算并存入缓存。
        """
        map_h = get_map_hash(p_map)
        cache_key = (goal_pos, map_h)

        if cache_key in self.true_distance_cache:
            return self.true_distance_cache[cache_key]
        else:
            if len(self.true_distance_cache) >= self.cache_max_size:
                # 简单FIFO缓存淘汰策略
                self.true_distance_cache.pop(next(iter(self.true_distance_cache)))

            if self.config.get('verbose', False):
                logging.debug(f"Heuristic cache miss for goal {goal_pos}. Calculating new heuristic map.")

            heuristic_map = self._run_reverse_bfs(goal_pos, p_map)
            self.true_distance_cache[cache_key] = heuristic_map
            return heuristic_map



class CorridorScheduler:
    """一个简单的贪心调度器，用于管理狭窄通道。"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.corridors: List[List[Tuple[int, int]]] = []
        self.corridor_indices: Dict[Tuple[int, int], int] = {} # cell -> corridor_idx
        self.last_map_hash = None

    def _is_corridor_cell(self, r: int, c: int, grid: np.ndarray) -> bool:
        """检查一个单元格是否是走廊的一部分（自由邻居数量<=2）。"""
        if grid[r, c] != FREE_CELL:
            return False
        h, w = grid.shape
        free_dirs = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w and grid[nr, nc] == FREE_CELL):
                continue
            free_dirs += 1
        return free_dirs <= 2

    def build_corridors(self, persistent_map: np.ndarray):
        """从地图中构建所有宽度不大于 `corridor_width_max` 的走廊段。"""
        current_map_hash = get_map_hash(persistent_map)
        if current_map_hash == self.last_map_hash:
            return
        self.last_map_hash = current_map_hash
        self.corridors = []
        self.corridor_indices = {}
        
        grid = (persistent_map == FREE_CELL).astype(np.int8)
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        for r in range(h):
            for c in range(w):
                if grid[r, c] and not visited[r, c] and self._is_corridor_cell(r, c, persistent_map):
                    corridor: List[Tuple[int, int]] = []
                    q = deque([(r, c)])
                    visited[r, c] = True
                    while q:
                        curr_r, curr_c = q.popleft()
                        corridor.append((curr_r, curr_c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] and not visited[nr, nc]:
                                if self._is_corridor_cell(nr, nc, persistent_map):
                                    visited[nr, nc] = True
                                    q.append((nr, nc))
                    
                    if len(corridor) > self.config.get('corridor_width_max', 3):
                        corridor_idx = len(self.corridors)
                        self.corridors.append(corridor)
                        for pos in corridor:
                            self.corridor_indices[pos] = corridor_idx
        
        if self.config.get('verbose', False) and self.corridors:
            logging.info(f"CorridorScheduler: Built {len(self.corridors)} corridors.")

    def schedule_flow(self, proposed_paths: Dict[int, List[Tuple[int, int]]], agent_goals: Dict[int, Tuple[int, int]], current_step: int) -> Dict[int, List[Tuple[int, int]]]:
        """为通过走廊的智能体进行贪心调度，返回成功调度的路径。"""
        if not self.corridors:
            return {}

        scheduled_paths = {}
        reserved_slots: Dict[Tuple[int, int, int], int] = {} # (r, c, t) -> agent_id
        agent_intends_corridor: Dict[int, int] = {} 
        for aid, path in proposed_paths.items():
            for pos in path:
                if pos in self.corridor_indices:
                    agent_intends_corridor[aid] = self.corridor_indices[pos]
                    break
        
        sorted_agents = sorted(
            agent_intends_corridor.keys(),
            key=lambda aid: heuristic(proposed_paths[aid][0], agent_goals[aid])
        )

        phase_len = self.config.get('corridor_phase_len', 8)

        for aid in sorted_agents:
            path = proposed_paths[aid]
            corridor_idx = agent_intends_corridor[aid]
            
            path_is_valid = True
            temp_reservations = {}
            for t, pos in enumerate(path):
                if (pos[0], pos[1], t) in reserved_slots:
                    path_is_valid = False
                    break
                
                if pos in self.corridor_indices and self.corridor_indices[pos] == corridor_idx:
                    time_phase = ((current_step + t) % (2 * phase_len)) // phase_len
                    corridor_phase = corridor_idx % 2
                    if time_phase != corridor_phase:
                        path_is_valid = False
                        break

                temp_reservations[(pos[0], pos[1], t)] = aid
            
            if path_is_valid:
                scheduled_paths[aid] = path
                reserved_slots.update(temp_reservations)

        if self.config.get('verbose', False) and scheduled_paths:
            logging.info(f"CorridorScheduler: Scheduled {len(scheduled_paths)} agents.")
            
        return scheduled_paths


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
### V18 TRANSPLANT ###
class AgentMemory:
    """ 
    智能体记忆模块，完全移植自 v18 的先进逻辑。
    实现了基于回归的徘徊检测和区域惩罚。
    """
    def __init__(self, agent_id: int, map_shape: Tuple[int, int], sim_params: Dict):
        self.agent_id = agent_id
        self.map_shape = map_shape
        self.params = sim_params
        
        self.pattering_history_len = self.params.get('pattering_history_len', 15)
        self.position_history: Deque[Tuple[int, int]] = deque(maxlen=self.pattering_history_len)
        self.heuristic_history: Deque[int] = deque(maxlen=self.pattering_history_len)

        # dynamic_cost_map 现在只用于处理“徘徊”导致的长期惩罚
        self.dynamic_cost_map = np.zeros(map_shape, dtype=np.float32)
        
        # v19 的状态名 'ESCAPING' 被 v18 的 'PATTERING' 替代，含义更清晰
        self.pattering_status = "NORMAL" # NORMAL, PATTERING
        
        self.pattering_cooldown_counter = 0 # 新增冷却计数器

    def update_after_step(self, new_pos: Tuple[int, int], current_step: int, goal_pos: Tuple[int, int]):
        self.position_history.append(new_pos)
        current_dist = heuristic(new_pos, goal_pos)
        self.heuristic_history.append(current_dist)
        ### --- 冷却机制逻辑 --- ###
        if self.pattering_cooldown_counter > 0:
            self.pattering_cooldown_counter -= 1
        ### --- 结束 --- ###
        # 成本图衰减
        self.dynamic_cost_map *= self.params['dynamic_cost_decay']
        self.dynamic_cost_map[self.dynamic_cost_map < 0.1] = 0
        
        # 如果智能体取得明显进展，重置其徘徊状态
        if self.pattering_status == "PATTERING" and len(self.heuristic_history) > 0:
            if current_dist < self.heuristic_history[0] - self.params.get('pattering_reset_threshold', 3):
                logging.debug(f"Agent {self.agent_id} made significant progress. Resetting status to NORMAL.")
                self.pattering_status = "NORMAL"

    def check_and_handle_pattering(self):
        """ 
        使用基于回归分析的智能徘徊检测。
        如果检测到徘徊，则调用“徘徊区域”惩罚机制。
        """
        ### --- 冷却机制逻辑 --- ###
        # 如果在冷却期内，直接跳过徘徊检测
        if self.pattering_cooldown_counter > 0:
            return
        ### --- 结束 --- ###
        if self.pattering_status != "NORMAL": return
        if len(self.position_history) < self.pattering_history_len: return

        # 条件1: 低探索度
        low_exploration = len(set(self.position_history)) <= self.params.get('pattering_unique_pos_threshold', 5)
        if not low_exploration: return

        # 条件2: 基于回归的趋势停滞检测
        steps_x = np.arange(len(self.heuristic_history))
        distances_y = np.array(list(self.heuristic_history))
        
        if np.std(distances_y) < 0.5:
            slope = 0.0 # 完全卡住
        else:
            try:
                slope, _ = np.polyfit(steps_x, distances_y, 1)
            except np.linalg.LinAlgError:
                slope = 0.0

        no_progress_trend = slope >= self.params.get('pattering_slope_threshold', -0.1)

        if no_progress_trend:
            logging.warning(f"LDAM: Agent {self.agent_id} detected PATTERING (slope: {slope:.2f}). Status -> PATTERING.")
            self.pattering_status = "PATTERING"
            self._penalize_pattering_zone()
            ### --- 冷却机制逻辑 --- ###
            self.pattering_cooldown_counter = self.params.get('pattering_cooldown_steps', 5) 
            ### --- 结束 --- ###

    def _penalize_pattering_zone(self):
        """
        智能“徘徊区域”惩罚。惩罚智能体最近访问过的所有独特位置。
        """
        penalty = self.params.get('pattering_zone_penalty', 100.0)
        pattering_zone = set(self.position_history)
        logging.debug(f"Agent {self.agent_id} penalizing its pattering zone of {len(pattering_zone)} cells.")
        
        for r, c in pattering_zone:
            if 0 <= r < self.map_shape[0] and 0 <= c < self.map_shape[1]:
                self.dynamic_cost_map[r, c] += penalty
### END V18 TRANSPLANT ###


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
    # ====================================================================
    # 【新增方法】 获取经过验证的子目标 (get_validated_subgoal_from_path)
    # ====================================================================
    def get_validated_subgoal_from_path(self, region_path: List[int], p_map: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        获取、验证并可能修正高层路径中的下一个子目标。
        确保子目标在当前已知的自由空间内。
        """
        if not region_path or len(region_path) < 2:
            return None

        target_region_id = region_path[1]
        candidate_subgoal = self.region_centers.get(target_region_id)

        if not candidate_subgoal:
            return None

        # 步骤 1 & 2: 获取并验证
        if p_map[candidate_subgoal] == FREE_CELL:
            return candidate_subgoal # 运气好，中心点就是有效的

        # 步骤 3: 修正无效子目标
        # 在目标区域内，从中心点开始BFS，寻找最近的有效点
        if self.config.get('verbose'):
            logging.debug(f"Subgoal {candidate_subgoal} in region {target_region_id} is invalid. Searching for alternative...")

        q: Deque[Tuple[int, int]] = deque([candidate_subgoal])
        visited = {candidate_subgoal}
        
        start_r, start_c = (target_region_id // self.regions_w) * self.region_size, (target_region_id % self.regions_w) * self.region_size
        end_r, end_c = start_r + self.region_size, start_c + self.region_size

        while q:
            r, c = q.popleft()
            for dr, dc in random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], 4):
                nr, nc = r + dr, c + dc
                # 确保搜索不会越界到其他区域
                if not (start_r <= nr < end_r and start_c <= nc < end_c):
                    continue
                # 确保搜索不会越出地图边界
                if not (0 <= nr < self.map_h and 0 <= nc < self.map_w):
                    continue
                
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    if p_map[nr, nc] == FREE_CELL:
                        if self.config.get('verbose'):
                            logging.debug(f"Found valid alternative subgoal at {(nr, nc)}.")
                        return (nr, nc) # 找到了！
                    q.append((nr, nc))

        if self.config.get('verbose'):
            logging.warning(f"Could not find any valid subgoal in region {target_region_id}.")
        return None # 整个区域都没有已知的自由空间





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

### V18 TRANSPLANT ###
def run_single_agent_astar(
    start_pos_global: Tuple[int, int], 
    goal_pos_global: Tuple[int, int], 
    grid_map: np.ndarray, 
    max_path_len: int, 
    persistent_cost_map: Optional[np.ndarray], 
    temporary_cost_map: Optional[Dict[Tuple[int, int], float]], # 接收临时冲突成本
    other_agent_positions: Optional[Set[Tuple[int, int]]], 
    sim_params: Dict[str, Any],
    heuristic_map: Optional[np.ndarray] = None
) -> Optional[List[Tuple[int, int]]]:
    """
    A* 规划器，移植自 v18，可以接收并组合持久化和临时成本。
    """
    h, w = grid_map.shape
    
    # 组合多种成本来源
    final_dyn_cost = persistent_cost_map.copy() if persistent_cost_map is not None else np.zeros((h, w), dtype=np.float32)
    if temporary_cost_map:
        for (r, c), cost in temporary_cost_map.items():
            if 0 <= r < h and 0 <= c < w:
                final_dyn_cost[r, c] += cost
    if 0 <= start_pos_global[0] < h and 0 <= start_pos_global[1] < w:
        final_dyn_cost[start_pos_global] = 0.0

    if not (0 <= start_pos_global[0] < h and 0 <= start_pos_global[1] < w) or grid_map[start_pos_global] == OBSTACLE_CELL:
        return None

    unknown_soft_cost = sim_params.get('unknown_soft_cost', 4.5)
    final_dyn_cost[grid_map == UNKNOWN_CELL] += unknown_soft_cost

    planning_obstacle_map = (grid_map == OBSTACLE_CELL)
    if other_agent_positions:
        for r, c in other_agent_positions:
            if 0 <= r < h and 0 <= c < w: planning_obstacle_map[r, c] = 1

    # 【核心修改】选择启发函数
    if heuristic_map is not None:
        # 使用传入的真实距离启发图
        get_h_score = lambda pos: heuristic_map[pos]
    else:
        # 回退到曼哈顿距离
        get_h_score = lambda pos: heuristic(pos, goal_pos_global)

    initial_h = get_h_score(start_pos_global)
    if initial_h == float('inf'):
         # 根据启发图，目标点从起点开始就是不可达的
        return None

    open_set = [(initial_h, 0, start_pos_global)] # (f_score, g_score, pos)
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

        if (path and len(path) > max_path_len) or nodes_expanded > node_limit: break

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w and not planning_obstacle_map[neighbor]): continue

            move_cost = 1 + final_dyn_cost[neighbor]

            new_g = g + move_cost
            if new_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor], g_score[neighbor] = current, new_g
                h_val = get_h_score(neighbor)
                if h_val == float('inf'):
                    continue # 如果邻居在启发图上不可达，则剪枝

                f_score = new_g + h_val
                heapq.heappush(open_set, (f_score, new_g, neighbor))

    return path
    
### END V18 TRANSPLANT ###


### V18 TRANSPLANT ###
def _build_conflict_groups(
    initial_paths: Dict[int, List[Tuple[int, int]]], 
    active_agents: List[int], 
    max_timestep: int, 
    sim_params: Dict
) -> Tuple[List[Set[int]], Dict[Tuple[int, int], float]]:
    """
    重构后的冲突组构建函数，移植自 v18。
    不再修改 AgentMemory，而是返回一个临时的冲突成本字典。
    """
    if not active_agents: 
        return [], {}
    
    paths_for_detection = {aid: p for aid, p in initial_paths.items() if aid in active_agents}
    conflicts = detect_all_conflicts_spacetime(paths_for_detection, max_timestep)

    temporary_conflict_costs: Dict[Tuple[int, int], float] = defaultdict(float)
    congestion_penalty = sim_params.get('congestion_penalty', 25.0)
    
    G = nx.Graph()
    G.add_nodes_from(active_agents)
    for conflict in conflicts:
        if G.has_node(conflict.agent1_id) and G.has_node(conflict.agent2_id):
            G.add_edge(conflict.agent1_id, conflict.agent2_id)
        # 为顶点冲突增加临时惩罚
        if conflict.type == Conflict.VERTEX and hasattr(conflict, 'location1') and conflict.location1:
            pos = conflict.location1
            temporary_conflict_costs[pos] += congestion_penalty
    
    components = [set(c) for c in nx.connected_components(G) if len(c) > 0]
    
    # 返回冲突组件和新建的临时成本字典
    return components, temporary_conflict_costs
### END V18 TRANSPLANT ###

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
    """ unet 4 通道空间输入 """
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
# ====================================================================

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
            
            # path_to_empty = run_single_agent_astar(agent_start_pos, current_empty, cbs_map_for_solver, 50,None, None, other_agents, sim_params)
            # --- FIX 1: Push & Rotate ---
            # 收集其他未移动智能体的位置
            other_agents_in_group_pos = {
                tuple(solution_paths[other_aid][-1]) 
                for other_aid in group_list 
                if other_aid != closest_agent_id and other_aid not in moved_agents
            }
            
            # 正确的调用
            path_to_empty = run_single_agent_astar(
                agent_start_pos, 
                current_empty, 
                cbs_map_for_solver, 
                50, 
                None,                          # persistent_cost_map
                None,                          # temporary_cost_map
                other_agents_in_group_pos,     # 正确传入需要避开的智能体
                sim_params
            )

            
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


def solve_by_coordinated_retreat(group_list: List[int], group_agents_data: List[Dict], persistent_known_map: np.ndarray, cbs_map_for_solver: np.ndarray, agents_global_positions: List[Tuple[int, int]], agents_global_goals: List[Tuple[int, int]], sim_params: Dict) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    if len(group_list) < 2: return None
    verbose = sim_params['verbose']
    agents_with_dist = sorted([{'id': aid, 'dist': heuristic(tuple(agents_global_positions[aid]), tuple(agents_global_goals[aid]))} for aid in group_list], key=lambda x: x['dist'], reverse=True)
    num_yielders = max(1, len(group_list) // 2)
    yielder_ids, mover_ids = {d['id'] for d in agents_with_dist[:num_yielders]}, {d['id'] for d in agents_with_dist[num_yielders:]}
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
        
        # path_coords = run_single_agent_astar(pos, retreat_goal, (yielder_planning_map != FREE_CELL), sim_params['local_plan_horizon_base'],None, None, None, sim_params)
        # --- FIX 2: Coordinated Retreat ---
        # 收集除当前退让者外的所有智能体位置
        other_agents_on_map_pos = {
            tuple(p) for i, p in enumerate(agents_global_positions) if i != aid
        }
        
        # 正确的调用
        path_coords = run_single_agent_astar(
            pos, 
            retreat_goal, 
            cbs_map_for_solver, # 使用原始地图，A*内部会处理动态障碍
            sim_params['local_plan_horizon_base'], 
            None,                      # persistent_cost_map
            None,                      # temporary_cost_map
            other_agents_on_map_pos,   # 正确传入需要避开的智能体
            sim_params
        )

        
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
            # path = run_single_agent_astar(group_pos[aid], best_pod, cbs_map, 50,None, None, None, sim_params)

            # --- FIX 3: Group Evaporation ---
            # 收集除当前蒸发者外的所有智能体位置
            other_agents_on_map_pos = {
                tuple(p) for i, p in enumerate(agents_pos) if i != aid
            }
            
            # 正确的调用
            path = run_single_agent_astar(
                group_pos[aid], 
                best_pod, 
                cbs_map, 
                50, 
                None,                       # persistent_cost_map
                None,                       # temporary_cost_map
                other_agents_on_map_pos,    # 正确传入需要避开的智能体
                sim_params
            )

    
            if path:
                solution_paths[aid] = [group_pos[aid]] + path
            else: # 如果找不到路径，就让它待在原地
                solution_paths[aid] = [group_pos[aid]]

    logging.info(f"Group Evaporation successful, assigning {len(solution_paths)} agents to escape pods.")
    return solution_paths

def _solve_single_group_with_defense_cascade(
    group_list: List[int], group_agents_data: List[Dict], cbs_map_for_solver: np.ndarray,
    agents_global_positions: List[Tuple[int, int]], agents_global_goals: List[Tuple[int, int]],
    consecutive_cbs_fails_count: Dict[frozenset, int], sim_params: Dict
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
    
    # 策略 2: 大团体蒸发
    if len(group_list) > sim_params['evaporation_group_size_threshold']:
        solution = solve_by_group_evaporation(group_list, cbs_map_for_solver, agents_global_positions, sim_params)
        if solution: return solution
        
    # 策略 3: 协同后撤
    solution = solve_by_coordinated_retreat(group_list, group_agents_data, sim_params['persistent_known_map'], cbs_map_for_solver, agents_global_positions, agents_global_goals, sim_params)
    if solution: return solution
    
    # 策略 4: 推挤旋转 (仅用于微小团体)
    if len(group_list) <= sim_params['push_rotate_group_size_threshold']:
        solution = solve_by_push_and_rotate(group_list, cbs_map_for_solver, agents_global_positions, sim_params)
        if solution: return solution
    
    # 最后手段: 强制洗牌
    return solve_by_forced_shuffle(group_list, cbs_map_for_solver, agents_global_positions, sim_params.get('verbose', False))

### --- 新增的智能逃逸目标搜索函数 --- ###
def _find_best_escape_subgoal(
    start_pos: Tuple[int, int], 
    goal_pos: Tuple[int, int], 
    persistent_map: np.ndarray, 
    claimed_subgoals: Set[Tuple[int, int]],
    max_search_radius: int = 20  # 限制搜索范围，防止性能开销过大
) -> Optional[Tuple[int, int]]:
    """
    寻找一个最佳的逃逸子目标。
    "最佳"定义为 f_score = g_score(从起点到该点) + h_score(从该点到终点) 最小的前沿点。
    """
    h, w = persistent_map.shape
    if persistent_map[start_pos] != FREE_CELL: return None
    
    q: Deque[Tuple[int, Tuple[int, int]]] = deque([(0, start_pos)]) # (cost_from_start, pos)
    visited = {start_pos}
    
    potential_frontiers = []

    # 使用BFS/Dijkstra寻找一定范围内的所有前沿点
    while q:
        cost, pos = q.popleft()
        
        if cost >= max_search_radius:
            continue

        is_frontier = False
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                if persistent_map[nr, nc] == UNKNOWN_CELL:
                    is_frontier = True
                elif persistent_map[nr, nc] == FREE_CELL and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((cost + 1, (nr, nc)))
        
        if is_frontier and pos not in claimed_subgoals:
            # 计算该前沿点的 f_score
            g_score = cost + 1
            h_score = heuristic(pos, goal_pos)
            f_score = g_score + h_score
            potential_frontiers.append((f_score, pos))

    if not potential_frontiers:
        return None

    # 返回 f_score 最低的前沿点
    potential_frontiers.sort(key=lambda x: x[0])
    return potential_frontiers[0][1]



# ====================================================================
# 4. 主仿真循环 (v19 "Synergistic Intelligence" Logic)
# ====================================================================
def run_mapf_simulation(env, unet_model, device, max_episode_steps=1024, config: Optional[Dict] = None, **kwargs):
    # --- 1. 初始化 ---
    unet_model.eval(); obs_list, _ = env.reset(); num_agents = env.grid_config.num_agents
    all_pos, all_goals = list(env.get_agents_xy()), list(env.get_targets_xy())
    map_h, map_w = env.unwrapped.grid.get_obstacles().shape
    p_map = np.full((map_h, map_w), UNKNOWN_CELL, dtype=np.int8)
    obs_radius, window_size = env.grid_config.obs_radius, env.grid_config.obs_radius * 2 + 1
    
    default_config = {
        'pattering_history_len': 12,
        'pattering_unique_pos_threshold': 10,
        'pattering_slope_threshold': -0.1,
        'pattering_zone_penalty': 300.0,
        'pattering_reset_threshold': 3,
        'congestion_penalty': 25.0,
        'dynamic_cost_decay': 0.98,
        'escape_plan_max_len': 120,
        'unknown_soft_cost': 6,
        'pattering_cost_threshold_for_unet': 20.0, # 新增：徘徊成本多高时U-Net视其为障碍
        'n_exec_steps': 8,
        'hp_cbs_trigger_fails': 3, 'hp_cbs_group_size_threshold': 8, 'hp_cbs_max_iterations': 300, 'hp_cbs_plan_len': 100,
        'line_of_sight_penalty': 60.0, 'evaporation_group_size_threshold': 10, 'evaporation_min_dist': 8, 'evaporation_max_dist': 30,
        'push_rotate_group_size_threshold': 6,  'pattering_no_progress_steps_threshold': 3,
        'pattering_progress_check_interval': 2,
        'pattering_heuristic_improvement_threshold': 1, 
        'local_plan_horizon_base': 80, 'region_size': 16, 'use_arg_planner': True,
        'coarse_region_size': 64, 'use_very_coarse_arg': True, 'n_exec_steps': 7, 'cbs_max_iterations': 120,
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
    heuristic_manager = HeuristicManager(sim_params)

    varg_planner = VeryCoarseRegionGraph((map_h, map_w), sim_params, p_map) if sim_params.get('use_very_coarse_arg', False) else None
    arg_planner = AdaptiveRegionGraph((map_h, map_w), sim_params, p_map) if sim_params.get('use_arg_planner', False) else None
    agent_memories = {i: AgentMemory(i, (map_h, map_w), sim_params) for i in range(num_agents)}
    consecutive_cbs_fails: Dict[frozenset, int] = defaultdict(int)
    active, paths = [True]*num_agents, {i: [tuple(all_pos[i])] for i in range(num_agents)}
    steps, success, errors, start_time = 0, True, [], time.time()

    ### V18 TRANSPLANT ###
    # 临时冲突成本图，每轮规划前重置
    temporary_conflict_costs: Dict[Tuple[int, int], float] = {}
    ### END V18 TRANSPLANT ###

    # --- 3. 主仿真循环 ---
    while any(active) and steps < max_episode_steps:
        if time.time() - start_time > sim_params['cbs_time_limit_s']:
            success=False; errors.append(f"TIMEOUT@{steps}"); break
        active_ids = [i for i, a in enumerate(active) if a];
        if not active_ids: break
        
        # --- 3.1 徘徊检测  ---
        for aid in active_ids: 
            agent_memories[aid].check_and_handle_pattering()

        # --- 3.2 路径提议  ---
        proposed_paths: Dict[int, List[Tuple[int, int]]] = {}
        agents_for_unet, s_feats, ns_feats = [], [], []

        for aid in active_ids:
            pos, goal = tuple(all_pos[aid]), tuple(all_goals[aid])
            mem = agent_memories[aid]
            claimed_subgoals_this_step: Set[Tuple[int,int]] = set()
            # Case 1: 智能体处于徘徊状态，启动专家 A* 逃逸 
            if mem.pattering_status == "PATTERING" and steps > (max_episode_steps/2):
                # 尝试寻找一个“前沿”子目标作为逃逸点
                # claimed_subgoals_this_step 是一个需要在此规划循环开始时定义的空集合: set()
                #escape_target = _find_frontier_subgoal(pos, p_map, claimed_subgoals_this_step)
                escape_target = _find_best_escape_subgoal(pos, goal, p_map, claimed_subgoals_this_step)
                other_agents_pos = {p for i, p in enumerate(all_pos) if i != aid and active[i]}

                # 如果找到了逃逸点，就用它；否则，回退到使用最终目标
                target_for_astar = escape_target if escape_target else goal
                if escape_target:
                    claimed_subgoals_this_step.add(escape_target) # 避免多个智能体冲向同一个出口
                

                # 【核心修改】为 A* 获取最佳启发图
                h_map_for_escape = heuristic_manager.get_true_distance_heuristic(target_for_astar, p_map)

                path = run_single_agent_astar(
                    pos,
                    target_for_astar,
                    p_map,
                    sim_params['escape_plan_max_len'],
                    mem.dynamic_cost_map,
                    temporary_conflict_costs,
                    other_agents_pos,
                    sim_params,
                    heuristic_map=h_map_for_escape # <-- 传入启发图
                )

                if path:
                    proposed_paths[aid] = [pos] + path
                    logging.debug(f"Agent {aid} (PATTERING) found an A* escape path.")
                    continue  # <--- 只有在成功时才 continue
            
                # 如果A*失败了，给U-Net一个机会
                else:
                    logging.warning(f"Agent {aid} (PATTERING) failed A* escape. Falling back to U-Net.")


            # Case 2 (默认): 使用 U-Net，但为其提供最好的导航信息
            target_for_unet = goal
            temp_obs = obs_list[aid]
            target_map = np.zeros((window_size, window_size), dtype=np.float32)

            goal_is_local = heuristic(pos, goal) <= obs_radius
            
            if not goal_is_local and arg_planner:
                subgoal = None
                if varg_planner:
                    coarse_path = varg_planner.find_high_level_path(pos, goal)
                    if coarse_path:
                        # 【核心修改】使用验证过的子目标
                        subgoal = varg_planner.get_validated_subgoal_from_path(coarse_path, p_map)
                if not subgoal:
                    fine_path = arg_planner.find_high_level_path(pos, goal)
                    if fine_path:
                        # 【核心修改】使用验证过的子目标
                        subgoal = arg_planner.get_validated_subgoal_from_path(fine_path, p_map)

                if subgoal:
                    target_for_unet = subgoal
                    
                    # 【核心修改】为引导路径的A*也使用最佳启发图
                    h_map_for_guidance = heuristic_manager.get_true_distance_heuristic(subgoal, p_map)
                    
                    guidance_path = run_single_agent_astar(
                        pos,
                        subgoal,
                        p_map,
                        obs_radius * 3,
                        None, None, None,
                        sim_params,
                        heuristic_map=h_map_for_guidance # <-- 传入启发图
                    )
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
            
            temp_obs = obs_list[aid].copy()
            
            ### V18 TRANSPLANT ###
            # 将所有动态成本“绘制”到U-Net的障碍物输入上
            # 1. 复制原始障碍物图
            unet_obstacles = temp_obs.get("obstacles", np.ones((window_size, window_size))).astype(np.float32)

            # 2. 绘制智能体自身的徘徊成本图
            tl_r, tl_c = pos[0] - obs_radius, pos[1] - obs_radius
            for r_local in range(window_size):
                for c_local in range(window_size):
                    r_global, c_global = tl_r + r_local, tl_c + c_local
                    if 0 <= r_global < map_h and 0 <= c_global < map_w:
                        if mem.dynamic_cost_map[r_global, c_global] > sim_params['pattering_cost_threshold_for_unet']:
                             unet_obstacles[r_local, c_local] = 1.0 # 视为障碍

            # 3. 绘制全局的即时冲突成本图
            for (r_conflict, c_conflict), penalty in temporary_conflict_costs.items():
                lr, lc = r_conflict - tl_r, c_conflict - tl_c
                if 0 <= lr < window_size and 0 <= lc < window_size:
                    unet_obstacles[lr, lc] = 1.0 # 冲突点视为障碍
            
            temp_obs['obstacles'] = unet_obstacles

            ### END V18 TRANSPLANT ###
            
            target_map = np.zeros((window_size, window_size), dtype=np.float32)
            target_local_coords = global_to_local_coords(target_for_unet, pos, obs_radius, window_size)
            if target_local_coords: target_map[target_local_coords] = 1.0
            temp_obs['target'] = target_map
            
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

        # --- 3.3 冲突解决与路径最终确认 (使用 v18 成本生成逻辑) ---
        final_paths: Dict[int, List[Tuple[int, int]]] = {}
        
        progressive_proposals = {aid: p for aid, p in proposed_paths.items() if p and (len(set(p)) > 1 or tuple(all_pos[aid]) == tuple(all_goals[aid]))}
        
        ### V18 TRANSPLANT ###
        # 检测冲突并生成 *下一轮* 使用的临时成本图
        conflict_components, new_temporary_costs = _build_conflict_groups(
            progressive_proposals, active_ids, sim_params['n_exec_steps'], sim_params
        )
        temporary_conflict_costs = new_temporary_costs # 更新主循环的临时成本，供下一轮规划使用
        ### END V18 TRANSPLANT ###
        handled_agents = set()


        for group_set in conflict_components:
            if len(group_set) < 2: continue
            group = sorted(list(group_set))
            g_data = [{'id': aid, 'start_local': tuple(all_pos[aid]), 'goal_local': proposed_paths.get(aid, [tuple(all_pos[aid])])[-1]} for aid in group]
            solution = _solve_single_group_with_defense_cascade(group, g_data, (p_map != FREE_CELL), all_pos, all_goals, consecutive_cbs_fails, sim_params)
            
            if solution: final_paths.update(solution)
            else: final_paths.update({aid: [tuple(all_pos[aid])] for aid in group})
            handled_agents.update(group)

        for aid in active_ids:
            if aid in handled_agents: continue
            
            proposal = proposed_paths.get(aid)
            pos = tuple(all_pos[aid]); goal = tuple(all_goals[aid])

            if proposal and (len(set(proposal)) > 1 or pos == goal):
                final_paths[aid] = proposal
            else:
                # 无冲突但无进取心 -> 单独被困！强制调用专家A*逃逸
                logging.warning(f"Agent {aid} is conflict-free but not progressive. Forcing A* escape.")
                other_agents_pos = {p for i, p in enumerate(all_pos) if i != aid and active[i]}
                # 这里的 A* 调用也使用新签名
                h_map_for_solo_escape = heuristic_manager.get_true_distance_heuristic(goal, p_map)
                escape_path = run_single_agent_astar(
                    pos, goal, p_map, sim_params['escape_plan_max_len'],
                    agent_memories[aid].dynamic_cost_map,
                    temporary_conflict_costs,
                    other_agents_pos, sim_params,
                    heuristic_map=h_map_for_solo_escape # <-- 传入启发图
                )
                
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
