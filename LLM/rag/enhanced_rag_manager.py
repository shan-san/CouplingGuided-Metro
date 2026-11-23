"""
Enhanced RAG Manager for Metro Dispatching

利用检索增强生成匹配历史相似状态、维护策略与调度案例，生成初始解与经验约束。

核心功能：
1. 向量化状态表示 - 将调度状态编码为稠密向量
2. 相似状态检索 - 快速匹配历史相似场景
3. 策略案例库 - 维护成功调度案例及其效果
4. 经验约束生成 - 从历史案例提取可复用的约束

支持：
- FAISS/ChromaDB 向量数据库（可选）
- SentenceTransformers 文本嵌入（可选）
- 基于规则的启发式匹配（默认）
"""

import json
import os
import uuid
import hashlib
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class StateEncoder:
    """状态编码器：将调度状态转换为向量表示"""
    
    def __init__(
        self,
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ):
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self._encoder = None
        
        if use_embeddings:
            self._init_encoder()
    
    def _init_encoder(self):
        """初始化文本嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            try:
                self._encoder = SentenceTransformer(self.embedding_model)
                logger.info(f"已加载嵌入模型: {self.embedding_model}")
            except Exception as first_error:
                cached_model = self._find_cached_sentence_transformer(self.embedding_model)
                if cached_model is None:
                    raise first_error
                self._encoder = SentenceTransformer(str(cached_model))
                logger.info(f"已从本地缓存加载嵌入模型: {cached_model}")
        except ImportError as e:
            logger.warning(f"导入 sentence_transformers 失败，使用启发式编码: {e}")
            self.use_embeddings = False
        except Exception as e:
            logger.warning(f"加载嵌入模型失败: {e}，使用启发式编码")
            self.use_embeddings = False

    @staticmethod
    def _find_cached_sentence_transformer(model_name: str) -> Optional[Path]:
        """Find a HuggingFace snapshot for offline SentenceTransformer loading."""
        if os.path.exists(model_name):
            return Path(model_name)

        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        candidate_cache_names = ["models--" + model_name.replace("/", "--")]
        if "/" not in model_name:
            candidate_cache_names.append(f"models--sentence-transformers--{model_name}")

        for model_cache_name in candidate_cache_names:
            snapshots_dir = cache_root / "hub" / model_cache_name / "snapshots"
            if not snapshots_dir.exists():
                continue

            for snapshot in sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if (
                    snapshot.is_dir()
                    and (snapshot / "modules.json").exists()
                    and (snapshot / "config.json").exists()
                    and ((snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists())
                ):
                    return snapshot
        return None
    
    def encode_state(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """将状态字典编码为向量
        
        Args:
            state_dict: 包含以下字段的状态字典:
                - n_agents: 列车数量
                - n_nodes: 节点数量
                - disruptions: 中断信息列表
                - agent_positions: 列车位置列表
                - waiting_passengers: 站台等待人数
                - high_wait_nodes: 高等待站点
        
        Returns:
            np.ndarray: 状态向量
        """
        if state_dict.get("dispatch_state") and not self.use_embeddings:
            return self._encode_native(state_dict)
        if self.use_embeddings and self._encoder is not None:
            return self._encode_with_embeddings(state_dict)
        else:
            return self._encode_heuristic(state_dict)

    def _encode_native(self, state):
        """44 fixed dimensions, without source-specific train/node indices."""
        from delay_events import EVENT_TARGETS
        from dispatch_plan import PARAMETERS
        dispatch = state["dispatch_state"]
        coupling = state.get("coupling", {})
        events = dispatch.get("events", state.get("disruptions", []))
        families = coupling.get("covered_action_families", [])
        values = [float(any(e["type"] == kind and e.get("active", True) for e in events)) for kind in sorted(EVENT_TARGETS)]
        values += [float(kind in families) for kind in PARAMETERS]
        evidence = coupling.get("evidence", [])
        values += [float(any(code in e.get("relations", ()) and e.get("active", False)
                             for e in evidence)) for code in ("C1", "C2", "C3", "C4")]
        vehicles = list(dispatch["vehicles"].values())
        n = max(1, len(vehicles))
        waiting = sum(state.get("waiting_passengers", {}).values())
        outside = sum(sum(q.get("outside", {}).values()) for q in state.get("queues", {}).values())
        gaps = sum(sum(row.values()) for row in coupling.get("current_od_gap", {}).values())
        values += [min(1., n/100.), min(1., len(dispatch.get("operations", {}))/n),
            sum(bool(v["failed"]) for v in vehicles)/n, sum(v["depot"] is not None for v in vehicles)/n,
            min(1., waiting/(1000*n)), min(1., outside/(1000*n)), min(1., gaps/(1000*n))]
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _normalize_segment_id(segment: Any) -> str:
        if segment is None:
            return ""
        if isinstance(segment, (tuple, list)) and len(segment) >= 2:
            try:
                return f"{int(segment[0])}-{int(segment[1])}"
            except (TypeError, ValueError):
                return ""
        text = str(segment).strip()
        if not text:
            return ""
        match = hashlib.md5(text.encode("utf-8")).hexdigest()
        if "-" in text:
            return text
        return match[:8]

    def _extract_disruption_descriptors(self, state_dict: Dict[str, Any]) -> List[str]:
        descriptors: List[str] = []
        for disruption in state_dict.get("disruptions", []) or []:
            segment_id = self._normalize_segment_id(
                disruption.get("segment_id", disruption.get("segment", ""))
            )
            dtype = str(disruption.get("type", "UNKNOWN")).upper()
            severity = float(disruption.get("severity", 1.0) or 1.0)
            affected_nodes = [
                str(int(node_id))
                for node_id in disruption.get("affected_nodes", []) or []
                if isinstance(node_id, (int, float))
            ]
            descriptor = f"segment={segment_id}|type={dtype}|severity={severity:.2f}"
            if affected_nodes:
                descriptor += f"|affected={'/'.join(affected_nodes[:4])}"
            descriptors.append(descriptor)
        return descriptors

    def _extract_high_wait_descriptors(self, state_dict: Dict[str, Any]) -> List[str]:
        descriptors: List[str] = []
        high_wait_nodes = state_dict.get("high_wait_nodes", []) or []
        for item in high_wait_nodes[:5]:
            if isinstance(item, dict):
                node_idx = item.get("node_idx", item.get("node_id", -1))
                waiting = item.get("waiting", 0)
            else:
                node_idx = item
                waiting = state_dict.get("waiting_passengers", {}).get(item, 0)
            try:
                descriptors.append(f"node={int(node_idx)}|waiting={int(waiting)}")
            except (TypeError, ValueError):
                continue
        return descriptors

    def _extract_agent_status_descriptors(self, state_dict: Dict[str, Any]) -> List[str]:
        descriptors: List[str] = []
        agent_positions = state_dict.get("agent_positions", []) or []
        for agent_idx, info in enumerate(agent_positions[:10]):
            if not isinstance(info, dict):
                continue
            try:
                node_id = int(info.get("node_id", info.get("position", -1)))
            except (TypeError, ValueError):
                node_id = -1
            dwell_time = float(info.get("dwell_time", 0.0) or 0.0)
            expected_dwell = float(info.get("expected_dwell", 0.0) or 0.0)
            passengers = int(info.get("passengers", 0) or 0)
            if expected_dwell > 0 and dwell_time >= expected_dwell + 5.0:
                descriptors.append(
                    f"agent={agent_idx}|node={node_id}|over_dwell={dwell_time:.1f}/{expected_dwell:.1f}|passengers={passengers}"
                )
        return descriptors
    
    def _encode_with_embeddings(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """使用文本嵌入编码状态"""
        # 构建状态描述文本
        text_parts = []
        
        # 规模信息
        n_agents = state_dict.get("n_agents", 0)
        n_nodes = state_dict.get("n_nodes", 0)
        text_parts.append(f"{n_agents} trains, {n_nodes} nodes")
        
        # 中断信息
        disruptions = state_dict.get("disruptions", [])
        if disruptions:
            disruption_types = defaultdict(int)
            for d in disruptions:
                dtype = d.get("type", "UNKNOWN")
                disruption_types[dtype] += 1
            text_parts.append(f"disruptions: {dict(disruption_types)}")
            descriptors = self._extract_disruption_descriptors(state_dict)
            if descriptors:
                text_parts.append("disruption_signatures: " + "; ".join(descriptors[:6]))
        else:
            text_parts.append("no disruptions")
        
        # 等待人数分布
        waiting = state_dict.get("waiting_passengers", {})
        if waiting:
            total_waiting = sum(waiting.values())
            max_waiting = max(waiting.values()) if waiting else 0
            text_parts.append(f"waiting: total={total_waiting}, max={max_waiting}")
        high_wait_descriptors = self._extract_high_wait_descriptors(state_dict)
        if high_wait_descriptors:
            text_parts.append("high_wait_nodes: " + "; ".join(high_wait_descriptors))
        
        # 列车分布
        positions = state_dict.get("agent_positions", [])
        at_platform = sum(1 for p in positions if p.get("at_platform", False))
        text_parts.append(f"trains at platform: {at_platform}/{n_agents}")
        agent_descriptors = self._extract_agent_status_descriptors(state_dict)
        if agent_descriptors:
            text_parts.append("critical_agents: " + "; ".join(agent_descriptors[:6]))
        
        text = ". ".join(text_parts)
        
        try:
            embedding = self._encoder.encode([text])[0]
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"嵌入编码失败: {e}，使用启发式编码")
            return self._encode_heuristic(state_dict)
    
    def _encode_heuristic(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """启发式状态编码（基于规则）
        
        生成固定维度的特征向量：
        - 规模特征 (4维): n_agents, n_nodes, n_platforms, utilization
        - 中断特征 (8维): 各类型中断数量、位置分布
        - 乘客特征 (8维): 等待人数统计
        - 列车特征 (12维): 位置、状态分布
        """
        features = []
        
        # === 规模特征 (4维) ===
        n_agents = state_dict.get("n_agents", 0)
        n_nodes = state_dict.get("n_nodes", 0)
        n_platforms = state_dict.get("n_platforms", 0)
        utilization = n_agents / max(n_platforms, 1) if n_platforms > 0 else 0
        
        features.extend([
            min(n_agents / 20, 1.0),  # 归一化
            min(n_nodes / 100, 1.0),
            min(n_platforms / 50, 1.0),
            min(utilization, 1.0),
        ])
        
        # === 中断特征 (8维) ===
        disruptions = state_dict.get("disruptions", [])
        n_blocks = sum(1 for d in disruptions if d.get("type") in ("BLOCK", "BLOCKAGE"))
        n_speed_limits = sum(1 for d in disruptions if d.get("type") == "SPEED_LIMIT")
        n_total_disruptions = len(disruptions)
        
        # 中断影响范围（受影响节点数）
        affected_nodes = set()
        for d in disruptions:
            nodes = d.get("affected_nodes", [])
            affected_nodes.update(nodes)
        n_affected = len(affected_nodes)
        
        # 中断位置分布（用于匹配相似场景）
        disruption_positions = [
            d.get("position_ratio", 0.5) for d in disruptions
        ]
        avg_disruption_pos = np.mean(disruption_positions) if disruption_positions else 0.5
        std_disruption_pos = np.std(disruption_positions) if len(disruption_positions) > 1 else 0
        
        features.extend([
            min(n_blocks / 5, 1.0),
            min(n_speed_limits / 5, 1.0),
            min(n_total_disruptions / 10, 1.0),
            min(n_affected / max(n_nodes, 1), 1.0),
            avg_disruption_pos,
            min(std_disruption_pos, 0.5) * 2,
            1.0 if n_blocks > 0 else 0.0,
            1.0 if n_speed_limits > 0 else 0.0,
        ])
        
        # === 乘客特征 (8维) ===
        waiting = state_dict.get("waiting_passengers", {})
        if waiting:
            waiting_values = list(waiting.values())
            total_waiting = sum(waiting_values)
            max_waiting = max(waiting_values)
            mean_waiting = np.mean(waiting_values)
            std_waiting = np.std(waiting_values) if len(waiting_values) > 1 else 0
            high_wait_count = sum(1 for w in waiting_values if w > 500)
        else:
            total_waiting = max_waiting = mean_waiting = std_waiting = high_wait_count = 0
        
        features.extend([
            min(total_waiting / 10000, 1.0),
            min(max_waiting / 2000, 1.0),
            min(mean_waiting / 500, 1.0),
            min(std_waiting / 500, 1.0),
            min(high_wait_count / max(n_platforms, 1), 1.0),
            1.0 if max_waiting > 1000 else 0.0,
            1.0 if high_wait_count > 3 else 0.0,
            min(total_waiting / max(n_agents * 500, 1), 1.0),  # 人均负载
        ])
        
        # === 列车特征 (12维) ===
        positions = state_dict.get("agent_positions", [])
        if positions:
            at_platform = [p for p in positions if p.get("at_platform", False)]
            in_transit = [p for p in positions if not p.get("at_platform", False)]
            
            n_at_platform = len(at_platform)
            n_in_transit = len(in_transit)
            
            # 停站时间分布
            dwell_times = [p.get("dwell_time", 0) for p in at_platform]
            expected_dwells = [p.get("expected_dwell", 1) for p in at_platform]
            
            avg_dwell_ratio = np.mean([
                d / max(e, 1) for d, e in zip(dwell_times, expected_dwells)
            ]) if dwell_times else 0
            
            # 列车拥挤度
            passengers = [p.get("passengers", 0) for p in positions]
            capacities = [p.get("capacity", 2520) for p in positions]
            load_ratios = [p / max(c, 1) for p, c in zip(passengers, capacities)]
            avg_load = np.mean(load_ratios) if load_ratios else 0
            
            # 列车密度分布
            position_nodes = [p.get("node_id", 0) for p in positions]
            unique_positions = len(set(position_nodes))
            density = len(positions) / max(unique_positions, 1)
        else:
            n_at_platform = n_in_transit = 0
            avg_dwell_ratio = avg_load = density = 0
        
        features.extend([
            n_at_platform / max(n_agents, 1),
            n_in_transit / max(n_agents, 1),
            avg_dwell_ratio,
            avg_load,
            min(density / 2, 1.0),
            1.0 if n_at_platform > n_agents * 0.7 else 0.0,  # 大部分在站
            1.0 if avg_load > 0.8 else 0.0,  # 高负载
            min(n_agents / 10, 1.0),
            1.0 if n_in_transit > n_agents * 0.5 else 0.0,  # 大部分在途
            0.0,  # 预留
            0.0,  # 预留
            0.0,  # 预留
        ])

        # === 中断身份特征 (8维) ===
        disruption_buckets = [0.0] * 4
        endpoint_values: List[float] = []
        severity_values: List[float] = []
        for descriptor in self._extract_disruption_descriptors(state_dict):
            bucket = int(hashlib.md5(descriptor.encode("utf-8")).hexdigest(), 16) % len(disruption_buckets)
            disruption_buckets[bucket] += 1.0
        if disruptions:
            disruption_buckets = [value / len(disruptions) for value in disruption_buckets]

        for disruption in disruptions:
            segment_id = self._normalize_segment_id(
                disruption.get("segment_id", disruption.get("segment", ""))
            )
            if "-" in segment_id:
                parts = segment_id.split("-", 1)
                try:
                    endpoint_values.append(int(parts[0]) / max(n_nodes, 1))
                    endpoint_values.append(int(parts[1]) / max(n_nodes, 1))
                except (TypeError, ValueError):
                    pass
            severity_values.append(float(disruption.get("severity", 1.0) or 1.0))

        endpoint_mean = float(np.mean(endpoint_values)) if endpoint_values else 0.0
        endpoint_std = float(np.std(endpoint_values)) if len(endpoint_values) > 1 else 0.0
        severity_mean = float(np.mean(severity_values)) if severity_values else 0.0
        severity_max = float(np.max(severity_values)) if severity_values else 0.0
        features.extend(disruption_buckets)
        features.extend([
            endpoint_mean,
            min(endpoint_std, 1.0),
            min(severity_mean, 2.0) / 2.0,
            min(severity_max, 2.0) / 2.0,
        ])

        # === 列车-中断交互特征 (4维) ===
        over_dwell_ratios: List[float] = []
        disrupted_nodes = set()
        for disruption in disruptions:
            for node_id in disruption.get("affected_nodes", []) or []:
                try:
                    disrupted_nodes.add(int(node_id))
                except (TypeError, ValueError):
                    continue

        disrupted_agent_count = 0
        for info in positions:
            if not isinstance(info, dict):
                continue
            try:
                node_id = int(info.get("node_id", info.get("position", -1)))
            except (TypeError, ValueError):
                node_id = -1
            dwell_time = float(info.get("dwell_time", 0.0) or 0.0)
            expected_dwell = float(info.get("expected_dwell", 0.0) or 0.0)
            if node_id in disrupted_nodes:
                disrupted_agent_count += 1
            if expected_dwell > 0:
                over_dwell_ratios.append(max(dwell_time - expected_dwell, 0.0) / max(expected_dwell, 1.0))

        max_over_dwell = float(np.max(over_dwell_ratios)) if over_dwell_ratios else 0.0
        mean_over_dwell = float(np.mean(over_dwell_ratios)) if over_dwell_ratios else 0.0
        high_wait_nodes = state_dict.get("high_wait_nodes", []) or []
        features.extend([
            disrupted_agent_count / max(n_agents, 1),
            min(max_over_dwell, 3.0) / 3.0,
            min(mean_over_dwell, 2.0) / 2.0,
            min(len(high_wait_nodes) / max(n_platforms, 1), 1.0),
        ])
        
        return np.array(features, dtype=np.float32)
    
    @property
    def feature_dim(self) -> int:
        """特征向量维度"""
        if self.use_embeddings and self._encoder is not None:
            return self.embedding_dim
        else:
            return 44  # 启发式编码的固定维度


class CaseLibrary:
    """策略案例库：维护成功调度案例及其效果"""
    
    def __init__(self, max_cases: int = 10000):
        self.max_cases = max_cases
        self.cases: List[Dict[str, Any]] = []
        self.case_index: Dict[str, int] = {}  # case_id -> index
        self._normalized_matrix: Optional[np.ndarray] = None
        self._matrix_case_indices: List[int] = []
        self._vector_dim: Optional[int] = None
        
        # 性能指标统计
        self.stats = {
            "total_cases": 0,
            "successful_cases": 0,
            "avg_reward": 0.0,
            "best_reward": float("-inf"),
        }
    
    def add_case(
        self,
        case_id: str,
        state_vector: np.ndarray,
        state_description: str,
        action_plan: Dict[str, Any],
        outcome: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """添加新案例
        
        Args:
            case_id: 唯一标识
            state_vector: 状态向量
            state_description: 状态描述文本
            action_plan: 执行的调度方案
            outcome: 执行结果（reward, metrics等）
            metadata: 额外元数据
        
        Returns:
            是否成功添加
        """
        normalized_metadata = self._normalize_metadata(metadata)
        summary = self._build_case_summary(
            state_description=state_description,
            action_plan=action_plan,
            outcome=outcome,
            metadata=normalized_metadata,
        )

        if case_id in self.case_index:
            # 更新已有案例
            idx = self.case_index[case_id]
            self.cases[idx].update({
                "state_vector": state_vector.tolist(),
                "state_description": state_description,
                "action_plan": action_plan,
                "outcome": outcome,
                "metadata": normalized_metadata,
                "summary": summary,
                "updated_at": datetime.now().isoformat(),
            })
            self._rebuild_vector_index()
            return True
        
        # 检查容量
        if len(self.cases) >= self.max_cases:
            self._evict_low_quality_case()
        
        case = {
            "case_id": case_id,
            "state_vector": state_vector.tolist(),
            "state_description": state_description,
            "action_plan": action_plan,
            "outcome": outcome,
            "metadata": normalized_metadata,
            "summary": summary,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "success_rate": 0.0,
        }
        
        self.cases.append(case)
        self.case_index[case_id] = len(self.cases) - 1
        self._rebuild_vector_index()
        
        # 更新统计
        self._update_stats(outcome)
        
        return True

    def _normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """统一 metadata 结构，避免旧案例缺少常用字段。"""
        return dict(metadata or {})

    def _build_case_summary(
        self,
        state_description: str,
        action_plan: Dict[str, Any],
        outcome: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        """为提示词构造稳定、可读的案例摘要。"""
        state_dict = metadata.get("state_dict") if isinstance(metadata.get("state_dict"), dict) else {}
        disruptions = state_dict.get("disruptions", []) if isinstance(state_dict, dict) else []
        high_wait_nodes = state_dict.get("high_wait_nodes", []) if isinstance(state_dict, dict) else []

        selected_plan = action_plan.get("selected_plan") if isinstance(action_plan, dict) else {}
        if not isinstance(selected_plan, dict):
            selected_plan = {}

        method = (
            selected_plan.get("method")
            or metadata.get("selected_method")
            or "unknown"
        )
        focus_agents = selected_plan.get("focus_agents") or []
        focus_nodes = selected_plan.get("focus_nodes") or []
        reward = float(outcome.get("total_reward", 0.0) or 0.0)
        success = bool(outcome.get("success", False))

        disruption_desc = "无中断"
        if disruptions:
            formatted = []
            for disruption in disruptions[:2]:
                segment = disruption.get("segment_id") or disruption.get("segment") or "unknown"
                dtype = disruption.get("type", "UNKNOWN")
                formatted.append(f"{segment}:{dtype}")
            disruption_desc = ", ".join(formatted)

        parts = [
            f"方法={method}",
            f"中断={disruption_desc}",
            f"高等待站点={len(high_wait_nodes)}",
            f"focus_agents={list(focus_agents)[:4]}",
            f"focus_nodes={list(focus_nodes)[:4]}",
            f"success={success}",
            f"reward={reward:.3f}",
        ]

        summary = " | ".join(parts)
        if not summary.strip() and state_description:
            summary = str(state_description).strip()
        return summary[:200]

    def _normalize_loaded_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """兼容旧案例格式，补齐摘要与 metadata。"""
        normalized = dict(case)
        metadata = self._normalize_metadata(normalized.get("metadata"))
        normalized["metadata"] = metadata
        normalized["summary"] = str(normalized.get("summary") or self._build_case_summary(
            state_description=str(normalized.get("state_description", "")),
            action_plan=normalized.get("action_plan") if isinstance(normalized.get("action_plan"), dict) else {},
            outcome=normalized.get("outcome") if isinstance(normalized.get("outcome"), dict) else {},
            metadata=metadata,
        )).strip()
        return normalized

    def _rebuild_vector_index(self):
        """预构建归一化案例矩阵，避免查询时逐条做余弦扫描。"""
        self._normalized_matrix = None
        self._matrix_case_indices = []
        self._vector_dim = None

        normalized_rows = []
        valid_case_indices = []
        expected_dim = None

        for idx, case in enumerate(self.cases):
            raw_vector = case.get("state_vector")
            if raw_vector is None:
                continue

            vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
            if vector.size == 0:
                continue

            if expected_dim is None:
                expected_dim = int(vector.shape[0])
            if int(vector.shape[0]) != expected_dim:
                continue

            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                continue

            normalized_rows.append(vector / norm)
            valid_case_indices.append(idx)

        if normalized_rows:
            self._normalized_matrix = np.vstack(normalized_rows).astype(np.float32, copy=False)
            self._matrix_case_indices = valid_case_indices
            self._vector_dim = expected_dim
    
    def _update_stats(self, outcome: Dict[str, Any]):
        """更新统计信息"""
        self.stats["total_cases"] = len(self.cases)
        
        reward = outcome.get("total_reward", 0)
        if reward > self.stats["best_reward"]:
            self.stats["best_reward"] = reward
        
        if outcome.get("success", False):
            self.stats["successful_cases"] += 1
    
    def _evict_low_quality_case(self):
        """移除低质量案例（基于使用频率和效果）"""
        if not self.cases:
            return
        
        # 计算案例质量分数
        scores = []
        for case in self.cases:
            outcome = case.get("outcome", {})
            reward = outcome.get("total_reward", 0)
            usage = case.get("usage_count", 0)
            success_rate = case.get("success_rate", 0)
            
            # 质量分数 = 奖励 * 使用频率权重 * 成功率
            quality = reward * (1 + np.log1p(usage)) * (0.5 + success_rate)
            scores.append(quality)
        
        # 移除最低质量的案例
        min_idx = np.argmin(scores)
        removed_case = self.cases.pop(min_idx)
        del self.case_index[removed_case["case_id"]]
        
        # 更新索引
        for i, case in enumerate(self.cases):
            self.case_index[case["case_id"]] = i
        self._rebuild_vector_index()

    def _search_normalized_matrix(
        self,
        query_vector: np.ndarray,
        k: int,
        min_similarity: float,
    ) -> List[Dict[str, Any]]:
        """使用预归一化矩阵做单次矩阵乘法检索。"""
        if self._normalized_matrix is None or self._vector_dim is None:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if int(query.shape[0]) != int(self._vector_dim):
            logger.warning(
                "RAG向量维度不匹配，跳过矩阵检索: query=%s, matrix_dim=%s",
                query.shape,
                (self._vector_dim,),
            )
            return []

        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []

        normalized_query = query / query_norm
        similarities = self._normalized_matrix @ normalized_query
        candidate_positions = np.flatnonzero(similarities >= float(min_similarity))
        if candidate_positions.size == 0:
            return []

        candidate_positions = candidate_positions[
            np.argsort(similarities[candidate_positions])[::-1]
        ]

        results = []
        for matrix_pos in candidate_positions[:k]:
            sim = float(similarities[matrix_pos])
            case_idx = self._matrix_case_indices[int(matrix_pos)]
            case = self.cases[case_idx]
            case["usage_count"] = case.get("usage_count", 0) + 1
            results.append({
                "similarity": sim,
                **case,
            })

        return results
    
    def get_similar_cases(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """检索相似案例
        
        Args:
            query_vector: 查询状态向量
            k: 返回数量
            min_similarity: 最小相似度阈值
        
        Returns:
            相似案例列表（按相似度降序）
        """
        if not self.cases:
            return []

        if self._normalized_matrix is not None:
            return self._search_normalized_matrix(
                query_vector=query_vector,
                k=k,
                min_similarity=min_similarity,
            )
        
        # 计算余弦相似度
        similarities = []
        for case in self.cases:
            case_vector = np.array(case["state_vector"], dtype=np.float32)
            sim = self._cosine_similarity(query_vector, case_vector)
            if sim >= min_similarity:
                similarities.append((sim, case))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 更新使用计数
        results = []
        for sim, case in similarities[:k]:
            case["usage_count"] = case.get("usage_count", 0) + 1
            results.append({
                "similarity": sim,
                **case,
            })
        
        return results
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        if a.shape != b.shape:
            logger.warning("RAG向量维度不匹配，跳过该案例: query=%s, case=%s", a.shape, b.shape)
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def update_case_feedback(
        self,
        case_id: str,
        feedback: Dict[str, Any],
    ) -> bool:
        """更新案例反馈（用于在线学习）"""
        if case_id not in self.case_index:
            return False
        
        idx = self.case_index[case_id]
        case = self.cases[idx]
        
        # 更新成功率
        was_successful = feedback.get("success", False)
        current_rate = case.get("success_rate", 0)
        usage = case.get("usage_count", 1)
        
        # 指数移动平均更新成功率
        alpha = 0.1
        new_rate = current_rate * (1 - alpha) + (1.0 if was_successful else 0.0) * alpha
        case["success_rate"] = new_rate
        
        return True
    
    def save(self, path: str):
        """保存案例库"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "cases": self.cases,
                "stats": self.stats,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"案例库已保存: {path}, 共 {len(self.cases)} 条案例")
    
    def load(self, path: str):
        """加载案例库"""
        if not os.path.exists(path):
            logger.warning(f"案例库文件不存在: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        loaded_cases = data.get("cases", [])
        self.cases = [self._normalize_loaded_case(case) for case in loaded_cases]

        self.stats = data.get("stats", self.stats)
        
        # 重建索引
        self.case_index = {
            case["case_id"]: i for i, case in enumerate(self.cases)
        }
        self._rebuild_vector_index()
        
        logger.info(f"案例库已加载: {path}, 共 {len(self.cases)} 条案例")


class ConstraintExtractor:
    """经验约束提取器：从历史案例提取可复用的约束"""
    
    def __init__(self):
        # 约束模板库
        self.constraint_templates = {
            "dwell_time": "agent_{agent_idx} 在节点 {node_idx} 必须停留至少 {min_dwell} 步",
            "avoid_segment": "agent_{agent_idx} 应避免进入区段 {segment}",
            "priority_serve": "优先服务等待人数超过 {threshold} 的站点",
            "maintain_headway": "相邻列车间隔应保持在 {min_headway} 到 {max_headway} 之间",
            "capacity_limit": "列车 {agent_idx} 载客率超过 {threshold} 时应优先卸客",
        }
    
    def extract_constraints(
        self,
        similar_cases: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """从相似案例中提取约束
        
        Args:
            similar_cases: 相似案例列表
            current_state: 当前状态
        
        Returns:
            约束列表，每个约束包含:
            - type: 约束类型
            - description: 约束描述
            - variables: 涉及的变量
            - confidence: 置信度
        """
        constraints = []
        
        if not similar_cases:
            return constraints
        
        # 1. 提取停站时间约束
        dwell_constraints = self._extract_dwell_constraints(similar_cases)
        constraints.extend(dwell_constraints)
        
        # 2. 提取避让约束（基于中断信息）
        avoidance_constraints = self._extract_avoidance_constraints(
            similar_cases, current_state
        )
        constraints.extend(avoidance_constraints)
        
        # 3. 提取服务优先级约束
        priority_constraints = self._extract_priority_constraints(
            similar_cases, current_state
        )
        constraints.extend(priority_constraints)
        
        # 4. 提取间隔约束
        headway_constraints = self._extract_headway_constraints(similar_cases)
        constraints.extend(headway_constraints)
        
        return constraints
    
    def _extract_dwell_constraints(
        self,
        cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """提取停站时间约束"""
        constraints = []
        
        # 统计各站点的推荐停站时间
        dwell_stats = defaultdict(list)
        
        for case in cases:
            action_plan = case.get("action_plan", {})
            steps = action_plan.get("steps", [])
            
            for step in steps:
                for agent_key, decision in step.items():
                    if not str(agent_key).startswith("agent_"):
                        continue
                    if decision.get("action", 0) == 0:  # 停留
                        node = decision.get("next_node", -1)
                        if node >= 0:
                            dwell_stats[node].append(1)
        
        # 生成约束
        for node, dwells in dwell_stats.items():
            if len(dwells) >= 2:  # 至少有2个样本
                avg_dwell = np.mean(dwells)
                if avg_dwell > 1:
                    constraints.append({
                        "type": "dwell_time",
                        "description": f"基于历史经验，节点 {node} 推荐停留时间",
                        "variables": {"node_idx": node, "min_dwell": int(avg_dwell)},
                        "confidence": min(len(dwells) / 10, 1.0),
                    })
        
        return constraints
    
    def _extract_avoidance_constraints(
        self,
        cases: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """提取避让约束"""
        constraints = []
        
        # 获取当前中断信息
        disruptions = current_state.get("disruptions", [])
        disrupted_nodes = set()
        
        for d in disruptions:
            if d.get("type") in ("BLOCK", "BLOCKAGE"):
                nodes = d.get("affected_nodes", [])
                disrupted_nodes.update(nodes)
        
        # 从案例中学习避让策略
        avoidance_patterns = defaultdict(int)
        
        for case in cases:
            case_state = case.get("metadata", {}).get("state_dict", {})
            case_disruptions = case_state.get("disruptions", [])
            
            for d in case_disruptions:
                if d.get("type") in ("BLOCK", "BLOCKAGE"):
                    segment = d.get("segment_id", "")
                    if segment:
                        avoidance_patterns[segment] += 1
        
        # 生成约束
        for segment, count in avoidance_patterns.items():
            if count >= 2:  # 至少出现2次
                constraints.append({
                    "type": "avoid_segment",
                    "description": f"历史案例中频繁避让区段 {segment}",
                    "variables": {"segment": segment},
                    "confidence": min(count / 5, 1.0),
                })
        
        return constraints
    
    def _extract_priority_constraints(
        self,
        cases: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """提取服务优先级约束"""
        constraints = []
        
        # 分析历史案例中的服务模式
        threshold_candidates = [500, 800, 1000, 1500]
        
        for threshold in threshold_candidates:
            priority_cases = 0
            for case in cases:
                outcome = case.get("outcome", {})
                if outcome.get("success", False):
                    high_wait_served = outcome.get("high_wait_nodes_served", 0)
                    total_high_wait = outcome.get("total_high_wait_nodes", 0)
                    if total_high_wait > 0 and high_wait_served / total_high_wait > 0.7:
                        priority_cases += 1
            
            if priority_cases >= len(cases) * 0.3:
                constraints.append({
                    "type": "priority_serve",
                    "description": f"优先服务等待人数超过 {threshold} 的站点",
                    "variables": {"threshold": threshold},
                    "confidence": priority_cases / len(cases),
                })
                break
        
        return constraints
    
    def _extract_headway_constraints(
        self,
        cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """提取间隔约束"""
        constraints = []
        
        # 统计成功案例的间隔分布
        headways = []
        
        for case in cases:
            outcome = case.get("outcome", {})
            if outcome.get("success", False):
                avg_headway = outcome.get("avg_headway", 0)
                if avg_headway > 0:
                    headways.append(avg_headway)
        
        if len(headways) >= 3:
            min_headway = np.percentile(headways, 25)
            max_headway = np.percentile(headways, 75)
            
            constraints.append({
                "type": "maintain_headway",
                "description": f"列车间隔应保持在 {min_headway:.0f} 到 {max_headway:.0f} 秒之间",
                "variables": {
                    "min_headway": int(min_headway),
                    "max_headway": int(max_headway),
                },
                "confidence": 0.8,
            })
        
        return constraints


class EnhancedRAGManager:
    """增强的RAG管理器
    
    集成向量检索、案例库和约束提取，为LLM Agent提供经验支持。
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_cases: int = 10000,
    ):
        self.db_path = db_path
        self._native_profiles = {}
        
        # 状态编码器
        self.encoder = StateEncoder(
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
        )
        
        # 案例库
        self.case_library = CaseLibrary(max_cases=max_cases)
        
        # 约束提取器
        self.constraint_extractor = ConstraintExtractor()
        
        # 加载已有数据
        if db_path and os.path.exists(db_path):
            self.load(db_path)
        
        # 运行统计
        self.stats = {
            "queries": 0,
            "cache_hits": 0,
            "cases_added": 0,
            "constraints_generated": 0,
        }
    
    def add_experience(
        self,
        state_dict: Dict[str, Any],
        state_description: str,
        action_plan: Dict[str, Any],
        outcome: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """添加调度经验
        
        Args:
            state_dict: 状态字典
            state_description: 状态描述
            action_plan: 执行的调度方案
            outcome: 执行结果
            metadata: 额外元数据
        
        Returns:
            案例ID
        """
        # 生成案例ID
        case_id = self._generate_case_id(state_dict)
        
        # 编码状态
        state_vector = self.encoder.encode_state(state_dict)
        
        # 添加到案例库
        full_metadata = metadata or {}
        full_metadata["state_dict"] = state_dict
        
        success = self.case_library.add_case(
            case_id=case_id,
            state_vector=state_vector,
            state_description=state_description,
            action_plan=action_plan,
            outcome=outcome,
            metadata=full_metadata,
        )
        
        if success:
            self._native_profiles.pop(case_id, None)
            self.stats["cases_added"] += 1
        
        return case_id
    
    def retrieve(
        self,
        current_state: Dict[str, Any],
        k: int = 5,
        min_similarity: float = 0.5,
    ) -> Dict[str, Any]:
        """检索相关经验和约束
        
        Args:
            current_state: 当前状态字典
            k: 检索数量
            min_similarity: 最小相似度
        
        Returns:
            包含以下字段:
            - similar_cases: 相似案例列表
            - constraints: 提取的约束
            - warm_start_hints: 初始解建议
            - strategy_recommendations: 策略建议
        """
        self.stats["queries"] += 1
        if current_state.get("dispatch_state"):
            return self._retrieve_native(current_state, k, min_similarity)
        
        # 编码当前状态
        query_vector = self.encoder.encode_state(current_state)

        first_case_vector = None
        if self.case_library.cases:
            first_case_vector = self.case_library.cases[0].get("state_vector")

        if first_case_vector is not None and len(first_case_vector) != int(query_vector.shape[0]):
            logger.warning(
                "RAG query/case 维度不一致，将回退到基于 state_dict 的同构启发式检索: query=%s, case=%s",
                query_vector.shape,
                (len(first_case_vector),),
            )
            similar_cases = self._retrieve_with_state_dict_fallback(
                current_state=current_state,
                k=k,
                min_similarity=min_similarity,
            )
        else:
            # 检索相似案例
            similar_cases = self.case_library.get_similar_cases(
                query_vector=query_vector,
                k=k,
                min_similarity=min_similarity,
            )

        similar_cases = self._rerank_similar_cases(
            similar_cases=similar_cases,
            current_state=current_state,
            query_vector=query_vector,
            k=k,
        )

        # 提取约束
        constraints = self.constraint_extractor.extract_constraints(
            similar_cases, current_state
        )
        self.stats["constraints_generated"] += len(constraints)
        
        # 生成初始解建议
        warm_start_hints = self._generate_warm_start_hints(similar_cases, current_state)
        
        # 生成策略建议
        strategy_recommendations = self._generate_strategy_recommendations(
            similar_cases, current_state
        )

        # 生成工具调用建议
        tool_call_recommendations = self._generate_tool_call_recommendations(
            similar_cases, current_state
        )
        
        return {
            "similar_cases": similar_cases,
            "constraints": constraints,
            "warm_start_hints": warm_start_hints,
            "strategy_recommendations": strategy_recommendations,
            "tool_call_recommendations": tool_call_recommendations,
        }

    def _retrieve_native(self, current_state, k, min_similarity):
        from LLM.rag.native_case import case_profile, state_profile, transfer_match
        query = self.encoder._encode_native(current_state)
        current = state_profile(current_state)
        ranked, invalid_vectors, unreadable = [], 0, 0
        for case in self.case_library.cases:
            if not case.get("metadata", {}).get("native_dispatch"):
                continue
            vector = np.asarray(case.get("state_vector", []), dtype=np.float32)
            if vector.shape != query.shape or not np.isfinite(vector).all():
                invalid_vectors += 1
                continue
            similarity = self.case_library._cosine_similarity(query, vector)
            if similarity < min_similarity:
                continue
            key = (id(case), id(case.get("state_description")), id(case.get("metadata", {}).get("state_dict")))
            cached = self._native_profiles.get(case["case_id"])
            if cached is None or cached[0] != key:
                cached = self._native_profiles[case["case_id"]] = (key, case_profile(case))
            profile = cached[1]
            if not profile["state_available"]:
                unreadable += 1
                continue
            match = transfer_match(current, profile, similarity)
            ranked.append(dict(case, similarity=float(similarity), transfer_profile=profile, transfer_match=match))
        ranked.sort(key=lambda c: (-c["transfer_match"]["score"], -c["similarity"], str(c["case_id"])))
        selected = ranked[:max(0, k)]
        for case in selected:
            original = self.case_library.cases[self.case_library.case_index[case["case_id"]]]
            original["usage_count"] = original.get("usage_count", 0)+1
        audit = dict(method="native_coupling_rerank_v1", eligible_cases=len(ranked), selected_cases=len(selected),
            invalid_vectors=invalid_vectors, unreadable_profiles=unreadable,
            case_ids=[c["case_id"] for c in selected],
            matches=[dict(case_id=c["case_id"], **c["transfer_match"]) for c in selected],
            exact_topology_required=False, exact_event_set_required=False, historical_budget_required=False)
        logger.info("native_rag_retrieval %s", json.dumps(audit, ensure_ascii=False))
        return dict(similar_cases=selected, constraints=[], warm_start_hints=[],
            strategy_recommendations=[c.get("action_plan", {}).get("specification", {}) for c in selected],
            tool_call_recommendations=[], retrieval_diagnostics=audit)

    def _extract_disruption_signature(
        self,
        state_dict: Dict[str, Any],
    ) -> Tuple[Set[str], Set[str], Set[int]]:
        segments: Set[str] = set()
        types: Set[str] = set()
        nodes: Set[int] = set()

        for disruption in state_dict.get("disruptions", []) or []:
            segment_id = self.encoder._normalize_segment_id(
                disruption.get("segment_id", disruption.get("segment", ""))
            )
            if segment_id:
                segments.add(segment_id)
            dtype = str(disruption.get("type", "UNKNOWN")).upper()
            if dtype:
                types.add(dtype)
            for node_id in disruption.get("affected_nodes", []) or []:
                try:
                    nodes.add(int(node_id))
                except (TypeError, ValueError):
                    continue

            if not disruption.get("affected_nodes") and "-" in segment_id:
                left, right = segment_id.split("-", 1)
                try:
                    nodes.add(int(left))
                    nodes.add(int(right))
                except (TypeError, ValueError):
                    pass

        return segments, types, nodes

    def _compute_case_match_score(
        self,
        current_state: Dict[str, Any],
        case_state: Dict[str, Any],
    ) -> Tuple[float, bool]:
        current_segments, current_types, current_nodes = self._extract_disruption_signature(current_state)
        case_segments, case_types, case_nodes = self._extract_disruption_signature(case_state)

        if not current_segments and not current_types:
            return 0.0, False

        exact_segment_overlap = len(current_segments & case_segments)
        type_overlap = len(current_types & case_types)
        node_overlap = len(current_nodes & case_nodes)

        score = 0.0
        if current_segments:
            score += 3.0 * (exact_segment_overlap / len(current_segments))
        if current_types:
            score += 1.5 * (type_overlap / len(current_types))
        if current_nodes:
            score += 1.0 * (node_overlap / len(current_nodes))

        score -= 0.15 * abs(len(current_segments) - len(case_segments))
        score -= 0.10 * abs(len(current_types) - len(case_types))
        return score, exact_segment_overlap > 0

    def _rerank_similar_cases(
        self,
        similar_cases: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        query_vector: np.ndarray,
        k: int,
    ) -> List[Dict[str, Any]]:
        """按中断签名重排案例，并在必要时补充全库扫描得到的强匹配候选。"""
        current_has_disruption = bool(current_state.get("disruptions"))
        if not similar_cases and not current_has_disruption:
            return similar_cases

        candidates: Dict[str, Dict[str, Any]] = {}
        for case in similar_cases:
            case_copy = dict(case)
            case_id = str(case_copy.get("case_id", ""))
            case_state = case_copy.get("metadata", {}).get("state_dict", {})
            match_score, exact_match = self._compute_case_match_score(current_state, case_state)
            case_copy["scenario_match_score"] = match_score
            case_copy["exact_disruption_match"] = exact_match
            candidates[case_id] = case_copy

        if current_has_disruption:
            for case in self.case_library.cases:
                case_id = str(case.get("case_id", ""))
                case_state = case.get("metadata", {}).get("state_dict", {})
                if not isinstance(case_state, dict):
                    continue

                match_score, exact_match = self._compute_case_match_score(current_state, case_state)
                if match_score <= 0.0:
                    continue

                case_copy = dict(case)
                case_copy.setdefault("similarity", 0.0)
                case_copy["scenario_match_score"] = match_score
                case_copy["exact_disruption_match"] = exact_match

                raw_vector = case.get("state_vector")
                if isinstance(raw_vector, list) and len(raw_vector) == int(query_vector.shape[0]):
                    case_copy["similarity"] = self.case_library._cosine_similarity(
                        query_vector,
                        np.asarray(raw_vector, dtype=np.float32),
                    )

                existing = candidates.get(case_id)
                if existing is None:
                    candidates[case_id] = case_copy
                else:
                    if (
                        float(case_copy.get("scenario_match_score", 0.0)) > float(existing.get("scenario_match_score", 0.0))
                        or bool(case_copy.get("exact_disruption_match")) and not bool(existing.get("exact_disruption_match"))
                    ):
                        merged = dict(existing)
                        merged.update(case_copy)
                        candidates[case_id] = merged

        ranked = sorted(
            candidates.values(),
            key=lambda case: (
                bool(case.get("exact_disruption_match", False)),
                float(case.get("scenario_match_score", 0.0)),
                float(case.get("similarity", 0.0)),
                float(case.get("outcome", {}).get("total_reward", float("-inf"))),
            ),
            reverse=True,
        )
        return ranked[:k]

    def _retrieve_with_state_dict_fallback(
        self,
        current_state: Dict[str, Any],
        k: int = 5,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """当案例向量与查询向量维度不一致时，使用双方 state_dict 的同构启发式编码做回退检索。"""
        query_vector = self.encoder._encode_heuristic(current_state)

        similarities: List[Tuple[float, Dict[str, Any]]] = []
        for case in self.case_library.cases:
            case_state = case.get("metadata", {}).get("state_dict")
            if not case_state:
                continue

            case_vector = case.get("_heuristic_state_vector")
            if case_vector is None:
                case_vector = self.encoder._encode_heuristic(case_state)
                case["_heuristic_state_vector"] = case_vector

            sim = self.case_library._cosine_similarity(query_vector, case_vector)
            if sim >= min_similarity:
                similarities.append((sim, case))

        similarities.sort(key=lambda item: item[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for sim, case in similarities[:k]:
            case["usage_count"] = case.get("usage_count", 0) + 1
            results.append({
                "similarity": sim,
                **case,
            })
        return results
    
    def _generate_case_id(self, state_dict: Dict[str, Any]) -> str:
        """生成案例ID"""
        # 基于状态特征的哈希
        key_features = json.dumps({
            "n_agents": state_dict.get("n_agents", 0),
            "disruptions": [
                d.get("segment_id", "") for d in state_dict.get("disruptions", [])
            ],
            "high_wait_count": len(state_dict.get("high_wait_nodes", [])),
        }, sort_keys=True)
        
        hash_val = hashlib.md5(key_features.encode()).hexdigest()[:8]
        timestamp = int(time.time() * 1000) % 100000
        
        return f"case_{hash_val}_{timestamp}"
    
    def _generate_warm_start_hints(
        self,
        similar_cases: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """从相似案例生成初始解建议"""
        hints = {
            "stay": {},  # stay[agent,node,t] -> value
            "move": {},  # move[agent,origin,dest,t] -> value
            "focus_agents": [],
            "focus_nodes": [],
        }
        
        if not similar_cases:
            return hints
        
        # 选择最成功的案例作为参考
        best_case = max(
            similar_cases,
            key=lambda c: c.get("outcome", {}).get("total_reward", float("-inf"))
        )
        
        action_plan = best_case.get("action_plan", {})
        steps = action_plan.get("steps", []) if isinstance(action_plan, dict) else []

        # 旧 schema：直接从多步动作中取第一步。
        if steps:
            first_step = steps[0]
            for agent_key, decision in first_step.items():
                try:
                    agent_idx = int(agent_key.replace("agent_", ""))
                    action = decision.get("action", 0)
                    node = decision.get("next_node", -1)

                    if action == 0 and node >= 0:
                        hints["stay"][f"stay[{agent_idx},{node},0]"] = 1

                    reasoning = decision.get("reasoning", "")
                    if "中断" in reasoning or "关键" in reasoning:
                        hints["focus_agents"].append(agent_idx)
                except (ValueError, TypeError):
                    continue

        # 当前 gold case schema：从 selected_plan/tool_calls 中恢复 focus 信息。
        selected_plan = action_plan.get("selected_plan") if isinstance(action_plan, dict) else {}
        if isinstance(selected_plan, dict):
            for agent_idx in selected_plan.get("focus_agents", []) or []:
                try:
                    parsed = int(agent_idx)
                except (TypeError, ValueError):
                    continue
                if parsed not in hints["focus_agents"]:
                    hints["focus_agents"].append(parsed)

            for node_idx in selected_plan.get("focus_nodes", []) or []:
                try:
                    parsed = int(node_idx)
                except (TypeError, ValueError):
                    continue
                if parsed not in hints["focus_nodes"]:
                    hints["focus_nodes"].append(parsed)

        for tool_call in action_plan.get("tool_calls", []) if isinstance(action_plan, dict) else []:
            inputs = tool_call.get("inputs") if isinstance(tool_call, dict) else {}
            if not isinstance(inputs, dict):
                continue
            config = inputs.get("config") if isinstance(inputs.get("config"), dict) else {}

            for agent_idx in config.get("focus_agents", []) or []:
                try:
                    parsed = int(agent_idx)
                except (TypeError, ValueError):
                    continue
                if parsed not in hints["focus_agents"]:
                    hints["focus_agents"].append(parsed)

            for node_idx in config.get("focus_nodes", []) or []:
                try:
                    parsed = int(node_idx)
                except (TypeError, ValueError):
                    continue
                if parsed not in hints["focus_nodes"]:
                    hints["focus_nodes"].append(parsed)
        
        # 从当前状态提取focus_nodes
        high_wait_nodes = current_state.get("high_wait_nodes", [])
        for node_info in high_wait_nodes[:10]:
            node_idx = node_info.get("node_idx", node_info)
            try:
                parsed = int(node_idx)
            except (TypeError, ValueError):
                continue
            if parsed not in hints["focus_nodes"]:
                hints["focus_nodes"].append(parsed)
        
        return hints
    
    def _generate_strategy_recommendations(
        self,
        similar_cases: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> List[str]:
        """生成策略建议"""
        recommendations = []
        
        if not similar_cases:
            return ["无历史经验可参考，建议使用默认策略"]
        
        # 分析成功案例的共同特征
        successful_cases = [
            c for c in similar_cases
            if c.get("outcome", {}).get("success", False)
        ]
        
        if successful_cases:
            # 提取共同策略
            common_strategies = defaultdict(int)
            for case in successful_cases:
                metadata = case.get("metadata", {})
                strategies = metadata.get("applied_strategies", [])
                for s in strategies:
                    common_strategies[s] += 1
            
            for strategy, count in sorted(
                common_strategies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]:
                recommendations.append(
                    f"推荐策略: {strategy} (历史成功率: {count}/{len(successful_cases)})"
                )

            # 工具链偏好（从历史成功案例提取）
            tool_chain_stats = defaultdict(int)
            for case in successful_cases:
                metadata = case.get("metadata", {})
                tool_chain = metadata.get("tool_chain", [])
                for tool_name in tool_chain:
                    tool_chain_stats[str(tool_name)] += 1

            for tool_name, count in sorted(
                tool_chain_stats.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:2]:
                recommendations.append(
                    f"推荐工具: {tool_name} (成功案例支持: {count}/{len(successful_cases)})"
                )
        
        # 基于当前状态的建议
        disruptions = current_state.get("disruptions", [])
        if disruptions:
            block_count = sum(1 for d in disruptions if d.get("type") in ("BLOCK", "BLOCKAGE"))
            if block_count > 0:
                recommendations.append(
                    f"检测到 {block_count} 处区间中断，建议启用避让策略"
                )
        
        high_wait = current_state.get("high_wait_nodes", [])
        if len(high_wait) > 3:
            recommendations.append(
                f"检测到 {len(high_wait)} 个高等待站点，建议优先服务"
            )
        
        return recommendations if recommendations else ["状态正常，维持当前调度策略"]

    def _generate_tool_call_recommendations(
        self,
        similar_cases: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """生成面向LLM Agent的工具调用建议。"""
        if not similar_cases:
            return [
                {
                    "order": 1,
                    "tool": "small_model_guidance.generate_guidance",
                    "reason": "无相似案例时先生成focus/warm_start",
                },
                {
                    "order": 2,
                    "tool": "optimization_toolkit.solve",
                    "reason": "使用hybrid或heuristic进行保守求解",
                    "method": "hybrid",
                },
            ]

        successful_cases = [
            c for c in similar_cases
            if c.get("outcome", {}).get("success", False)
        ]
        reference_cases = successful_cases if successful_cases else similar_cases

        tool_counter = defaultdict(int)
        method_counter = defaultdict(int)

        for case in reference_cases:
            metadata = case.get("metadata", {})
            for tool_name in metadata.get("tool_chain", []):
                tool_counter[str(tool_name)] += 1

            action_plan = case.get("action_plan", {})
            for item in action_plan.get("tool_calls", []):
                tool_name = str(item.get("tool", ""))
                if tool_name:
                    tool_counter[tool_name] += 1
                method = item.get("inputs", {}).get("method")
                if method:
                    method_counter[str(method)] += 1

        recommended_tools = [
            name for name, _ in sorted(tool_counter.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
        recommended_method = None
        if method_counter:
            recommended_method = sorted(method_counter.items(), key=lambda x: x[1], reverse=True)[0][0]

        # 当前状态修正（无历史method时）
        if not recommended_method:
            disruptions = current_state.get("disruptions", [])
            waiting = current_state.get("waiting_passengers", {})
            max_wait = max(waiting.values()) if waiting else 0
            if disruptions:
                recommended_method = "benders"
            elif max_wait > 1000:
                recommended_method = "queueing"
            else:
                recommended_method = "hybrid"

        calls: List[Dict[str, Any]] = []
        order = 1
        if "small_model_guidance.generate_guidance" in recommended_tools or not recommended_tools:
            calls.append({
                "order": order,
                "tool": "small_model_guidance.generate_guidance",
                "reason": "先缩小关键agent/node搜索空间",
            })
            order += 1

        calls.append({
            "order": order,
            "tool": "optimization_toolkit.solve",
            "reason": "基于历史成功案例的主求解器",
            "method": recommended_method,
        })

        return calls
    
    def update_feedback(
        self,
        case_id: str,
        feedback: Dict[str, Any],
    ) -> bool:
        """更新案例反馈"""
        return self.case_library.update_case_feedback(case_id, feedback)
    
    def save(self, path: Optional[str] = None):
        """保存数据"""
        save_path = path or self.db_path
        if save_path:
            self.case_library.save(save_path)
    
    def load(self, path: Optional[str] = None):
        """加载数据"""
        load_path = path or self.db_path
        if load_path:
            self._native_profiles.clear()
            self.case_library.load(load_path)

            target_dim = int(self.encoder.feature_dim)
            for case in self.case_library.cases:
                raw_vector = case.get("state_vector")
                vector_dim = len(raw_vector) if isinstance(raw_vector, list) else None
                if vector_dim == target_dim:
                    continue

                case_state = case.get("metadata", {}).get("state_dict")
                if not isinstance(case_state, dict):
                    continue

                try:
                    aligned_vector = self.encoder.encode_state(case_state)
                except Exception as exc:
                    logger.warning(
                        "重建案例向量失败，保留原向量: case_id=%s, error=%s",
                        case.get("case_id"),
                        exc,
                    )
                    continue

                case["state_vector"] = aligned_vector.astype(np.float32).reshape(-1).tolist()

            self.case_library._rebuild_vector_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "case_library_stats": self.case_library.stats,
            "encoder_type": "embedding" if self.encoder.use_embeddings else "heuristic",
            "feature_dim": self.encoder.feature_dim,
        }


# 向后兼容：保留原始RAGManager接口
class RAGManager(EnhancedRAGManager):
    """向后兼容的RAG管理器别名"""
    pass
