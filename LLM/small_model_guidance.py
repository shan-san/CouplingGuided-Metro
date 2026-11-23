"""
小规模语言模型求解引导模块

使用经SFT微调的小规模语言模型提供求解指导；主控制流程不进行后续RL或在线更新。

核心功能：
1. 小模型推理 - 快速生成初始解和约束建议
2. 知识蒸馏 - 从大模型学习求解策略
3. 在线微调 - 基于求解反馈持续优化
4. 多模型融合 - 小模型快速响应 + 大模型复杂推理
"""

import json
import logging
import time
import os
import re
import subprocess
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

NATIVE_GUIDANCE_SYSTEM = ("Generate JSON solver guidance for the supplied current native-command variable catalog. "
    "Return warm_start, variable_fixing and branching_priority keyed only by supplied command_g_i names. "
    "Do not generate dispatching actions or refer to variable names from another state.")


def native_guidance_messages(state_description):
    system = NATIVE_GUIDANCE_SYSTEM
    try:
        payload = json.loads(state_description)
        if payload.get("encoding") == "metro_guidance_group_v2":
            from LLM.finetune.guidance_view import SYSTEM
            system = SYSTEM
    except (ValueError, TypeError):
        pass
    return [dict(role="system", content=system), dict(role="user", content=state_description)]


@dataclass
class GuidanceOutput:
    """求解引导输出"""
    warm_start: Dict[str, int]  # 变量名 -> 建议值
    variable_fixing: Dict[str, int]  # 可固定的变量
    focus_agents: List[int]  # 需要重点关注的agent
    focus_nodes: List[int]  # 需要重点关注的节点
    strategy_hints: List[str]  # 策略建议
    confidence: float  # 置信度
    generation_time_ms: float
    branching_priority: Dict[str, int] = field(default_factory=dict)


@dataclass
class TrainingExample:
    """训练样本"""
    input_state: Dict[str, Any]  # 输入状态
    guidance: GuidanceOutput  # 生成的引导
    solve_result: Dict[str, Any]  # 求解结果
    reward: float  # 奖励信号
    timestamp: str


class SmallModelGuidance:
    """小规模语言模型求解引导
    
    特点：
    - 参数量小（如7B-14B），推理速度快
    - 专门针对调度问题微调
    - 输出结构化的求解引导
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "qwen2.5-7b-instruct",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        use_local: bool = True,
        native_max_groups: int = 8,
        backend: str = "auto",
        tokenizer_path: Optional[str] = None,
        request_timeout: float = 30.0,
    ):
        """初始化小模型
        
        Args:
            model_path: 本地模型路径
            api_key: API密钥（使用API时）
            base_url: API地址
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
            use_local: 是否使用本地模型
        """
        if backend not in ("auto", "local", "vllm"):
            raise ValueError("Unknown small-model backend")
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("Small-model request timeout must be finite and positive")
        self.backend = backend
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.request_timeout = request_timeout
        self.api_key = api_key or ("EMPTY" if backend == "vllm" else None)
        self.base_url = base_url or ("http://127.0.0.1:8001/v1" if backend == "vllm" else None)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local = backend == "local" or (backend == "auto" and use_local)
        self.native_input_encoding = None
        self.native_context_tokens = 8192
        if native_max_groups < 1:
            raise ValueError("native_max_groups must be positive")
        self.native_max_groups = native_max_groups
        if model_path:
            manifest = Path(model_path)/"sft_manifest.json"
            if manifest.exists():
                prepared = json.loads(manifest.read_text())
                self.native_input_encoding = prepared.get("input_encoding")
                self.native_context_tokens = prepared.get("max_length", 8192)
        
        # 初始化模型
        self._model = None
        self._tokenizer = None
        self._llm_client = None
        
        if backend == "vllm":
            if not model_path or not (Path(model_path)/"adapter_config.json").is_file():
                raise ValueError("vLLM requires --small_model_path pointing to the SFT adapter metadata")
            if self.native_input_encoding != "metro_guidance_group_v2":
                raise ValueError("vLLM native guidance requires a metro_guidance_group_v2 SFT manifest")
            if type(self.native_context_tokens) is not int or self.native_context_tokens <= 0:
                raise ValueError("Invalid SFT context limit")
            self._init_tokenizer(local_files_only=True)
            self._init_api_client()
        elif self.use_local and model_path:
            self._init_local_model()
        elif self.api_key:
            if self.native_input_encoding == "metro_guidance_group_v2":
                self._init_tokenizer(local_files_only=True)
            self._init_api_client()
        
        # 提示模板（针对小模型优化）
        self.prompt_template = self._create_prompt_template()
        
        # 统计
        self.stats = {
            "total_calls": 0,
            "total_time_ms": 0.0,
            "avg_time_ms": 0.0,
            "cache_hits": 0,
            "api_requests": 0,
            "api_errors": 0,
        }
        
        # 简单缓存
        self._cache: Dict[str, GuidanceOutput] = {}
        self._cache_max_size = 1000

    def _init_tokenizer(self, *, local_files_only=False):
        from transformers import AutoTokenizer
        kwargs = dict(trust_remote_code=True, local_files_only=local_files_only)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, fix_mistral_regex=True, **kwargs)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, **kwargs)
        self._tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    def _select_least_used_gpu(self) -> Optional[str]:
        """返回显存占用最低的 GPU 编号字符串。"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            logger.warning(f"无法查询 GPU 占用，保持当前设备可见性: {exc}")
            return None

        candidates: List[Tuple[int, int, int]] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                index_str, used_str, total_str = [part.strip() for part in line.split(",")[:3]]
                candidates.append((int(index_str), int(used_str), int(total_str)))
            except (TypeError, ValueError):
                continue

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[1], item[1] / max(item[2], 1), item[0]))
        selected_index, used_mem, total_mem = candidates[0]
        logger.info(
            "自动选择单卡运行: GPU %s (used=%s MiB / total=%s MiB)",
            selected_index,
            used_mem,
            total_mem,
        )
        return str(selected_index)

    def _configure_single_gpu(self) -> None:
        """自动限制为单卡运行，优先选择显存占用最低的 GPU。"""
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            logger.info("检测到已设置 CUDA_VISIBLE_DEVICES=%s，保持现有配置", visible_devices)
            return

        selected_gpu = self._select_least_used_gpu()
        if selected_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    def _init_local_model(self):
        """初始化本地模型"""
        self._configure_single_gpu()

        try:
            from transformers import AutoConfig, AutoModelForCausalLM
            import torch
        except ImportError as e:
            logger.warning(f"导入 transformers/torch 失败，回退到API模式: {e}")
            self._init_api_client()
            return

        try:
            logger.info(f"加载本地模型: {self.model_path}")
            
            self._init_tokenizer()
            
            from pathlib import Path
            adapter_config = Path(self.model_path) / "adapter_config.json"
            base_path = json.loads(adapter_config.read_text())["base_model_name_or_path"] if adapter_config.exists() else self.model_path
            config = AutoConfig.from_pretrained(base_path)
            model_class = AutoModelForCausalLM
            if any("ForConditionalGeneration" in name for name in (config.architectures or [])):
                from transformers import AutoModelForImageTextToText
                model_class = AutoModelForImageTextToText
            loaded_model = model_class.from_pretrained(
                base_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            if adapter_config.exists():
                from peft import PeftModel
                loaded_model = PeftModel.from_pretrained(loaded_model, self.model_path)
            self._model = loaded_model
            
            logger.info("本地模型加载完成")
        except ImportError as e:
            if "causal_conv1d" in str(e):
                logger.error(
                    "加载本地模型失败: causal_conv1d 与当前 torch/CUDA 环境不兼容，请在 torch 环境中重装或卸载 causal-conv1d 后重试。原始错误: %s",
                    e,
                )
            else:
                logger.error(f"加载本地模型时触发 ImportError: {e}")
            self._init_api_client()
        except Exception as e:
            logger.error(f"加载本地模型失败: {e}")
            self._init_api_client()
    
    def _init_api_client(self):
        """初始化API客户端"""
        if self.backend == "vllm":
            import httpx
            from openai import OpenAI
            # The external LLM uses the HPC proxy; the local model service does not.
            self._llm_client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                timeout=self.request_timeout, max_retries=0,
                http_client=httpx.Client(trust_env=False, timeout=self.request_timeout))
            try:
                models = self._llm_client.models.list().data
                selected = next((m for m in models if m.id == self.model_name), None)
                if selected is None or not getattr(selected, "parent", None):
                    raise ValueError(f"vLLM must serve the LoRA adapter {self.model_name!r}, not the base model")
                root = getattr(selected, "root", None)
                if root and Path(root).resolve() != Path(self.model_path).resolve():
                    raise ValueError("The served adapter path differs from the benchmark adapter metadata")
            except Exception:
                self._llm_client.close()
                self._llm_client = None
                raise
            logger.info("vLLM LoRA service ready: %s (%s)", self.model_name, self.base_url)
            return
        if not self.api_key:
            logger.warning("未提供API密钥，小模型将返回默认引导")
            return
        
        try:
            from langchain_openai import ChatOpenAI
            
            self._llm_client = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key,
                base_url=self.base_url,
                max_tokens=self.max_tokens,
                max_retries=2,
                request_timeout=self.request_timeout,
            )
            
            logger.info(f"API客户端初始化完成: {self.model_name}")
        except ImportError:
            logger.warning("langchain_openai未安装")
        except Exception as e:
            logger.error(f"初始化API客户端失败: {e}")

    def _build_api_request_kwargs(self, max_tokens: int) -> Dict[str, Any]:
        """构造 API 调用参数。"""
        request_kwargs: Dict[str, Any] = {"max_tokens": max_tokens}

        base_url = (self.base_url or "").lower()
        model_name = (self.model_name or "").lower()
        is_local_vllm_qwen = (
            ("127.0.0.1" in base_url or "localhost" in base_url)
            and "qwen" in model_name
        )

        if self.backend == "vllm" or is_local_vllm_qwen:
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        if self.backend == "vllm":
            request_kwargs["response_format"] = {"type": "json_object"}

        return request_kwargs
    
    def _create_prompt_template(self) -> str:
        """创建针对小模型优化的提示模板

        与 generate_finetune_data.py 的 GUIDANCE_SYSTEM_PROMPT 完全对齐，
        保证 SFT 训练数据和推理使用同一 prompt 格式。
        """
        # system prompt 部分（与训练数据共用）
        self._system_prompt = (
            "你是一个地铁调度MILP求解加速专家。根据当前网络状态，输出求解引导。\n\n"
            "**任务**:\n"
            "1. 识别 focus_agents: 哪些列车需要 MILP 优化（其最优动作可能与常规 MARL 策略不同）\n"
            "2. 识别 focus_nodes: 关键节点范围\n"
            "3. 提供 warm_start 和 variable_fixing 加速 MILP 求解\n"
            "4. 给出策略推理说明\n\n"
            "**focus_agents 判断依据**:\n"
            "- 在中断区或邻近中断的列车\n"
            "- 在高等待站点需要决策的列车\n"
            "- 位置接近可能冲突的列车\n\n"
            "**输出格式 (严格 JSON)**:\n"
            '{\n'
            '  "focus_agents": [0, 2],\n'
            '  "focus_nodes": [13, 14, 15],\n'
            '  "warm_start": {"stay[0,13,0]": 1},\n'
            '  "variable_fixing": {"stay[0,13,0]": 1},\n'
            '  "strategy_reasoning": "列车0邻近中断需等待，列车2在高等待站点"\n'
            '}\n\n'
            "**输出约束**:\n"
            "- 不要输出 Thinking Process、思考过程、分析过程\n"
            "- 不要输出 markdown 代码块\n"
            "- 只输出一个 JSON 对象，JSON 外不要有任何文本\n\n"
            "**变量命名规则**:\n"
            "- stay[agent_idx, node_idx, t]: 列车在节点停留\n"
            "- x[agent_idx, node_idx, t]: 列车在节点位置\n"
            "- 只用 stay 和 x 变量，不用 move 变量\n"
            "- 只有站台节点才能 stay=1"
        )
        # user prompt 模板（state_description 由调用方填充）
        return self._system_prompt + "\n\n## 当前状态\n__STATE_DESCRIPTION__\n\n## 输出\n"

    def _build_prompt(self, state_description: str, native: bool = False) -> str:
        """构造最终 prompt，避免 JSON 示例中的花括号被 format 误解析。"""
        if native:
            return ("Return only a JSON object with warm_start, variable_fixing and branching_priority. "
                    "Use only names in solver_variables, with binary values for starts/fixes and nonnegative "
                    "integer priorities. Do not output dispatch actions or legacy move/stay variables. "
                    "Leave uncertain variable fixes empty.\n" + state_description)
        return self.prompt_template.replace("__STATE_DESCRIPTION__", state_description, 1)
    
    def generate_guidance(
        self,
        state_description: str,
        state_dict: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        time_budget_s: Optional[float] = None,
    ) -> GuidanceOutput:
        """生成求解引导
        
        Args:
            state_description: 状态描述文本
            state_dict: 结构化状态字典
            max_tokens: 可选的临时生成长度上限
        
        Returns:
            GuidanceOutput
        """
        start_time = time.perf_counter()
        if time_budget_s is not None and (not math.isfinite(time_budget_s) or time_budget_s <= 0):
            raise ValueError("Guidance budget must be finite and positive")
        deadline = math.inf if time_budget_s is None else start_time + time_budget_s
        generation_max_tokens = int(max_tokens) if max_tokens is not None else self.max_tokens
        
        # 检查缓存
        cache_key = self._get_cache_key(state_description)
        if cache_key in self._cache:
            self.stats["cache_hits"] += 1
            cached = self._cache[cache_key]
            cached.generation_time_ms = 0.0
            return cached
        
        # 生成引导
        native = bool(state_dict and "solver_variables" in state_dict)
        if native and self.native_input_encoding == "metro_guidance_group_v2":
            guidance = self._generate_grouped(json.loads(state_description), generation_max_tokens, deadline=deadline)
        elif self._model is not None:
            guidance = self._generate_local(state_description, generation_max_tokens, native=native)
        elif self._llm_client is not None:
            guidance = self._generate_api(state_description, generation_max_tokens, native=native,
                timeout_s=None if time_budget_s is None else max(1e-6, deadline-time.perf_counter()))
        else:
            guidance = self._generate_default(state_dict)
        
        generation_time = (time.perf_counter() - start_time) * 1000
        guidance.generation_time_ms = generation_time
        
        # 更新统计
        self.stats["total_calls"] += 1
        self.stats["total_time_ms"] += generation_time
        self.stats["avg_time_ms"] = self.stats["total_time_ms"] / self.stats["total_calls"]
        
        # 缓存结果
        if "guidance_time_budget_exhausted" not in guidance.strategy_hints:
            self._cache_result(cache_key, guidance)
        
        return guidance

    def _generate_grouped(self, payload, max_tokens, *, deadline=math.inf):
        from LLM.finetune.guidance_view import group_queries, group_messages, chat_tokens, merge_guidance, relationship_map
        if self._tokenizer is None:
            raise ValueError("Bounded native guidance requires the adapter's target tokenizer")
        max_tokens = min(max_tokens, 512)
        queries, predictions, skipped = [], [], []
        pending = list(group_queries(payload))
        def relevance(item):
            group, query = item
            vehicle = query.get("vehicle") or {}
            coupling = query["coupling"]
            return (-int(vehicle.get("failed", False) or vehicle.get("needs_inspection", False)),
                    -len(query["participants"]), -len(coupling["event_ids"]),
                    -len(relationship_map(coupling["relationships"])), group)
        pending.sort(key=relevance)
        from MILP.guidance_labels import round_robin
        pending = list(round_robin(pending, lambda item: item[0].startswith("platform_")))
        skipped.extend(g for g, _ in pending[self.native_max_groups:])
        exhausted = False
        # Guidance is partial: unqueried groups remain in the exact MILP domain.
        for index, (group, query) in enumerate(pending[:self.native_max_groups]):
            if chat_tokens(self._tokenizer, group_messages(query), generation=True) + max_tokens > self.native_context_tokens:
                skipped.append(group)
                continue
            text = json.dumps(query, ensure_ascii=False, separators=(",", ":"))
            available = deadline-time.perf_counter()
            if available <= 0:
                exhausted = True
                skipped.extend(g for g, _ in pending[index:self.native_max_groups])
                break
            try:
                prediction = (self._generate_local(text, max_tokens, native=True) if self._model is not None
                              else self._generate_api(text, max_tokens, native=True,
                                  timeout_s=None if math.isinf(deadline) else min(self.request_timeout, available)))
            except TimeoutError:
                exhausted = True
                skipped.extend(g for g, _ in pending[index:self.native_max_groups])
                break
            values = {k: getattr(prediction, k) for k in ("warm_start", "variable_fixing", "branching_priority")}
            try:
                merged = merge_guidance([query], [values])
            except ValueError:
                if self.backend == "vllm":
                    raise
                skipped.append(group)
                continue
            queries.append(query)
            predictions.append(merged)
        merged = merge_guidance(queries, predictions)
        return GuidanceOutput(**merged, focus_agents=[], focus_nodes=[],
            strategy_hints=[f"bounded_guidance_skipped_groups={len(skipped)}"] +
                (["guidance_time_budget_exhausted"] if exhausted else []),
            confidence=0. if not queries else .5, generation_time_ms=0.)
    
    def _generate_local(self, state_description: str, max_tokens: int, native: bool = False) -> GuidanceOutput:
        """使用本地模型生成"""
        import torch
        from LLM.finetune.chat_format import render_chat
        
        prompt = (render_chat(self._tokenizer, native_guidance_messages(state_description), generation=True)
                  if native else self._build_prompt(state_description))
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=not native).to(self._model.device)
        do_sample = self.temperature is not None and float(self.temperature) > 0.0
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(self.temperature)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
            )
        
        response = self._tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        
        return self._parse_response(response)
    
    def _generate_api(self, state_description: str, max_tokens: int, native: bool = False,
                      *, timeout_s=None) -> GuidanceOutput:
        """使用API生成"""
        prompt = self._build_prompt(state_description, native=native)
        
        try:
            if self.backend == "vllm":
                self.stats["api_requests"] += 1
                options = dict(model=self.model_name, temperature=self.temperature,
                               **self._build_api_request_kwargs(max_tokens))
                if timeout_s is None:
                    response = self._llm_client.chat.completions.create(
                        messages=native_guidance_messages(state_description), **options)
                else:
                    from LLM.strategic_api import complete
                    response = complete(api_key=self.api_key, base_url=self.base_url, trust_env=False,
                        messages=native_guidance_messages(state_description), options=options,
                        timeout_s=min(timeout_s, self.request_timeout))
                choice = response.choices[0]
                if choice.finish_reason != "stop":
                    raise ValueError(f"Incomplete vLLM guidance: finish_reason={choice.finish_reason}")
                content = choice.message.content
                parsed = json.loads(content)
                keys = ("warm_start", "variable_fixing", "branching_priority")
                if not isinstance(parsed, dict) or any(not isinstance(parsed.get(k), dict) for k in keys):
                    raise ValueError("vLLM guidance must contain three variable-guidance mappings")
                query = json.loads(state_description)
                if query.get("encoding") == "metro_guidance_group_v2":
                    from LLM.finetune.guidance_view import validate_guidance
                    validate_guidance(query, parsed)
                return self._parse_response(content)
            request_llm = self._llm_client.bind(**self._build_api_request_kwargs(max_tokens))
            response = request_llm.invoke(native_guidance_messages(state_description) if native else prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return self._parse_response(content)
        except Exception as e:
            self.stats["api_errors"] += 1
            if isinstance(e, TimeoutError) and timeout_s is not None:
                raise
            if self.backend == "vllm":
                raise RuntimeError(f"vLLM guidance request failed: {e}") from e
            logger.warning(f"API调用失败: {e}")
            return self._generate_default(None)
    
    def _generate_default(
        self,
        state_dict: Optional[Dict[str, Any]],
    ) -> GuidanceOutput:
        """生成默认引导（无模型时）"""
        focus_agents = []
        focus_nodes = []
        warm_start = {}
        variable_fixing = {}
        hints = []
        
        if state_dict:
            # 从状态中提取基本信息（兼容 collect_enhanced_rag_cases 和 train_online_rl 两种格式）
            disruptions = state_dict.get("disruptions", [])
            high_wait_nodes = state_dict.get("high_wait_nodes", [])
            agent_positions = state_dict.get("agent_positions") or state_dict.get("agent_states", [])
            
            # 中断相关agent
            disrupted_nodes = set()
            for d in disruptions:
                nodes = d.get("affected_nodes", [])
                disrupted_nodes.update(nodes)
            
            for i, agent in enumerate(agent_positions):
                pos = agent.get("node_id", agent.get("position", -1))
                if pos in disrupted_nodes:
                    focus_agents.append(i)
                
                # 停站中的agent
                if agent.get("dwell_time", 0) < agent.get("expected_dwell", 0):
                    node_idx = pos
                    var_name = f"stay[{i},{node_idx},0]"
                    warm_start[var_name] = 1
                    variable_fixing[var_name] = 1
            
            # 高等待节点（兼容 node_idx/waiting 和 node_id/count 两种键名）
            for n in high_wait_nodes[:10]:
                if isinstance(n, dict):
                    nid = n.get("node_idx", n.get("node_id", -1))
                else:
                    nid = n
                if nid >= 0:
                    focus_nodes.append(nid)
            
            if disruptions:
                hints.append("检测到中断，建议使用避让策略")
            if high_wait_nodes:
                hints.append(f"有{len(high_wait_nodes)}个高等待站点需要关注")
        
        return GuidanceOutput(
            warm_start=warm_start,
            variable_fixing=variable_fixing,
            focus_agents=focus_agents,
            focus_nodes=focus_nodes,
            strategy_hints=hints,
            confidence=0.5,  # 默认引导置信度较低
            generation_time_ms=0.0,
        )
    
    def _parse_response(self, response: str) -> GuidanceOutput:
        """解析模型响应"""
        raw_response = response or ""
        try:
            response = raw_response.strip()

            candidates: List[str] = []
            fence_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", response, flags=re.IGNORECASE)
            candidates.extend(block.strip() for block in fence_blocks if block and block.strip())
            candidates.append(response)

            parsed_data: Optional[Dict[str, Any]] = None
            for candidate in candidates:
                candidate = candidate.strip()
                if not candidate:
                    continue

                # 尝试截取首个完整 JSON 对象，兼容前后混入文本。
                first_brace = candidate.find("{")
                if first_brace >= 0:
                    candidate = candidate[first_brace:]

                if candidate.startswith("{"):
                    depth = 0
                    end_index = -1
                    for index, char in enumerate(candidate):
                        if char == "{":
                            depth += 1
                        elif char == "}":
                            depth -= 1
                            if depth == 0:
                                end_index = index + 1
                                break
                    if end_index > 0:
                        candidate = candidate[:end_index]

                # 清理非法控制字符（保留常见空白）。
                candidate = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", candidate)

                # 多策略解析：原文、宽松、修复引号/尾逗号。
                parse_trials = [
                    candidate,
                    candidate.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'"),
                ]
                parse_trials.append(re.sub(r",\s*([}\]])", r"\1", parse_trials[-1]))

                for trial in parse_trials:
                    for strict_flag in (True, False):
                        try:
                            parsed = json.loads(trial, strict=strict_flag)
                            if isinstance(parsed, dict):
                                parsed_data = parsed
                                break
                        except json.JSONDecodeError:
                            continue
                    if parsed_data is not None:
                        break

                if parsed_data is not None:
                    break

            data = parsed_data

            # 最后兜底：字段级正则提取，避免直接降级默认引导。
            if data is None:
                data = {}

                def _extract_int_list(field_name: str) -> List[int]:
                    match = re.search(rf'"?{field_name}"?\s*:\s*\[([^\]]*)\]', response)
                    if not match:
                        return []
                    values = re.findall(r"-?\d+", match.group(1))
                    return [int(v) for v in values]

                def _extract_flat_int_dict(field_name: str) -> Dict[str, int]:
                    match = re.search(rf'"?{field_name}"?\s*:\s*\{{([\s\S]*?)\}}', response)
                    if not match:
                        return {}
                    body = match.group(1)
                    pairs = re.findall(r'"([^"\\]+)"\s*:\s*(-?\d+)', body)
                    return {k: int(v) for k, v in pairs}

                data["focus_agents"] = _extract_int_list("focus_agents")
                data["focus_nodes"] = _extract_int_list("focus_nodes")
                data["warm_start"] = _extract_flat_int_dict("warm_start")
                data["variable_fixing"] = _extract_flat_int_dict("variable_fixing")

                reasoning_match = re.search(r'"?strategy_reasoning"?\s*:\s*"([\s\S]*?)"', response)
                if reasoning_match:
                    data["strategy_reasoning"] = reasoning_match.group(1).strip()

                # 仅当兜底仍无有效字段时，记录warning。
                if not (
                    data.get("focus_agents")
                    or data.get("focus_nodes")
                    or data.get("warm_start")
                    or data.get("variable_fixing")
                ):
                    preview = raw_response.replace("\n", " ")[:220]
                    logger.warning("解析响应失败: 无法提取有效JSON字段, raw_preview=%s", preview)

            strategy_hints = data.get("hints", data.get("strategy_hints", []))
            if isinstance(strategy_hints, str):
                strategy_hints = [strategy_hints]

            strategy_reasoning = data.get("strategy_reasoning")
            if isinstance(strategy_reasoning, str) and strategy_reasoning.strip():
                strategy_hints = list(strategy_hints) + [strategy_reasoning.strip()]
            
            return GuidanceOutput(
                warm_start=data.get("warm_start", {}),
                variable_fixing=data.get("variable_fixing", {}),
                branching_priority=data.get("branching_priority", {}),
                focus_agents=data.get("focus_agents", []),
                focus_nodes=data.get("focus_nodes", []),
                strategy_hints=strategy_hints,
                confidence=data.get("confidence", 0.7),
                generation_time_ms=0.0,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            preview = raw_response.replace("\n", " ")[:220]
            logger.warning("解析响应失败: %s, raw_preview=%s", e, preview)
            return self._generate_default(None)
    
    def _get_cache_key(self, state_description: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(state_description.encode()).hexdigest()[:16]
    
    def _cache_result(self, key: str, guidance: GuidanceOutput):
        """缓存结果"""
        if len(self._cache) >= self._cache_max_size:
            # 简单的LRU：删除第一个
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[key] = guidance
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "cache_size": len(self._cache),
            "model_type": "local" if self._model else "api" if self._llm_client else "default",
        }


class KnowledgeDistillation:
    """知识蒸馏：从大模型学习求解策略
    
    收集大模型的输出作为训练数据，用于微调小模型。
    """
    
    def __init__(
        self,
        output_dir: str = "distillation_data",
        max_examples: int = 10000,
    ):
        self.output_dir = output_dir
        self.max_examples = max_examples
        
        # 训练样本缓存
        self.examples: deque[TrainingExample] = deque(maxlen=max_examples)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def record_teacher_output(
        self,
        input_state: Dict[str, Any],
        teacher_guidance: Dict[str, Any],
        solve_result: Dict[str, Any],
        reward: float,
    ):
        """记录教师模型（大模型）输出
        
        Args:
            input_state: 输入状态
            teacher_guidance: 大模型生成的引导
            solve_result: 使用该引导的求解结果
            reward: 奖励信号
        """
        from datetime import datetime
        
        # 转换为GuidanceOutput格式
        guidance = GuidanceOutput(
            warm_start=teacher_guidance.get("warm_start", {}),
            variable_fixing=teacher_guidance.get("variable_fixing", {}),
            focus_agents=teacher_guidance.get("focus_agents", []),
            focus_nodes=teacher_guidance.get("focus_nodes", []),
            strategy_hints=teacher_guidance.get("strategy_hints", []),
            confidence=teacher_guidance.get("confidence", 0.8),
            generation_time_ms=0.0,
        )
        
        example = TrainingExample(
            input_state=input_state,
            guidance=guidance,
            solve_result=solve_result,
            reward=reward,
            timestamp=datetime.now().isoformat(),
        )
        
        self.examples.append(example)
        
        # 定期保存
        if len(self.examples) % 100 == 0:
            self.save_examples()
    
    def save_examples(self, path: Optional[str] = None):
        """保存训练样本"""
        save_path = path or os.path.join(self.output_dir, "distillation_examples.jsonl")
        
        with open(save_path, "w", encoding="utf-8") as f:
            for example in self.examples:
                data = {
                    "input_state": example.input_state,
                    "guidance": {
                        "warm_start": example.guidance.warm_start,
                        "variable_fixing": example.guidance.variable_fixing,
                        "focus_agents": example.guidance.focus_agents,
                        "focus_nodes": example.guidance.focus_nodes,
                        "strategy_hints": example.guidance.strategy_hints,
                        "confidence": example.guidance.confidence,
                    },
                    "solve_result": example.solve_result,
                    "reward": example.reward,
                    "timestamp": example.timestamp,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        logger.info(f"已保存 {len(self.examples)} 条蒸馏样本到 {save_path}")
    
    def load_examples(self, path: Optional[str] = None):
        """加载训练样本"""
        load_path = path or os.path.join(self.output_dir, "distillation_examples.jsonl")
        
        if not os.path.exists(load_path):
            return
        
        self.examples.clear()
        
        with open(load_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                
                guidance = GuidanceOutput(
                    warm_start=data["guidance"]["warm_start"],
                    variable_fixing=data["guidance"]["variable_fixing"],
                    focus_agents=data["guidance"]["focus_agents"],
                    focus_nodes=data["guidance"]["focus_nodes"],
                    strategy_hints=data["guidance"]["strategy_hints"],
                    confidence=data["guidance"]["confidence"],
                    generation_time_ms=0.0,
                )
                
                example = TrainingExample(
                    input_state=data["input_state"],
                    guidance=guidance,
                    solve_result=data["solve_result"],
                    reward=data["reward"],
                    timestamp=data["timestamp"],
                )
                
                self.examples.append(example)
        
        logger.info(f"已加载 {len(self.examples)} 条蒸馏样本")
    
    def export_for_sft(
        self,
        output_path: str,
        min_reward: float = 0.0,
        format: str = "chatml",
    ) -> int:
        """导出用于SFT的训练数据

        支持两种格式：
        - chatml: ChatML messages 格式（与 generate_finetune_data.py 对齐，推荐）
        - alpaca: Alpaca instruction/input/output 格式

        Args:
            output_path: 输出路径（JSONL）
            min_reward: 最小奖励阈值（只保留高质量样本）
            format: 输出格式 ("chatml" | "alpaca")

        Returns:
            导出样本数量
        """
        from LLM.finetune.generate_finetune_data import GUIDANCE_SYSTEM_PROMPT

        sft_records: list = []

        for example in self.examples:
            if example.reward < min_reward:
                continue

            user_content = self._format_state_description(example.input_state)
            guidance_json = self._format_guidance_json(example.guidance)

            if format == "chatml":
                record = {
                    "messages": [
                        {"role": "system", "content": GUIDANCE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": guidance_json},
                    ],
                    "metadata": {
                        "reward": example.reward,
                        "timestamp": example.timestamp,
                        "data_quality": "distilled",
                    },
                }
            else:  # alpaca
                record = {
                    "instruction": GUIDANCE_SYSTEM_PROMPT,
                    "input": user_content,
                    "output": guidance_json,
                }
            sft_records.append(record)

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in sft_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(f"已导出 {len(sft_records)} 条SFT样本到 {output_path} (格式={format})")
        return len(sft_records)

    def export_for_rl(
        self,
        output_path: str,
    ) -> int:
        """导出用于 RL 在线训练的数据

        每条记录包含: state_description, guidance, reward, has_disruption
        与 generate_finetune_data.py 产生的 RL JSONL 格式对齐。

        Returns:
            导出样本数量
        """
        rl_records: list = []

        for example in self.examples:
            user_content = self._format_state_description(example.input_state)
            guidance_dict = {
                "focus_agents": example.guidance.focus_agents,
                "focus_nodes": example.guidance.focus_nodes,
                "warm_start": example.guidance.warm_start,
                "variable_fixing": example.guidance.variable_fixing,
                "strategy_reasoning": (" ; ".join(example.guidance.strategy_hints)
                                        if example.guidance.strategy_hints else ""),
            }
            rl_records.append({
                "state_description": user_content,
                "guidance": guidance_dict,
                "reward": example.reward,
                "has_disruption": bool(example.input_state.get("disruptions")),
            })

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in rl_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(f"已导出 {len(rl_records)} 条RL样本到 {output_path}")
        return len(rl_records)

    def load_sft_jsonl(
        self,
        path: str,
    ) -> int:
        """从 generate_finetune_data.py 产生的 SFT JSONL 导入样本

        将 ChatML 记录转换为内部 TrainingExample 并追加到 self.examples。

        Returns:
            导入样本数量
        """
        from datetime import datetime
        count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                messages = data.get("messages", [])
                meta = data.get("metadata", {})

                # 解析 assistant 的 guidance JSON
                assistant_msg = next(
                    (m for m in messages if m["role"] == "assistant"), None
                )
                if assistant_msg is None:
                    continue
                try:
                    g = json.loads(assistant_msg["content"])
                except json.JSONDecodeError:
                    continue

                guidance = GuidanceOutput(
                    warm_start=g.get("warm_start", {}),
                    variable_fixing=g.get("variable_fixing", {}),
                    focus_agents=g.get("focus_agents", []),
                    focus_nodes=g.get("focus_nodes", []),
                    strategy_hints=[g.get("strategy_reasoning", "")],
                    confidence=0.9,
                    generation_time_ms=0.0,
                )

                user_msg = next(
                    (m for m in messages if m["role"] == "user"), None
                )
                input_state = {
                    "state_description": user_msg["content"] if user_msg else "",
                }

                example = TrainingExample(
                    input_state=input_state,
                    guidance=guidance,
                    solve_result=meta,
                    reward=meta.get("reward", 0.0),
                    timestamp=meta.get("timestamp", datetime.now().isoformat()),
                )
                self.examples.append(example)
                count += 1

        logger.info(f"从 {path} 导入 {count} 条SFT样本")
        return count

    def _format_state_description(self, state: Dict[str, Any]) -> str:
        """格式化状态描述（兼容 collect_enhanced_rag_cases 和 train_online_rl 两种 state_dict 格式）"""
        # 如果 input_state 本身就含有完整的状态描述
        if "state_description" in state:
            return state["state_description"]

        # 否则从结构化字段拼装
        parts: list = []
        parts.append("### 问题规模")
        parts.append(f"- 列车数量: {state.get('n_agents', 0)}")
        parts.append(f"- 节点数量: {state.get('n_nodes', 0)}")
        parts.append(f"- 站台数量: {state.get('n_platforms', 0)}")

        disruptions = state.get("disruptions", [])
        if disruptions:
            parts.append("\n### 区间中断")
            for d in disruptions:
                seg = d.get("segment_id", d.get("segment", "?"))
                dtype = d.get("type", "BLOCK")
                affected = d.get("affected_nodes", [])
                severity = d.get("severity", "")
                line = f"- {seg}: {dtype}"
                if affected:
                    line += f", 受影响节点={affected}"
                if severity:
                    line += f", 严重度={severity}"
                parts.append(line)

        # 兼容 agent_positions (collect 格式) 与 agent_states (online 格式)
        agent_positions = state.get("agent_positions") or state.get("agent_states", [])
        if agent_positions:
            parts.append("\n### 列车状态")
            for i, a in enumerate(agent_positions):
                pos = a.get("node_id", a.get("position", -1))
                pax = a.get("passengers", 0)
                at_plat = a.get("at_platform", "")
                line = f"- 列车{i}: 位置{pos}, 乘客={pax}"
                if at_plat != "":
                    line += f", 在站台={at_plat}"
                parts.append(line)

        # 等待乘客汇总（collect 格式含 waiting_passengers dict）
        waiting_map = state.get("waiting_passengers", {})
        if waiting_map:
            total_w = sum(waiting_map.values())
            max_w = max(waiting_map.values()) if waiting_map else 0
            parts.append(f"\n### 乘客等待汇总")
            parts.append(f"- 总等待: {total_w}, 最大单站: {max_w}")

        # 高等待节点（兼容 node_idx/waiting 和 node_id/count 两种键名）
        high_wait = state.get("high_wait_nodes", [])
        if high_wait:
            parts.append("\n### 高等待站台")
            for n in high_wait[:10]:
                if isinstance(n, int):
                    parts.append(f"- 节点{n}")
                else:
                    nid = n.get("node_idx", n.get("node_id", "?"))
                    cnt = n.get("waiting", n.get("count", 0))
                    parts.append(f"- 节点{nid}: {cnt}人")

        return "\n".join(parts)

    def _format_guidance_json(self, guidance: GuidanceOutput) -> str:
        """格式化 guidance 输出 JSON（与训练数据格式对齐）"""
        return json.dumps({
            "focus_agents": guidance.focus_agents,
            "focus_nodes": guidance.focus_nodes,
            "warm_start": guidance.warm_start,
            "variable_fixing": guidance.variable_fixing,
            "strategy_reasoning": (" ; ".join(guidance.strategy_hints)
                                    if guidance.strategy_hints else ""),
        }, ensure_ascii=False, indent=2)


class OnlineLearner:
    """在线学习器：基于求解反馈持续优化
    
    使用强化学习信号微调小模型的策略。
    """
    
    def __init__(
        self,
        small_model: SmallModelGuidance,
        learning_rate: float = 1e-5,
        buffer_size: int = 1000,
        finetune_scope: str = "head",
        max_train_length: int = 6144,
    ):
        self.small_model = small_model
        self.learning_rate = learning_rate
        self.finetune_scope = finetune_scope
        self.max_train_length = max_train_length
        
        # 经验回放缓冲
        self.replay_buffer: deque[TrainingExample] = deque(maxlen=buffer_size)

        # 训练态资源按需初始化，避免每次更新重复创建优化器和参数状态。
        self._optimizer = None
        self._trainable_param_names: List[str] = []
        
        # 性能追踪
        self.performance_history: deque[float] = deque(maxlen=1000)
        
        # 统计
        self.stats = {
            "update_count": 0,
            "avg_reward": 0.0,
            "improvement": 0.0,
        }

    def _collect_named_parameters(self, module) -> List[Tuple[str, Any]]:
        return list(module.named_parameters())

    def _mark_module_trainable(self, model, module_path: str) -> List[str]:
        if not hasattr(model, "get_submodule"):
            return []

        try:
            module = model.get_submodule(module_path)
        except Exception:
            return []

        matched: List[str] = []
        for param_name, param in module.named_parameters(recurse=False):
            param.requires_grad = True
            matched.append(f"{module_path}.{param_name}")
        return matched

    def _prepare_finetune(self, model) -> List[Any]:
        import torch

        named_parameters = self._collect_named_parameters(model)
        if not named_parameters:
            return []

        for _, param in named_parameters:
            param.requires_grad = (self.finetune_scope == "full")

        matched_names: List[str] = []
        if self.finetune_scope == "full":
            matched_names = [name for name, _ in named_parameters]
        else:
            for head_name in ("lm_head", "score", "classifier"):
                if hasattr(model, head_name):
                    for param_name, param in getattr(model, head_name).named_parameters(recurse=False):
                        param.requires_grad = True
                        matched_names.append(f"{head_name}.{param_name}")

            for module_path in (
                "model.norm",
                "transformer.ln_f",
                "final_layernorm",
                "language_model.model.norm",
            ):
                matched_names.extend(self._mark_module_trainable(model, module_path))

            if not matched_names:
                fallback_name, fallback_param = named_parameters[-1]
                fallback_param.requires_grad = True
                matched_names = [fallback_name]
                logger.warning("未识别到输出头，回退为仅训练最后一个参数张量: %s", fallback_name)

        trainable_params = [param for _, param in named_parameters if param.requires_grad]
        if not trainable_params:
            return []

        self._trainable_param_names = matched_names
        self._optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception as exc:
                logger.debug("启用 gradient checkpointing 失败: %s", exc)

        logger.info(
            "OnlineLearner finetune_scope=%s trainable_tensors=%s",
            self.finetune_scope,
            len(trainable_params),
        )
        return trainable_params
    
    def record_experience(
        self,
        state: Dict[str, Any],
        guidance: GuidanceOutput,
        result: Dict[str, Any],
        reward: float,
    ):
        """记录经验"""
        from datetime import datetime
        
        example = TrainingExample(
            input_state=state,
            guidance=guidance,
            solve_result=result,
            reward=reward,
            timestamp=datetime.now().isoformat(),
        )
        
        self.replay_buffer.append(example)
        self.performance_history.append(reward)
        
        # 更新统计
        if self.performance_history:
            self.stats["avg_reward"] = np.mean(self.performance_history)
    
    def should_update(self, min_samples: int = 32) -> bool:
        """是否应该更新模型"""
        return len(self.replay_buffer) >= min_samples
    
    def update_model(self, batch_size: int = 16) -> Dict[str, Any]:
        """更新模型（策略梯度 + KL 约束）

        使用 REINFORCE 方法计算策略梯度，并对小模型进行一步更新。
        若模型不支持微调（如 API 模型），则只返回统计信息。

        Returns:
            更新统计信息
        """
        if not self.should_update(batch_size):
            return {"updated": False}

        # 采样batch
        indices = np.random.choice(
            len(self.replay_buffer),
            size=min(batch_size, len(self.replay_buffer)),
            replace=False,
        )

        batch = [self.replay_buffer[i] for i in indices]

        # 计算baseline（用于方差降低）
        rewards = [ex.reward for ex in batch]
        baseline = np.mean(rewards)

        # 计算优势
        advantages = [ex.reward - baseline for ex in batch]

        # ---- 本地模型微调 ----
        model = self.small_model._model
        tokenizer = self.small_model._tokenizer

        if model is not None and tokenizer is not None:
            try:
                import torch
                from torch import nn

                trainable_params = [param for param in model.parameters() if param.requires_grad]
                if self._optimizer is None or not trainable_params:
                    trainable_params = self._prepare_finetune(model)

                optimizer = self._optimizer
                if optimizer is None or not trainable_params:
                    raise RuntimeError("没有可训练参数，无法执行RL更新")

                model.train()
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0

                for ex, advantage in zip(batch, advantages):
                    state_desc = self.small_model._build_prompt(
                        json.dumps(ex.input_state, ensure_ascii=False)
                    )
                    target_json = json.dumps(
                        {
                            "focus_agents": ex.guidance.focus_agents,
                            "focus_nodes": ex.guidance.focus_nodes,
                            "warm_start": ex.guidance.warm_start,
                            "variable_fixing": ex.guidance.variable_fixing,
                            "strategy_reasoning": (
                                " ; ".join(ex.guidance.strategy_hints)
                                if ex.guidance.strategy_hints else ""
                            ),
                        },
                        ensure_ascii=False,
                    )
                    full_text = state_desc + target_json
                    inputs = tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_train_length,
                    ).to(model.device)

                    labels = inputs["input_ids"].clone()
                    # 只对 target 部分计算 loss
                    prompt_len = len(
                        tokenizer(
                            state_desc,
                            return_tensors="pt",
                            truncation=True,
                            max_length=self.max_train_length,
                        )["input_ids"][0]
                    )
                    labels[0, : min(prompt_len, labels.shape[1])] = -100

                    outputs = model(**inputs, labels=labels, use_cache=False)
                    # REINFORCE: loss = -advantage * log_prob
                    loss = outputs.loss * (-float(advantage)) / max(len(batch), 1)
                    loss.backward()
                    total_loss += loss.item()

                    del outputs
                    del labels
                    del inputs

                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model.eval()

                self.stats["update_count"] += 1
                return {
                    "updated": True,
                    "batch_size": len(batch),
                    "avg_reward": float(np.mean(rewards)),
                    "avg_advantage": float(np.mean(advantages)),
                    "avg_loss": total_loss / max(len(batch), 1),
                    "finetune_scope": self.finetune_scope,
                    "trainable_tensors": len(self._trainable_param_names),
                }
            except Exception as e:
                if "out of memory" in str(e).lower():
                    try:
                        if self._optimizer is not None:
                            self._optimizer.zero_grad(set_to_none=True)
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                logger.warning(f"RL模型更新失败: {e}")

        # API / 无模型 fallback：只记录统计
        self.stats["update_count"] += 1
        return {
            "updated": False,
            "reason": "no_local_model",
            "batch_size": len(batch),
            "avg_reward": float(np.mean(rewards)),
            "avg_advantage": float(np.mean(advantages)),
        }
    
    def get_performance_trend(self, window: int = 100) -> float:
        """获取性能趋势"""
        if len(self.performance_history) < 2 * window:
            return 0.0
        
        recent = list(self.performance_history)[-window:]
        older = list(self.performance_history)[-2*window:-window]
        
        return np.mean(recent) - np.mean(older)


class MultiModelFusion:
    """多模型融合：小模型快速响应 + 大模型复杂推理
    
    策略：
    1. 小模型总是先响应（快速）
    2. 根据置信度决定是否调用大模型
    3. 融合两个模型的输出
    """
    
    def __init__(
        self,
        small_model: SmallModelGuidance,
        large_model_callback=None,  # 大模型调用回调
        confidence_threshold: float = 0.7,
        complexity_threshold: int = 10,  # agent数量阈值
    ):
        self.small_model = small_model
        self.large_model_callback = large_model_callback
        self.confidence_threshold = confidence_threshold
        self.complexity_threshold = complexity_threshold
        
        # 统计
        self.stats = {
            "small_model_only": 0,
            "large_model_called": 0,
            "fused_decisions": 0,
        }
    
    def generate_guidance(
        self,
        state_description: str,
        state_dict: Dict[str, Any],
        force_large_model: bool = False,
    ) -> Tuple[GuidanceOutput, str]:
        """生成融合引导
        
        Returns:
            (引导, 来源标识)
        """
        # 1. 小模型快速响应
        small_guidance = self.small_model.generate_guidance(
            state_description=state_description,
            state_dict=state_dict,
        )
        
        # 2. 判断是否需要大模型
        need_large_model = (
            force_large_model or
            small_guidance.confidence < self.confidence_threshold or
            self._is_complex_problem(state_dict)
        )
        
        if not need_large_model or self.large_model_callback is None:
            self.stats["small_model_only"] += 1
            return small_guidance, "small_model"
        
        # 3. 调用大模型
        self.stats["large_model_called"] += 1
        
        try:
            large_guidance = self.large_model_callback(
                state_description=state_description,
                state_dict=state_dict,
            )
            
            # 4. 融合输出
            fused = self._fuse_guidance(small_guidance, large_guidance)
            self.stats["fused_decisions"] += 1
            
            return fused, "fused"
        except Exception as e:
            logger.warning(f"大模型调用失败: {e}, 使用小模型结果")
            return small_guidance, "small_model_fallback"
    
    def _is_complex_problem(self, state_dict: Dict[str, Any]) -> bool:
        """判断问题是否复杂"""
        # 规模判断
        n_agents = state_dict.get("n_agents", 0)
        if n_agents > self.complexity_threshold:
            return True
        
        # 中断判断
        disruptions = state_dict.get("disruptions", [])
        if len(disruptions) > 2:
            return True
        
        # 高等待判断
        high_wait = state_dict.get("high_wait_nodes", [])
        if len(high_wait) > 5:
            return True
        
        return False
    
    def _fuse_guidance(
        self,
        small: GuidanceOutput,
        large: Dict[str, Any],
    ) -> GuidanceOutput:
        """融合两个模型的引导
        
        策略：
        - focus_agents/nodes: 取并集
        - warm_start: 优先使用大模型
        - variable_fixing: 取交集（保守策略）
        - hints: 合并
        """
        # 转换大模型输出
        large_warm_start = large.get("warm_start", {})
        large_fixing = large.get("variable_fixing", {})
        large_focus_agents = set(large.get("focus_agents", []))
        large_focus_nodes = set(large.get("focus_nodes", []))
        large_hints = large.get("strategy_hints", large.get("hints", []))
        
        # 融合
        fused_focus_agents = list(set(small.focus_agents) | large_focus_agents)
        fused_focus_nodes = list(set(small.focus_nodes) | large_focus_nodes)
        
        # warm_start: 大模型优先
        fused_warm_start = dict(small.warm_start)
        fused_warm_start.update(large_warm_start)
        
        # variable_fixing: 取交集
        fused_fixing = {}
        for var, val in small.variable_fixing.items():
            if var in large_fixing and large_fixing[var] == val:
                fused_fixing[var] = val
        
        # hints: 合并去重
        all_hints = list(small.strategy_hints) + list(large_hints)
        fused_hints = list(dict.fromkeys(all_hints))  # 保持顺序去重
        
        # 置信度: 加权平均（大模型权重更高）
        large_confidence = large.get("confidence", 0.8)
        fused_confidence = 0.3 * small.confidence + 0.7 * large_confidence
        
        return GuidanceOutput(
            warm_start=fused_warm_start,
            variable_fixing=fused_fixing,
            focus_agents=fused_focus_agents,
            focus_nodes=fused_focus_nodes,
            strategy_hints=fused_hints[:5],  # 限制数量
            confidence=fused_confidence,
            generation_time_ms=small.generation_time_ms,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = (
            self.stats["small_model_only"] +
            self.stats["large_model_called"]
        )
        
        return {
            **self.stats,
            "small_model_ratio": self.stats["small_model_only"] / max(total, 1),
            "large_model_ratio": self.stats["large_model_called"] / max(total, 1),
        }


def create_guidance_system(
    small_model_config: Optional[Dict[str, Any]] = None,
    large_model_callback=None,
    distillation_dir: str = "distillation_data",
) -> Dict[str, Any]:
    """创建完整的求解引导系统
    
    Returns:
        包含各组件的字典
    """
    # 小模型
    small_config = small_model_config or {}
    small_model = SmallModelGuidance(
        model_path=small_config.get("model_path"),
        api_key=small_config.get("api_key"),
        base_url=small_config.get("base_url"),
        model_name=small_config.get("model_name", "qwen2.5-7b-instruct"),
        use_local=small_config.get("use_local", False),
    )
    
    # 知识蒸馏
    distillation = KnowledgeDistillation(output_dir=distillation_dir)
    
    # 在线学习
    online_learner = OnlineLearner(small_model=small_model)
    
    # 多模型融合
    fusion = MultiModelFusion(
        small_model=small_model,
        large_model_callback=large_model_callback,
    )
    
    return {
        "small_model": small_model,
        "distillation": distillation,
        "online_learner": online_learner,
        "fusion": fusion,
    }
