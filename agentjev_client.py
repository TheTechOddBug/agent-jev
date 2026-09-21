"""
AgentJev Client SDK
===================
与 TypeSafe Jev / NanoJev 规范对齐的 Python 客户端。
用于在真实 Agent 循环、CI/CD 门控、运维自动化或多轮对话中快速执行 System One 决策。
"""

import json
import time
import urllib.request
from typing import Dict, List, Any, Optional, Union

class AgentJev:
    def __init__(self, endpoint: str = "http://127.0.0.1:8149"):
        self.endpoint = endpoint.rstrip("/")
        self.eval_url = f"{self.endpoint}/api/evaluate"
        self.info_url = f"{self.endpoint}/api/info"

    def info(self) -> Dict[str, Any]:
        """获取当前服务状态与模型信息"""
        req = urllib.request.Request(self.info_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def evaluate(self, state: Any, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        低阶 API：向 AgentJev 提交一次状态及多个决策问题。
        返回原始 answers 列表及耗时指标。
        """
        # 兼容字符串与字典对象
        if isinstance(state, (dict, list)):
            state_str = json.dumps(state, ensure_ascii=False, indent=2)
        else:
            state_str = str(state)

        payload = json.dumps({
            "state": state_str,
            "questions": questions
        }).encode("utf-8")

        req = urllib.request.Request(
            self.eval_url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        wall_ms = (time.time() - t0) * 1000

        result = data["results"][0]
        result["network_wall_ms"] = round(wall_ms, 2)
        result["server_usage"] = data.get("usage", {})
        return result

    def decide_boolean(self, state: Any, question: str, criteria: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        高阶 API 1：真假判断 (Boolean / noul)
        用于任务是否完成、测试是否通过、是否需要回滚或人工介入等场景。
        """
        q = {
            "id": "bool_decision",
            "type": "boolean",
            "question": question,
            "criteria": criteria or {"true": "The statement is true.", "false": "The statement is false."}
        }
        res = self.evaluate(state, [q])
        ans = res["answers"][0]
        return {
            "decision": ans["value"],            # True 或 False
            "prob_true": ans["distribution"]["true"],
            "prob_false": ans["distribution"]["false"],
            "confidence": max(ans["distribution"].values()),
            "wall_ms": res["network_wall_ms"]
        }

    def decide_choice(self, state: Any, question: str, options: Dict[str, str]) -> Dict[str, Any]:
        """
        高阶 API 2：动态候选决策 (Choice)
        从多个可行动作中选出最优方案，返回命中选项及置信度差 (margin)。
        """
        q = {
            "id": "choice_decision",
            "type": "choice",
            "question": question,
            "options": options
        }
        res = self.evaluate(state, [q])
        ans = res["answers"][0]
        dist = ans["distribution"]
        sorted_options = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        top_key, top_prob = sorted_options[0]
        second_prob = sorted_options[1][1] if len(sorted_options) > 1 else 0.0

        return {
            "best_action": top_key,
            "description": options.get(top_key, ""),
            "probability": top_prob,
            "margin": round(top_prob - second_prob, 4),  # 置信优势 (大于 0.2 通常表明极其坚决)
            "distribution": dist,
            "wall_ms": res["network_wall_ms"]
        }

    def score(self, state: Any, question: str, levels: List[str]) -> Dict[str, Any]:
        """
        高阶 API 3：等级评分 (Score)
        返回离散等级及连续期望得分 (0.0 ~ N-1.0)。
        """
        q = {
            "id": "score_decision",
            "type": "score",
            "question": question,
            "levels": levels
        }
        res = self.evaluate(state, [q])
        ans = res["answers"][0]
        return {
            "expected_score": round(ans["score"], 3),
            "level": ans["level"],
            "level_description": levels[ans["level"]],
            "distribution": ans["distribution"],
            "wall_ms": res["network_wall_ms"]
        }
