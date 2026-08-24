# -*- coding: utf-8 -*-
"""风险 QLoRA 隔离推理服务（生产接入的模型侧端点）。

以独立进程常驻加载合并后的 QLoRA 模型，向 FastAPI 生产应用暴露
POST /assess 与 GET /health。生产进程不加载 torch/transformers，
仅通过 HTTP + JSON 契约调用，超时/异常由调用方回退规则通道。

运行（隔离环境，D 盘）：
  D:/AegisTraining/envs/qlora-qwen35/python.exe serve_risk_qlora.py \
      --model-dir D:/AegisTraining/exports/aegis-risk-qwen3.5-2b-v9-merged \
      --host 127.0.0.1 --port 8301

与验收评测同源：system prompt 取自 aegis_training.data_contract（v2 契约），
temperature=0、max_new_tokens=64、宽容 JSON 解析。
生产部署默认 bf16（与验收口径一致）；--load-4bit 仅在显存紧张时使用，
注意 4-bit 推理可能偏移个别边界预测（高风险通道不建议）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Configure the checkout root through AEGIS_TRAINING_ROOT when needed.
TRAINING_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from aegis_training.data_contract import RISK_SYSTEM_PROMPT  # noqa: E402
from aegis_training.paths import training_root, under  # noqa: E402

TRAIN_ROOT = training_root()


def _guard(path: Path, allowed_roots: tuple, role: str) -> Path:
    for root in allowed_roots:
        try:
            return under(path, root, role)
        except ValueError:
            continue
    raise ValueError(f"{role} path escapes allowed roots: {path.resolve()}")


def parse_risk_json(content: str) -> dict | None:
    """与生产 _parse_risk_json 同逻辑的宽容解析；失败返回 None。"""
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text.strip()[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    level = str(data.get("risk_level", "")).strip().lower()
    if level not in {"low", "medium", "high"}:
        return None
    return {"risk_level": level, "reason": str(data.get("reason", ""))[:120]}


class ModelRunner:
    """加载合并模型并执行单条风险评估推理。"""

    def __init__(self, model_dir: Path, load_4bit: bool, max_new_tokens: int = 64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
        dtype = torch.bfloat16
        if load_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), quantization_config=quant, device_map="auto",
                trust_remote_code=False)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), torch_dtype=dtype, device_map="auto",
                trust_remote_code=False)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.call_count = 0

    def assess(self, message: str) -> dict | None:
        messages = [
            {"role": "system", "content": RISK_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt",
                                 truncation=True, max_length=768).to(self.model.device)
        with self.torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()
        self.call_count += 1
        return parse_risk_json(raw)


def build_handler(runner: ModelRunner):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 精简默认日志
            sys.stderr.write("[serve] " + (fmt % args) + "\n")

        def _send(self, code: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "calls": runner.call_count})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/assess":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                message = str(payload.get("message", "")).strip()
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "invalid json"})
                return
            if not message:
                self._send(400, {"error": "message required"})
                return
            started = time.perf_counter()
            try:
                out = runner.assess(message)
            except Exception as exc:  # 推理异常不让进程退出
                self._send(500, {"error": f"inference failed: {exc}"})
                return
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            if out is None:
                self._send(200, {"risk_level": None, "reason": "",
                                 "latency_ms": latency_ms})
            else:
                out["latency_ms"] = latency_ms
                self._send(200, out)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Aegis risk QLoRA inference service")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="默认取 AEGIS_QLORA_MODEL_DIR 或训练根 exports 下的 v9 合并目录")
    parser.add_argument("--host", default=os.environ.get("AEGIS_QLORA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AEGIS_QLORA_PORT", "8301")))
    parser.add_argument("--load-4bit", action="store_true",
                        help="4-bit NF4 加载（与 Ollama 共存时降低显存占用）")
    args = parser.parse_args()

    if args.model_dir is None:
        env_model = os.environ.get("AEGIS_QLORA_MODEL_DIR")
        args.model_dir = Path(env_model) if env_model else TRAIN_ROOT / "exports" / "aegis-risk-qwen3.5-2b-v9-merged"
    model_dir = _guard(args.model_dir, (TRAIN_ROOT,), "model-dir")
    print(f"loading model from {model_dir} (4bit={args.load_4bit}) ...", flush=True)
    runner = ModelRunner(model_dir, args.load_4bit)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(runner))
    print(f"risk-qlora service listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
