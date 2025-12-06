"""
简要介绍 (ZH):
  解释性推理运行器。将激活的概念图转化为 VLM 编码需要的输入（image_tensor + prompts），
  在推理层注入训练好的概念嵌入，并统一执行生成、可视化及结果整理。

Overview (EN):
  End-to-end reasoning runner. Normalizes activated concept graphs, injects learned concept embeddings
  into the VLM wrapper, builds prompts, triggers generation, and aggregates structured outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from concept_graph.graph_viz import draw_graph
from concept_graph.reasoning import inference_prompt_templates
from inference import inference_utils


def _normalize_key(path_like: Union[str, Path]) -> str:
    """Return a stable absolute string key for any path-like input."""
    if isinstance(path_like, Path):
        candidate = path_like
    else:
        candidate = Path(str(path_like))
    try:
        candidate = candidate.expanduser().resolve()
    except Exception:
        candidate = candidate.expanduser()
    return str(candidate)


def _normalize_activations(activated_concepts: Any) -> Dict[str, Dict[str, Any]]:
    """
    Support both dict[str, …] and list[{image_path: …}] formats and normalize keys to absolute paths.
    """
    if not activated_concepts:
        return {}
    if isinstance(activated_concepts, Mapping):
        return {_normalize_key(k): v for k, v in activated_concepts.items()}
    normalized: Dict[str, Dict[str, Any]] = {}
    for record in activated_concepts:
        if not isinstance(record, Mapping):
            continue
        img_path = record.get("image_path")
        if img_path is None:
            continue
        normalized[_normalize_key(img_path)] = record
    return normalized


def _iter_concept_embedding_batches(concept_embeddings: Any) -> List[Tuple[Any, Any]]:
    """
    Convert checkpoint payloads into a list of (iteration_identifier, payload) tuples.
    Handles:
      - dicts with 'keys'/'values' (single payload)
      - dict[int -> payload] produced by torch.load checkpoints
      - sequences of payloads
      - None (falls back to a single default iteration with no injection)
    """
    if concept_embeddings is None:
        return [(None, None)]
    if isinstance(concept_embeddings, Mapping):
        if "keys" in concept_embeddings and "values" in concept_embeddings:
            return [(None, concept_embeddings)]
        return list(concept_embeddings.items())
    if isinstance(concept_embeddings, Sequence) and not isinstance(concept_embeddings, (str, bytes)):
        return list(enumerate(concept_embeddings))
    return [(None, concept_embeddings)]


def _format_iteration_key(iteration_label: Any) -> str:
    if iteration_label is None:
        return "iteration_default"
    if isinstance(iteration_label, str):
        if iteration_label.startswith("iteration"):
            return iteration_label
        return f"iteration_{iteration_label}"
    return f"iteration_{iteration_label}"


def _iteration_to_int(iteration_label: Any) -> int:
    if isinstance(iteration_label, int):
        return iteration_label
    if isinstance(iteration_label, str):
        suffix = iteration_label.split("iteration_", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            pass
        try:
            return int(iteration_label)
        except ValueError:
            return -1
    return -1


def _prepare_image_list(images: Optional[Iterable[Any]], activation_lookup: Dict[str, Dict[str, Any]]) -> List[Path]:
    """
    Convert provided images iterable into Path objects; if empty, fall back to keys present in activation lookup.
    """
    resolved: List[Path] = []
    if images:
        for item in images:
            if isinstance(item, Path):
                resolved.append(item)
            elif isinstance(item, Mapping) and "image_path" in item:
                resolved.append(Path(str(item["image_path"])))
            else:
                resolved.append(Path(str(item)))
    else:
        resolved = [Path(k) for k in activation_lookup.keys()]
    return resolved


def _extract_concept_signal(record: Dict[str, Any]) -> Any:
    for key in ("concept_signals", "concept_signal", "signals"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def _build_prompts(record: Dict[str, Any], cfg: Any) -> List[str]:
    custom_builder = getattr(cfg, "prompt_builder", None)
    structured_cfg = getattr(cfg, "structured_prompt_cfg", None)
    if callable(custom_builder):
        prompts = custom_builder(record)
    else:
        prompt_mode = getattr(cfg, "prompt_mode", None)
        if prompt_mode in {"natural", "structured"}:
            prompts = [
                inference_prompt_templates.build_prompt(
                    activated_concepts=record,
                    mode=prompt_mode,
                    structured_cfg=structured_cfg,
                )
            ]
        else:
            prompts = inference_prompt_templates.get_prompts(record)
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = [p for p in prompts if isinstance(p, str) and p.strip()]
    max_prompts = getattr(cfg, "max_prompts_per_image", None)
    if isinstance(max_prompts, int) and max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def _summarize_activation(record: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if "labels_per_dim" in record:
        summary["labels_per_dim"] = record["labels_per_dim"]
    if "scores_per_dim" in record:
        summary["scores_per_dim"] = record["scores_per_dim"]
    if "activated_concepts" in record:
        summary["activated_concepts"] = record["activated_concepts"]
    if "concept_network" in record:
        summary["concept_network"] = record["concept_network"]
    return summary


def _maybe_draw_graph(record: Dict[str, Any],
                      image_path: Path,
                      iteration_key: str,
                      cfg: Any) -> Optional[str]:
    if not getattr(cfg, "export_concept_graph", False):
        return None
    network = record.get("concept_network")
    if not network:
        return None
    base_dir = getattr(cfg, "graph_output_dir", None)
    if base_dir is None:
        base_dir = getattr(cfg, "output_path", Path("./outputs")) / "concept_graphs"
    graph_dir = Path(base_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / f"{image_path.stem}_{iteration_key}.png"
    try:
        draw_graph(network, graph_path)
    except Exception as exc:
        if getattr(cfg, "strict_graph_export", False):
            raise
        print(f"[reasoning_runner] Failed to draw concept graph for {image_path}: {exc}")
        return None
    return str(graph_path)


def run_reasoning(vlm_wrapper,
                  activated_concepts,
                  images,
                  concept_embeddings,
                  cfg):
    """
    执行推理并返回结构化结果。

    Args:
        vlm_wrapper: 任何实现 preprocess/generate 接口的 VLM wrapper。
        activated_concepts: 每张图像的激活概念/概念图信息，可为 dict 或 list。
        images: 待推理图像路径列表；若为 None 则使用 activated_concepts 内的键。
        concept_embeddings: 训练阶段保存的 keys/values，或 iteration → payload 的 dict。
        cfg: 配置对象，需提供 prompt/graph 输出等可选字段。
    """
    activation_lookup = _normalize_activations(activated_concepts)
    image_paths = _prepare_image_list(images, activation_lookup)
    iteration_batches = _iter_concept_embedding_batches(concept_embeddings)
    if not iteration_batches:
        iteration_batches = [(None, None)]

    all_outputs: Dict[str, Dict[str, Any]] = {}
    for iteration_label, payload in iteration_batches:
        iteration_key = _format_iteration_key(iteration_label)
        if payload is not None:
            iteration_idx = _iteration_to_int(iteration_label)
            inference_utils.set_concept_embeddings(
                vlm_wrapper=vlm_wrapper,
                concept_embeddings=payload,
                iteration=iteration_idx,
                cfg=cfg,
            )
        print(f"[reasoning_runner] Running {iteration_key} on {len(image_paths)} images.")
        batch_outputs: Dict[str, Any] = {}
        for image_path in image_paths:
            abs_key = _normalize_key(image_path)
            record = activation_lookup.get(abs_key, {})
            prompts = _build_prompts(record, cfg)
            if not prompts:
                continue
            concept_signal = _extract_concept_signal(record)
            per_prompt_outputs: Dict[str, Dict[str, Any]] = {}
            for prompt in prompts:
                inputs = vlm_wrapper.preprocess(image_path=image_path, prompt=prompt)
                generated = vlm_wrapper.generate(inputs=inputs, concept_signals=concept_signal)
                text = generated[0] if isinstance(generated, list) else generated
                per_prompt_outputs[prompt] = {"text": text}
            graph_path = _maybe_draw_graph(record, Path(image_path), iteration_key, cfg)
            batch_outputs[str(image_path)] = {
                "activated_summary": _summarize_activation(record),
                "graph_path": graph_path,
                "prompts": per_prompt_outputs,
            }
        all_outputs[iteration_key] = batch_outputs
    return all_outputs
