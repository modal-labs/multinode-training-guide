import pathlib
import shlex
from dataclasses import dataclass


RECIPES_DIR = pathlib.Path(__file__).parent.resolve()


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    model_id: str
    args_file: str
    recommended_nodes: int
    gpu: str


RECIPES = {
    "qwen25-0p5b-lora": Recipe(
        name="qwen25-0p5b-lora",
        description="Single-node smoke test adapted from the upstream Miles LoRA example.",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        args_file="tests/qwen25-0p5b-lora.args",
        recommended_nodes=1,
        gpu="H100:8",
    ),
    "qwen3-30b-a3b-lora": Recipe(
        name="qwen3-30b-a3b-lora",
        description="Single-node Qwen3-30B-A3B all-layer bridge-mode LoRA recipe aligned with current best practices.",
        model_id="Qwen/Qwen3-30B-A3B",
        args_file="qwen3-30b-a3b-lora.args",
        recommended_nodes=1,
        gpu="H100:8",
    ),
    "qwen3-30b-a3b-lora-fewstep": Recipe(
        name="qwen3-30b-a3b-lora-fewstep",
        description="Single-node Qwen3-30B-A3B all-layer LoRA recipe trimmed to chase a few full RL steps.",
        model_id="Qwen/Qwen3-30B-A3B",
        args_file="tests/qwen3-30b-a3b-lora-fewstep.args",
        recommended_nodes=1,
        gpu="H100:8",
    ),
    "qwen3-30b-a3b-lora-greedy-debug": Recipe(
        name="qwen3-30b-a3b-lora-greedy-debug",
        description="Single-node Qwen3-30B-A3B attention-only debug/control recipe with greedy rollout.",
        model_id="Qwen/Qwen3-30B-A3B",
        args_file="tests/qwen3-30b-a3b-lora-greedy-debug.args",
        recommended_nodes=1,
        gpu="H100:8",
    ),
    "qwen3-30b-a3b-experts-lora": Recipe(
        name="qwen3-30b-a3b-experts-lora",
        description="Explicit all-layer Qwen3-30B-A3B LoRA recipe including expert linear_fc1/fc2 targets.",
        model_id="Qwen/Qwen3-30B-A3B",
        args_file="qwen3-30b-a3b-experts-lora.args",
        recommended_nodes=1,
        gpu="H100:8",
    ),
    "qwen3-30b-a3b-experts-fewstep": Recipe(
        name="qwen3-30b-a3b-experts-fewstep",
        description="Explicit all-layer Qwen3-30B-A3B few-step recipe including expert linear_fc1/fc2 targets.",
        model_id="Qwen/Qwen3-30B-A3B",
        args_file="tests/qwen3-30b-a3b-experts-fewstep.args",
        recommended_nodes=1,
        gpu="H100:8",
    ),
}


def get_recipe(name: str) -> Recipe:
    if name not in RECIPES:
        available = ", ".join(sorted(RECIPES))
        raise ValueError(f"Unknown recipe: {name}. Available recipes: {available}")
    return RECIPES[name]


def get_optional_recipe(name: str) -> Recipe | None:
    if not name:
        return None
    return get_recipe(name)


def iter_recipes() -> list[Recipe]:
    return sorted(RECIPES.values(), key=lambda item: item.name)


def clean_arg_text(arg_text: str) -> str:
    lines: list[str] = []
    for raw_line in arg_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_arg_text(arg_text: str) -> list[str]:
    cleaned = clean_arg_text(arg_text)
    return shlex.split(cleaned) if cleaned else []


def read_arg_file(path_str: str) -> str:
    if not path_str:
        return ""
    return pathlib.Path(path_str).read_text()


def merge_arg_texts(*parts: str) -> str:
    return "\n".join(part for part in parts if part and part.strip())


def load_recipe_text(recipe: Recipe, base_dir: pathlib.Path | None = None) -> str:
    recipe_dir = base_dir if base_dir is not None else RECIPES_DIR
    return (recipe_dir / recipe.args_file).read_text()


def format_recipe_table() -> list[str]:
    return [
        f"  - {recipe.name}: {recipe.description} "
        f"(model={recipe.model_id}, nodes={recipe.recommended_nodes}, gpu={recipe.gpu})"
        for recipe in iter_recipes()
    ]
