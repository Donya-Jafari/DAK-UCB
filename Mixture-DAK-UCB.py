from pathlib import Path
import argparse
import json

import numpy as np
import yaml
from PIL import Image
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(folder):
    folder = Path(folder).expanduser()
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])


def list_texts(path):
    path = Path(path).expanduser()
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_items(kind, path):
    if kind == "image":
        items = list_images(path)
    elif kind == "text":
        items = list_texts(path)
    else:
        raise ValueError("kind must be 'image' or 'text'")
    if not items:
        raise ValueError("dataset is empty")
    return items


def materialize(item, kind):
    if kind == "image":
        return Image.open(item).convert("RGB")
    return item


def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-12)


class EmbeddingBackend:
    def __init__(self, input_dim, output_dim, use_mock, seed=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_mock = use_mock
        self.rng = np.random.default_rng(seed)

    def encode_input(self, item, kind):
        if self.use_mock:
            dim = self.input_dim
            return normalize(self.rng.standard_normal(dim))
        raise NotImplementedError("Real embedding models are not implemented in this simplified script.")

    def encode_output(self, item, kind):
        if self.use_mock:
            dim = self.output_dim
            return normalize(self.rng.standard_normal(dim))
        raise NotImplementedError("Real embedding models are not implemented in this simplified script.")


class ModelBackend:
    def __init__(self, name, use_mock):
        self.name = name
        self.use_mock = use_mock

    def run(self, item, output_kind, iteration):
        if self.use_mock:
            if output_kind == "text":
                return "mock output {}".format(iteration)
            return None
        raise NotImplementedError("Real model loading is not implemented in this simplified script.")


def ucb_radius(current_x, observed_x, model_count, feature_dim, alpha, gamma):
    bonus = np.zeros(model_count)
    for m in range(model_count):
        target = rbf_kernel(observed_x[m], current_x, gamma=gamma).reshape(-1)
        reg = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
        reg.fit(observed_x[m].reshape(-1, feature_dim), target)
        bonus[m] = reg.predict(current_x)[0]
    self_kernel = rbf_kernel(current_x, current_x, gamma=gamma)[0, 0]
    return np.sqrt(np.maximum(0.0, self_kernel - bonus))


def simple_mmd(real_embs, fake_embs, gamma):
    k_rr = rbf_kernel(real_embs, real_embs, gamma=gamma)
    k_ff = rbf_kernel(fake_embs, fake_embs, gamma=gamma)
    k_rf = rbf_kernel(real_embs, fake_embs, gamma=gamma)
    return k_rr.mean() + k_ff.mean() - 2 * k_rf.mean()


def save_generated(item, kind, save_dir, iteration, model_name):
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    if kind == "text":
        out_file = save_dir / "generated.txt"
        with out_file.open("a", encoding="utf-8") as f:
            f.write("{:05d}\t{}\t{}\n".format(iteration, model_name, item))
    elif item is not None:
        out_path = save_dir / "{:05d}_{}.png".format(iteration, model_name)
        item.save(out_path)


def task_setup(task):
    if task == "image_captioning":
        return {
            "input_kind": "image",
            "output_kind": "text",
            "input_dim": 768,
            "output_dim": 512,
        }
    if task == "image_generation":
        return {
            "input_kind": "text",
            "output_kind": "image",
            "input_dim": 512,
            "output_dim": 768,
        }
    if task == "llm_prompting":
        return {
            "input_kind": "text",
            "output_kind": "text",
            "input_dim": 512,
            "output_dim": 512,
        }
    raise ValueError("Unknown task: {}".format(task))


def build_models(cfg, task, use_mock):
    if task == "image_captioning":
        specs = cfg["models"]["captioning"]
    elif task == "image_generation":
        specs = cfg["models"]["generation"]
    else:
        specs = cfg["models"].get("llm", [])
    return [ModelBackend(spec["name"], use_mock) for spec in specs]


def run_experiment(cfg):
    task = cfg["experiment"]["task"]
    setup = task_setup(task)
    input_kind = setup["input_kind"]
    output_kind = setup["output_kind"]
    input_dim = setup["input_dim"]
    output_dim = setup["output_dim"]

    data_cfg = cfg["data"]
    input_items = load_items(input_kind, data_cfg["input"])
    output_items = load_items(output_kind, data_cfg["output"])
    if len(input_items) != len(output_items):
        raise ValueError("input and output datasets must have the same length")

    hp = cfg["hyperparameters"]
    modes = cfg["routing"]
    runs = int(hp["runs"])
    gamma = float(hp["gamma"])
    alpha = float(hp["alpha"])
    lambd = 0.0 if modes.get("pak", False) else float(hp["lambd"])
    delta = float(hp["delta"])
    constant = float(hp["constant"])
    use_mock = bool(cfg["runtime"]["use_mock"])
    objective_name = cfg["experiment"].get("objective", "I-JRKE")

    models = build_models(cfg, task, use_mock)
    model_count = len(models)
    if model_count == 0:
        raise ValueError("No models configured for task {}".format(task))
    if 2.0 * model_count / delta <= 1.0:
        raise ValueError("delta must satisfy 2 * model_count / delta > 1")

    eta = np.sqrt(2.0 * np.log(2.0 * model_count / delta))
    embedder = EmbeddingBackend(input_dim, output_dim, use_mock, cfg["runtime"].get("seed"))

    div_runs = []
    dist_runs = []
    clip_runs = []

    for _ in range(runs):
        observed_inputs = [np.zeros((1, input_dim)) for _ in range(model_count)]
        observed_outputs = [np.zeros((1, output_dim)) for _ in range(model_count)]
        observed_targets = [np.zeros((1, output_dim)) for _ in range(model_count)]
        clip_scores = [np.zeros(1) for _ in range(model_count)]
        div_scores = [[np.zeros(1) for _ in range(model_count)] for _ in range(model_count)]
        clip_models = [[] for _ in range(model_count)]
        div_models = [[[] for _ in range(model_count)] for _ in range(model_count)]
        pred_score = np.ones(model_count) * np.inf
        visits = np.zeros(model_count, dtype=int)
        fake_embs = []
        real_embs = []
        running_div = 1.0
        running_clip = 0.0

        for i, raw_input in enumerate(input_items):
            raw_output = output_items[i]
            input_item = materialize(raw_input, input_kind)
            output_item = materialize(raw_output, output_kind)
            input_emb = embedder.encode_input(input_item, input_kind).reshape(1, -1)
            target_emb = embedder.encode_output(output_item, output_kind).reshape(1, -1)

            if i < model_count:
                chosen = int(np.argmax(pred_score))
            else:
                est_clip = np.array([clip_models[m].predict(input_emb)[0] for m in range(model_count)])
                est_div = np.array([[div_models[m][j].predict(input_emb)[0] for j in range(model_count)] for m in range(model_count)])
                bounds = ucb_radius(input_emb, observed_inputs, model_count, input_dim, alpha, gamma)
                est_div_lin = np.diag(est_div)
                d_sym = (est_div + est_div.T) / 2.0

                def objective(weights):
                    lin = est_clip @ weights
                    bonus = (eta * bounds) @ weights
                    lin_penalty = lambd * (est_div_lin @ weights)
                    quad_penalty = lambd * (weights.T @ d_sym @ weights)
                    if modes.get("mix", False):
                        return -(lin + bonus - quad_penalty)
                    return -(lin + bonus - lin_penalty)

                result = minimize(
                    objective,
                    np.ones(model_count) / model_count,
                    method="SLSQP",
                    bounds=[(0, 1)] * model_count,
                    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                )
                weights = np.ones(model_count) / model_count if modes.get("random", False) else result.x
                probs = np.maximum(weights, 0)
                probs = probs / probs.sum()
                chosen = np.random.choice(model_count, p=probs)
                if modes.get("onearm", False):
                    chosen = int(modes.get("oracle_choice", 0))
                if modes.get("random", False):
                    probs = np.ones(model_count) / model_count
                    chosen = np.random.choice(model_count, p=probs)

            generated = models[chosen].run(input_item, output_kind, i)
            generated_emb = embedder.encode_output(generated, output_kind).reshape(1, -1)
            save_generated(generated, output_kind, data_cfg["generated"], i, models[chosen].name)

            fake_embs.append(generated_emb)
            real_embs.append(target_emb)
            clip_score = 0.0 ## Placeholder for actual CLIP score computation between input_item and generated output
            running_clip += clip_score

            if i > 0:
                score = 0.0
                for m in range(model_count):
                    for sample in range(visits[m]):
                        kx = rbf_kernel(input_emb, observed_inputs[m][sample].reshape(1, -1), gamma=gamma)[0, 0]
                        ky = rbf_kernel(generated_emb, observed_outputs[m][sample].reshape(1, -1), gamma=gamma)[0, 0]
                        score += (kx ** 2) * (ky ** 2)
                running_div += 2 * score + 1

            if visits[chosen] == 0:
                observed_inputs[chosen][0] = input_emb
                observed_outputs[chosen][0] = generated_emb
                clip_scores[chosen][0] = clip_score
                if objective_name == "I-JRKE":
                    for m in range(model_count):
                        div = 0.0
                        for sample in range(visits[m]):
                            kx = rbf_kernel(input_emb, observed_inputs[m][sample].reshape(1, -1), gamma=gamma)[0, 0]
                            ky = rbf_kernel(generated_emb, observed_outputs[m][sample].reshape(1, -1), gamma=gamma)[0, 0]
                            div += (kx ** 2) * (ky ** 2)
                        div_scores[chosen][m][0] = div - np.sqrt(constant)
                else:
                    for m in range(model_count):
                        dist = 0.0
                        for sample in range(visits[m]):
                            out_prev = observed_outputs[m][sample].reshape(1, -1)
                            tgt_prev = observed_targets[m][sample].reshape(1, -1)
                            cross = (
                                rbf_kernel(out_prev, generated_emb, gamma=gamma)[0, 0]
                                + rbf_kernel(tgt_prev, target_emb, gamma=gamma)[0, 0]
                                - rbf_kernel(out_prev, target_emb, gamma=gamma)[0, 0]
                                - rbf_kernel(tgt_prev, generated_emb, gamma=gamma)[0, 0]
                            )
                            kx = rbf_kernel(input_emb, observed_inputs[m][sample].reshape(1, -1), gamma=gamma)[0, 0]
                            dist += kx * cross
                        div_scores[chosen][m][0] = dist - np.sqrt(constant)
                    observed_targets[chosen][0] = target_emb
            else:
                observed_inputs[chosen] = np.concatenate((observed_inputs[chosen], input_emb), axis=0)
                observed_outputs[chosen] = np.concatenate((observed_outputs[chosen], generated_emb), axis=0)
                clip_scores[chosen] = np.concatenate((clip_scores[chosen], np.array([clip_score])), axis=0)
                if objective_name == "I-JRKE":
                    for m in range(model_count):
                        div = 0.0
                        for sample in range(visits[m]):
                            kx = rbf_kernel(input_emb, observed_inputs[m][sample].reshape(1, -1), gamma=gamma)[0, 0]
                            ky = rbf_kernel(generated_emb, observed_outputs[m][sample].reshape(1, -1), gamma=gamma)[0, 0]
                            div += (kx ** 2) * (ky ** 2)
                        bound = np.sqrt(constant / visits[m]) if visits[m] else np.sqrt(constant)
                        value = (div / visits[m] - bound) if visits[m] else (div - bound)
                        div_scores[chosen][m] = np.concatenate((div_scores[chosen][m], np.array([value])), axis=0)
                else:
                    for m in range(model_count):
                        dist = 0.0
                        for sample in range(visits[m]):
                            out_prev = observed_outputs[m][sample].reshape(1, -1)
                            tgt_prev = observed_targets[m][sample].reshape(1, -1)
                            cross = (
                                rbf_kernel(out_prev, generated_emb, gamma=gamma)[0, 0]
                                + rbf_kernel(tgt_prev, target_emb, gamma=gamma)[0, 0]
                                - rbf_kernel(out_prev, target_emb, gamma=gamma)[0, 0]
                                - rbf_kernel(tgt_prev, generated_emb, gamma=gamma)[0, 0]
                            )
                            kx = rbf_kernel(input_emb, observed_inputs[m][sample].reshape(1, -1), gamma=gamma)[0, 0]
                            dist += kx * cross
                        value = (dist / visits[m]) - np.sqrt(constant / visits[m])
                        div_scores[chosen][m] = np.concatenate((div_scores[chosen][m], np.array([value])), axis=0)
                    observed_targets[chosen] = np.concatenate((observed_targets[chosen], target_emb), axis=0)

            pred_score[chosen] = 0.0
            visits[chosen] += 1
            x = observed_inputs[chosen].reshape(-1, input_dim)
            y = clip_scores[chosen].reshape(-1)
            clip_models[chosen] = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(x, y)
            for m in range(model_count):
                div_models[chosen][m] = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(x, div_scores[chosen][m].reshape(-1))

        div_runs.append(running_div / (len(input_items) ** 2))
        dist_runs.append(
            simple_mmd(
                np.array(real_embs).reshape(len(input_items), -1),
                np.array(fake_embs).reshape(len(input_items), -1),
                gamma,
            )
        )
        clip_runs.append(running_clip / len(input_items))

    result = {
        "experiment": cfg["experiment"]["name"],
        "task": task,
        "avg_diversity": float(np.mean(div_runs)),
        "avg_distance": float(np.mean(dist_runs)),
        "avg_clip": float(np.mean(clip_runs)),
    }

    result_path = Path(cfg["runtime"]["result_file"]).expanduser()
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result_path = run_experiment(load_config(args.config))
    print("Wrote results to {}".format(result_path))


if __name__ == "__main__":
    main()
