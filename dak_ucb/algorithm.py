import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import minimize

from .metrics import compute_kid, ucb_radius_estimator, rbf_kernel
from .models import ImageGeneratorBackend, CaptioningBackend
from .embeddings import EmbeddingBackend


def _load_inputs(task: str, input_cfg: dict):
    if task == "image_captioning":
        img_dir = Path(input_cfg["image_dir"]).expanduser()
        paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
        if not paths:
            raise ValueError(f"No images found in {img_dir}")
        return paths
    txt = Path(input_cfg["text_file"]).expanduser()
    lines = [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"No text lines found in {txt}")
    return lines


def run_dak_ucb(cfg: dict):
    exp, modes, hp, model_cfg, runtime = (
        cfg["experiment"],
        cfg["modes"],
        cfg["hyperparams"],
        cfg["models"],
        cfg["runtime"],
    )
    task = exp["task"]
    task_counts = exp.get("task_model_count", {})
    if task not in task_counts:
        raise ValueError(f"Missing task_model_count for task: {task}")

    image_models = model_cfg.get("image_generation", [])
    caption_models = model_cfg.get("captioning", [])
    llm_models = model_cfg.get("llm", [])
    active_models = (
        image_models if task == "image_generation" else caption_models if task == "image_captioning" else llm_models
    )

    model_count = int(task_counts[task])
    if model_count != len(active_models):
        raise ValueError(f"task_model_count[{task}]={model_count} but got {len(active_models)} models")
    if model_count == 0:
        raise ValueError(f"No models configured for task: {task}")

    constant = float(hp["constant"])
    delta = float(hp["delta"])
    alpha = float(hp["alpha"])
    gamma = float(hp["gamma"])
    lambd = float(hp["lambd"]) if not modes["pak"] else 0.0
    # Dimensions are task-specific and override config values.
    if task == "image_generation":
        input_dim = 512
        output_dim = 768
        input_type = "text"
        output_type = "image"
        prompt_dim = 512
        image_dim = 768
    elif task == "image_captioning":
        input_dim = 768
        output_dim = 512
        input_type = "image"
        output_type = "text"
        prompt_dim = 512
        image_dim = 768
    elif task == "llm_prompting":
        input_dim = 512
        output_dim = 512
        input_type = "text"
        output_type = "text"
        prompt_dim = 512
        image_dim = 512
    else:
        raise ValueError(f"Unknown task: {task}")

    if 2.0 * model_count / delta <= 1.0:
        raise ValueError("Invalid delta: must satisfy 2*model_count/delta > 1")
    eta = np.sqrt(2.0 * np.log(2.0 * model_count / delta))

    inputs = _load_inputs(task, exp["input"])
    output_cfg = exp["output"]
    output_image_dir = Path(output_cfg["image_dir"]).expanduser()
    output_text_file = Path(output_cfg["text_file"]).expanduser()
    iterations = len(inputs)
    runs = int(hp["runs"])

    emb = EmbeddingBackend(
        prompt_dim,
        image_dim,
        runtime["use_mock_data"],
        model_cfg["embeddings"]["clip_model_id"],
        model_cfg["embeddings"]["dinov2_model_id"],
        runtime["device"],
        runtime.get("seed"),
    )

    if task == "image_generation":
        model_backends = [
            ImageGeneratorBackend(m["name"], m["type"], m["model_id"], m.get("prior_id"), runtime["use_mock_data"], runtime["device"])
            for m in active_models
        ]
    elif task == "image_captioning":
        model_backends = [
            CaptioningBackend(m["name"], m["type"], m["model_id"], runtime["use_mock_data"], runtime["device"])
            for m in active_models
        ]
    else:
        model_backends = []

    def encode_input(item):
        return emb.encode_image(item) if input_type == "image" else emb.encode_prompt(item)

    def encode_output(item):
        return emb.encode_image(item) if output_type == "image" else emb.encode_prompt(item)

    def clip_image_for_score(input_item, output_item):
        if task == "image_generation":
            return output_item
        if task == "image_captioning":
            return input_item
        return None

    def save_output_image(image_obj, iteration_idx: int, model_name: str):
        if image_obj is None:
            return
        output_image_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_image_dir / f"{iteration_idx:05d}_{model_name}.png"
        image_obj.save(out_path)

    def append_output_text(text: str, iteration_idx: int, model_name: str, input_id: str):
        if not text:
            return
        output_text_file.parent.mkdir(parents=True, exist_ok=True)
        with output_text_file.open("a", encoding="utf-8") as f:
            f.write(f"{iteration_idx:05d}\t{model_name}\t{input_id}\t{text}\n")

    divs, clips, dists = [], [], []

    for _ in range(runs):
        all_clips, all_joint_rke, all_dists = [], [], []
        real_embeddings, fake_embeddings = [], []
        running_div, running_clip = 1, 0

        clip_regressors = [[] for _ in range(model_count)]
        div_regressors = [[[] for _ in range(model_count)] for _ in range(model_count)]
        observed_inputs = [np.zeros((1, input_dim)) for _ in range(model_count)]
        observed_outputs = [np.zeros((1, output_dim)) for _ in range(model_count)]
        observed_targets = [np.zeros((1, output_dim)) for _ in range(model_count)]
        clip_scores = [np.zeros((1,)) for _ in range(model_count)]
        div_scores = [[np.zeros((1,)) for _ in range(model_count)] for _ in range(model_count)]

        pred_score = np.ones(model_count) * np.inf
        visits = np.zeros((model_count,), dtype=int)

        for iteration in range(iterations):
            input_item = inputs[iteration]
            if input_type == "image":
                input_path = Path(input_item)
                input_item = Image.open(input_item).convert("RGB")
                input_id = input_path.name
            else:
                input_id = str(iteration)
            input_emb = encode_input(input_item).reshape(1, -1)

            if iteration < model_count:
                selected_model = np.argmax(pred_score)
            else:
                est_clip = np.array([clip_regressors[m].predict(input_emb)[0] for m in range(model_count)])
                est_div_lin = np.array([div_regressors[m][m].predict(input_emb)[0] for m in range(model_count)])
                est_div = np.array([[div_regressors[m][j].predict(input_emb)[0] for j in range(model_count)] for m in range(model_count)])
                est_bounds = ucb_radius_estimator(input_emb, observed_inputs, model_count, input_dim, alpha, gamma)
                d_sym = (est_div + est_div.T) / 2

                def objective(a):
                    lin1 = est_clip @ a
                    lin2 = (eta * est_bounds) @ a
                    lin3 = -lambd * est_div_lin @ a
                    quad = -lambd * a.T @ d_sym @ a
                    return -(quad + lin1 + lin2) if modes["mix"] else -(lin1 + lin2 + lin3)

                res = minimize(objective, np.ones(model_count) / model_count, method="SLSQP", bounds=[(0, 1)] * model_count, constraints=[{"type": "eq", "fun": lambda a: np.sum(a) - 1}])
                alpha_value = np.ones(model_count) / model_count if modes["random"] else res.x
                probs = np.maximum(alpha_value, 0)
                probs = probs / np.sum(probs)
                selected_model = np.random.choice(len(probs), p=probs)
                if modes["onearm"]:
                    selected_model = modes["oracle_choice"]

            if task == "image_generation":
                output_item = model_backends[selected_model].generate(input_item)
            elif task == "image_captioning":
                output_item = model_backends[selected_model].caption(input_item)
            else:
                output_item = ""

            model_name = active_models[selected_model]["name"]
            if task == "image_generation":
                save_output_image(output_item, iteration, model_name)
            else:
                append_output_text(output_item, iteration, model_name, input_id)

            output_emb = encode_output(output_item).reshape(1, -1)
            target_emb = output_emb

            fake_embeddings.append(output_emb)
            real_embeddings.append(target_emb)

            if iteration > 0:
                s = 0
                for m in range(model_count):
                    for sample in range(visits[m]):
                        c1 = rbf_kernel(input_emb.reshape(1, input_dim), observed_inputs[m][sample].reshape(1, input_dim), gamma).flat[0]
                        c2 = rbf_kernel(output_emb.reshape(1, output_dim), observed_outputs[m][sample].reshape(1, output_dim), gamma).flat[0]
                        s += (c1**2 * c2**2)
                running_div += (2 * s + 1)
            all_joint_rke.append(running_div / ((iteration + 1) ** 2))

            score = 0.0
            clip_image = clip_image_for_score(input_item, output_item)
            if clip_image is not None:
                image_clip = emb.encode_image_clip(clip_image)
                score_text_emb = output_emb if output_type == "text" else input_emb
                cosine_angle = np.ravel(image_clip).dot(np.ravel(score_text_emb))
                score = max(0.0, 100.0 * cosine_angle)
            running_clip += score
            all_clips.append(running_clip / (iteration + 1))
            all_dists.append(compute_kid(np.array(real_embeddings).reshape((iteration + 1, -1)), np.array(fake_embeddings).reshape((iteration + 1, -1)), gamma))

            if visits[selected_model] == 0:
                observed_inputs[selected_model][0] = input_emb
                clip_scores[selected_model][0] = score / 100
                observed_outputs[selected_model][0] = output_emb

                if exp["objective"] == "I-JRKE":
                    for i in range(model_count):
                        div = 0
                        for sample in range(visits[i]):
                            c1 = rbf_kernel(input_emb.reshape(1, input_dim), observed_inputs[i][sample].reshape(1, input_dim), gamma).flat[0]
                            c2 = rbf_kernel(output_emb.reshape(1, output_dim), observed_outputs[i][sample].reshape(1, output_dim), gamma).flat[0]
                            div += (c1**2 * c2**2)
                        bound = np.sqrt(constant)
                        div_scores[selected_model][i][0] = (div / visits[i] - bound) if visits[i] != 0 else (div - bound)
                else:
                    for i in range(model_count):
                        dist = 0
                        for sample in range(visits[i]):
                            k = (
                                rbf_kernel(observed_outputs[i][sample].reshape(1, output_dim), output_emb.reshape(1, output_dim), gamma).flat[0]
                                + rbf_kernel(observed_targets[i][sample].reshape(1, output_dim), target_emb.reshape(1, output_dim), gamma).flat[0]
                                - rbf_kernel(observed_outputs[i][sample].reshape(1, output_dim), target_emb.reshape(1, output_dim), gamma).flat[0]
                                - rbf_kernel(observed_targets[i][sample].reshape(1, output_dim), output_emb.reshape(1, output_dim), gamma).flat[0]
                            )
                            c1 = rbf_kernel(input_emb.reshape(1, input_dim), observed_inputs[i][sample].reshape(1, input_dim), gamma).flat[0]
                            dist += (c1 * k)
                        dist = dist / visits[i] if visits[i] != 0 else dist
                        div_scores[selected_model][i][0] = dist - np.sqrt(constant)
                    observed_targets[selected_model][0] = target_emb
            else:
                observed_inputs[selected_model] = np.concatenate((observed_inputs[selected_model], input_emb), axis=0)
                clip_scores[selected_model] = np.concatenate((clip_scores[selected_model], np.array([score / 100])), axis=0)
                observed_outputs[selected_model] = np.concatenate((observed_outputs[selected_model], output_emb), axis=0)

                if exp["objective"] == "I-JRKE":
                    for i in range(model_count):
                        div = 0
                        for sample in range(visits[i]):
                            c1 = rbf_kernel(input_emb.reshape(1, input_dim), observed_inputs[i][sample].reshape(1, input_dim), gamma).flat[0]
                            c2 = rbf_kernel(output_emb.reshape(1, output_dim), observed_outputs[i][sample].reshape(1, output_dim), gamma).flat[0]
                            div += (c1**2 * c2**2)
                        bound = np.sqrt(constant / (visits[i]))
                        val = (div / visits[i] - bound) if visits[i] != 0 else (div - bound)
                        div_scores[selected_model][i] = np.concatenate((div_scores[selected_model][i], np.array([val])), axis=0)
                else:
                    for i in range(model_count):
                        dist = 0
                        for sample in range(visits[i]):
                            k = (
                                rbf_kernel(observed_outputs[i][sample].reshape(1, output_dim), output_emb.reshape(1, output_dim), gamma).flat[0]
                                + rbf_kernel(observed_targets[i][sample].reshape(1, output_dim), target_emb.reshape(1, output_dim), gamma).flat[0]
                                - rbf_kernel(observed_outputs[i][sample].reshape(1, output_dim), target_emb.reshape(1, output_dim), gamma).flat[0]
                                - rbf_kernel(observed_targets[i][sample].reshape(1, output_dim), output_emb.reshape(1, output_dim), gamma).flat[0]
                            )
                            c1 = rbf_kernel(input_emb.reshape(1, input_dim), observed_inputs[i][sample].reshape(1, input_dim), gamma).flat[0]
                            dist += (c1 * k)
                        dist = dist / visits[i]
                        div_scores[selected_model][i] = np.concatenate((div_scores[selected_model][i], np.array([dist - np.sqrt(constant / visits[i])])), axis=0)
                    observed_targets[selected_model] = np.concatenate((observed_targets[selected_model], target_emb), axis=0)

            pred_score[selected_model] = 0
            visits[selected_model] += 1

            y = clip_scores[selected_model].reshape(-1)
            x = observed_inputs[selected_model].reshape(-1, input_dim)
            clip_regressors[selected_model] = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(x, y)
            for i in range(model_count):
                div_regressors[selected_model][i] = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(x, div_scores[selected_model][i].reshape(-1))

        sorted_models = sorted(zip([m["name"] for m in active_models], visits), key=lambda x: x[1], reverse=True)
        print("\nModels sorted by selection frequency:")
        for model, v in sorted_models:
            print(f"{model}: {int(v)} selections")

        divs.append(all_joint_rke)
        clips.append(all_clips)
        dists.append(all_dists)

    avg_div = np.mean(np.array(divs), axis=0)
    avg_clip = np.mean(np.array(clips), axis=0)
    avg_dist = np.mean(np.array(dists), axis=0)

    out_name = f"{exp['name']}_{exp['algorithm']}.txt"
    with open(out_name, "w", encoding="utf-8") as f:
        if exp["objective"] == "I-JRKE":
            f.write("Average 1/div :\n")
            f.write(", ".join([str(x) for x in avg_div]) + "\n")
        else:
            f.write("Average dist :\n")
            f.write(", ".join([str(x) for x in avg_dist]) + "\n")
        f.write("Average clip :\n")
        f.write(", ".join([str(x) for x in avg_clip]) + "\n")

    return out_name
