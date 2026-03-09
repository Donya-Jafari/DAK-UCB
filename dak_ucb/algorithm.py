import numpy as np
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import minimize

from .metrics import compute_kid, ucb_radius_estimator, rbf_kernel
from .models import ModelBackend
from .embeddings import EmbeddingBackend


def run_dak_ucb(cfg: dict):
    exp = cfg["experiment"]
    modes = cfg["modes"]
    hp = cfg["hyperparams"]
    model_names = cfg["models"]["names"]
    runtime = cfg["runtime"]

    constant = hp["constant"]
    runs = hp["runs"]
    model_count = hp["model_count"]
    iterations = hp["iterations"]
    prompt_feature_count = hp["prompt_feature_count"]
    image_feature_count = hp["image_feature_count"]
    delta = hp["delta"]
    alpha = hp["alpha"]
    gamma = hp["gamma"]
    lambd = hp["lambd"]

    if modes["pak"]:
        lambd = 0

    eta = np.sqrt(2.0 * np.log(2.0 * model_count / delta))

    emb = EmbeddingBackend(prompt_feature_count, image_feature_count, runtime["use_mock_data"], runtime.get("seed"))
    model_backends = [ModelBackend(name, runtime["use_mock_data"]) for name in model_names]

    divs, clips, dists = [], [], []

    for _ in range(runs):
        all_clips = []
        all_joint_rke = []
        all_dists = []
        real_embeddings = []
        fake_embeddings = []
        running_div = 1
        running_clip = 0

        clip_regressors = [[] for _ in range(model_count)]
        div_regressors = [[[] for _ in range(model_count)] for _ in range(model_count)]

        observed_prompts_per_model = [np.empty((1, prompt_feature_count)) for _ in range(model_count)]
        observed_images_per_model = [np.empty((1, image_feature_count)) for _ in range(model_count)]
        observed_targets_per_model = [np.empty((1, image_feature_count)) for _ in range(model_count)]
        clip_scores_per_model = [np.empty(1,) for _ in range(model_count)]
        div_scores_per_model = [[np.empty(1,) for _ in range(model_count)] for _ in range(model_count)]

        pred_score_per_model = np.ones(model_count) * np.inf
        visits_per_model = np.zeros((model_count,), dtype=int)

        for iteration in range(iterations):
            prompt = ""  # placeholder prompt
            prompt_emb = emb.encode_prompt(prompt)
            prompt_normalized_emb = prompt_emb.reshape(1, -1)

            if iteration < model_count:
                selected_model = np.argmax(pred_score_per_model)
            else:
                estimated_clip_scores = np.empty((model_count,))
                estimated_div_scores = np.empty((model_count, model_count))
                estimated_div_scores_lin = np.empty((model_count,))

                for model in range(model_count):
                    estimated_clip_scores[model] = clip_regressors[model].predict(prompt_normalized_emb)[0]
                    estimated_div_scores_lin[model] = div_regressors[model][model].predict(prompt_normalized_emb)[0]
                    for second_model in range(model_count):
                        estimated_div_scores[model][second_model] = div_regressors[model][second_model].predict(prompt_normalized_emb)[0]

                estimated_bounds = ucb_radius_estimator(
                    prompt_normalized_emb,
                    observed_prompts_per_model,
                    model_count,
                    prompt_feature_count,
                    alpha,
                    gamma,
                )
                alpha_init = np.ones(model_count) / model_count
                d_sym = (estimated_div_scores + estimated_div_scores.T) / 2

                def objective(alpha_vec):
                    linear_term1 = estimated_clip_scores @ alpha_vec
                    linear_term2 = (eta * estimated_bounds) @ alpha_vec
                    linear_term3 = -lambd * estimated_div_scores_lin @ alpha_vec
                    quadratic_term = -lambd * alpha_vec.T @ d_sym @ alpha_vec
                    if modes["mix"]:
                        return -(quadratic_term + linear_term1 + linear_term2)
                    return -(linear_term1 + linear_term2 + linear_term3)

                constraints = [{"type": "eq", "fun": lambda a: np.sum(a) - 1}]
                bounds = [(0, 1) for _ in range(model_count)]
                result = minimize(objective, alpha_init, method="SLSQP", bounds=bounds, constraints=constraints)

                alpha_value = result.x
                if modes["random"]:
                    alpha_value = np.ones(model_count) / model_count
                alpha_probabilities = np.maximum(alpha_value, 0)
                alpha_probabilities /= np.sum(alpha_probabilities)
                selected_model = np.random.choice(len(alpha_probabilities), p=alpha_probabilities)

                if modes["onearm"]:
                    selected_model = modes["oracle_choice"]

            image = model_backends[selected_model].generate(prompt)
            image_emb = emb.encode_image(image)
            image_normalized_emb = image_emb.reshape(1, -1)
            image_clip = emb.encode_image_clip(image)

            target_normalized_emb = emb.encode_target(prompt).reshape(1, -1)
            fake_embeddings.append(image_normalized_emb)
            real_embeddings.append(target_normalized_emb)

            if iteration > 0:
                s = 0
                for m in range(model_count):
                    for sample in range(visits_per_model[m]):
                        c1 = rbf_kernel(
                            prompt_normalized_emb.reshape(1, prompt_feature_count),
                            observed_prompts_per_model[m][sample].reshape(1, prompt_feature_count),
                            gamma,
                        ).flat[0]
                        c2 = rbf_kernel(
                            image_normalized_emb.reshape(1, image_feature_count),
                            observed_images_per_model[m][sample].reshape(1, image_feature_count),
                            gamma,
                        ).flat[0]
                        s += (c1**2 * c2**2)
                running_div += (2 * s + 1)
            all_joint_rke.append(running_div / ((iteration + 1) ** 2))

            cosine_angle = (image_clip.squeeze(0)).dot(prompt_normalized_emb.squeeze(0))
            score = max(0.0, 100.0 * cosine_angle)
            running_clip += score
            all_clips.append(running_clip / (iteration + 1))
            all_dists.append(
                compute_kid(
                    np.array(real_embeddings).reshape((iteration + 1, -1)),
                    np.array(fake_embeddings).reshape((iteration + 1, -1)),
                    gamma,
                )
            )

            if visits_per_model[selected_model] == 0:
                observed_prompts_per_model[selected_model][0] = prompt_normalized_emb
                clip_scores_per_model[selected_model][0] = score / 100
                observed_images_per_model[selected_model][0] = image_normalized_emb

                if exp["objective"] == "I-JRKE":
                    for i in range(model_count):
                        div = 0
                        for sample in range(visits_per_model[i]):
                            c1 = rbf_kernel(
                                prompt_normalized_emb.reshape(1, prompt_feature_count),
                                observed_prompts_per_model[i][sample].reshape(1, prompt_feature_count),
                                gamma,
                            ).flat[0]
                            c2 = rbf_kernel(
                                image_normalized_emb.reshape(1, image_feature_count),
                                observed_images_per_model[i][sample].reshape(1, image_feature_count),
                                gamma,
                            ).flat[0]
                            div += (c1**2 * c2**2)
                        bound = np.sqrt(constant)
                        if visits_per_model[i] != 0:
                            div_scores_per_model[selected_model][i][0] = div / visits_per_model[i] - bound
                        else:
                            div_scores_per_model[selected_model][i][0] = div - bound
                else:
                    for i in range(model_count):
                        dist = 0
                        for sample in range(visits_per_model[i]):
                            k = (
                                rbf_kernel(
                                    observed_images_per_model[i][sample].reshape(1, image_feature_count),
                                    image_normalized_emb.reshape(1, image_feature_count),
                                    gamma,
                                ).flat[0]
                                + rbf_kernel(
                                    observed_targets_per_model[i][sample].reshape(1, image_feature_count),
                                    target_normalized_emb.reshape(1, image_feature_count),
                                    gamma,
                                ).flat[0]
                                - rbf_kernel(
                                    observed_images_per_model[i][sample].reshape(1, image_feature_count),
                                    target_normalized_emb.reshape(1, image_feature_count),
                                    gamma,
                                ).flat[0]
                                - rbf_kernel(
                                    observed_targets_per_model[i][sample].reshape(1, image_feature_count),
                                    image_normalized_emb.reshape(1, image_feature_count),
                                    gamma,
                                ).flat[0]
                            )
                            c1 = rbf_kernel(
                                prompt_normalized_emb.reshape(1, prompt_feature_count),
                                observed_prompts_per_model[i][sample].reshape(1, prompt_feature_count),
                                gamma,
                            ).flat[0]
                            dist += (c1 * k)
                        if visits_per_model[i] != 0:
                            dist = dist / visits_per_model[i]
                        bound = np.sqrt(constant)
                        div_scores_per_model[selected_model][i][0] = dist - bound

                    observed_targets_per_model[selected_model][0] = target_normalized_emb

            else:
                observed_prompts_per_model[selected_model] = np.concatenate(
                    (observed_prompts_per_model[selected_model], prompt_normalized_emb), axis=0
                )
                clip_scores_per_model[selected_model] = np.concatenate(
                    (clip_scores_per_model[selected_model], np.array([score / 100])), axis=0
                )
                observed_images_per_model[selected_model] = np.concatenate(
                    (observed_images_per_model[selected_model], image_normalized_emb), axis=0
                )

                if exp["objective"] == "I-JRKE":
                    for i in range(model_count):
                        div = 0
                        for sample in range(visits_per_model[i]):
                            c1 = rbf_kernel(
                                prompt_normalized_emb.reshape(1, prompt_feature_count),
                                observed_prompts_per_model[i][sample].reshape(1, prompt_feature_count),
                                gamma,
                            ).flat[0]
                            c2 = rbf_kernel(
                                image_normalized_emb.reshape(1, image_feature_count),
                                observed_images_per_model[i][sample].reshape(1, image_feature_count),
                                gamma,
                            ).flat[0]
                            div += (c1**2 * c2**2)
                        bound = np.sqrt(constant / (visits_per_model[i]))
                        if visits_per_model[i] != 0:
                            div_scores_per_model[selected_model][i] = np.concatenate(
                                (div_scores_per_model[selected_model][i], np.array([(div / visits_per_model[i]) - bound])),
                                axis=0,
                            )
                        else:
                            div_scores_per_model[selected_model][i] = np.concatenate(
                                (div_scores_per_model[selected_model][i], np.array([div - bound])), axis=0
                            )
                else:
                    for i in range(model_count):
                        dist = 0
                        for sample in range(visits_per_model[i]):
                            k = (
                                rbf_kernel(
                                    observed_images_per_model[i][sample].reshape(1, image_feature_count),
                                    image_normalized_emb.reshape(1, image_feature_count),
                                    gamma,
                                ).flat[0]
                                + rbf_kernel(
                                    observed_targets_per_model[i][sample].reshape(1, image_feature_count),
                                    target_normalized_emb.reshape(1, image_feature_count),
                                    gamma,
                                ).flat[0]
                                - rbf_kernel(
                                    observed_images_per_model[i][sample].reshape(1, image_feature_count),
                                    target_normalized_emb.reshape(1, image_feature_count),
                                    gamma,
                                ).flat[0]
                                - rbf_kernel(
                                    observed_targets_per_model[i][sample].reshape(1, image_feature_count),
                                    image_normalized_emb.reshape(1, image_feature_count),
                                    gamma,
                                ).flat[0]
                            )
                            c1 = rbf_kernel(
                                prompt_normalized_emb.reshape(1, prompt_feature_count),
                                observed_prompts_per_model[i][sample].reshape(1, prompt_feature_count),
                                gamma,
                            ).flat[0]
                            dist += (c1 * k)
                        dist = dist / (visits_per_model[i])
                        bound = np.sqrt(constant / (visits_per_model[i]))
                        div_scores_per_model[selected_model][i] = np.concatenate(
                            (div_scores_per_model[selected_model][i], np.array([dist - bound])), axis=0
                        )

                    observed_targets_per_model[selected_model] = np.concatenate(
                        (observed_targets_per_model[selected_model], target_normalized_emb), axis=0
                    )

            pred_score_per_model[selected_model] = 0
            visits_per_model[selected_model] += 1

            y = clip_scores_per_model[selected_model].reshape(-1)
            x = observed_prompts_per_model[selected_model].reshape(-1, prompt_feature_count)

            clip_regressors[selected_model] = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(x, y)
            for i in range(model_count):
                div_regressors[selected_model][i] = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(
                    x, div_scores_per_model[selected_model][i].reshape(-1)
                )

        sorted_models = sorted(zip(model_names, visits_per_model), key=lambda x: x[1], reverse=True)
        print("\nModels sorted by selection frequency:")
        for model, visits in sorted_models:
            print(f"{model}: {int(visits)} selections")

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
