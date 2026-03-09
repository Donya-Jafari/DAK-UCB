import numpy as np
import sklearn.metrics
from sklearn.kernel_ridge import KernelRidge


def rbf_kernel(x, y, gamma):
    return sklearn.metrics.pairwise.rbf_kernel(x, y, gamma=gamma)


def compute_kid(real_embeddings, fake_embeddings, gamma):
    k_real = rbf_kernel(real_embeddings, real_embeddings, gamma)
    k_fake = rbf_kernel(fake_embeddings, fake_embeddings, gamma)
    k_cross = rbf_kernel(real_embeddings, fake_embeddings, gamma)
    return k_real.mean() + k_fake.mean() - 2 * k_cross.mean()


def ucb_radius_estimator(prompt_emb, observed_prompts, model_count, prompt_feature_count, alpha, gamma):
    bonus = np.zeros((model_count,))
    for model in range(model_count):
        reg_target_g = rbf_kernel(observed_prompts[model], prompt_emb, gamma=gamma).flatten()
        reg = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
        reg.fit(observed_prompts[model].reshape(-1, prompt_feature_count), reg_target_g.reshape(-1,))
        bonus[model] = reg.predict(prompt_emb)[0]

    c = rbf_kernel(prompt_emb.reshape(1, prompt_feature_count), prompt_emb.reshape(1, prompt_feature_count), gamma=gamma).flat[0]
    return np.sqrt(np.maximum(np.zeros((model_count,)), c * np.ones((model_count,)) - bonus))
