
# libraries: 
import torch 
import sklearn
import numpy as np 
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
import random
from scipy.optimize import minimize

# base lines: 
# if all False then DAK is running
RANDOM = False 
ONEARM = True
PAK = False
MIX = False    

# in case of one arm oracle baseline activation 
oracleChoice = 0

# results file configuration
Experiment = "Image-Captioning"
Algorithm = 'ONE-ARM-ORACLE'
Objective = "I-JRKE"             # or  "JKD"

constant = 1e-4   # UCB Bonus 
runs = 10
model_count = 3
iterations = 2000 
prompt_feature_count = 512
image_feature_count = 768
delta = 5.8       # KRR Bonus
alpha = 1.        # Regression regularization term multiplier
eta = np.sqrt(2. * np.log(2. * model_count / delta) )   # Exploration coefficient
gamma = 1         # RBK kernel band width
lambd =  1        # Diversity term multiplier
if PAK:
    lambd = 0
kernel = sklearn.metrics.pairwise.rbf_kernel   

all_models = [ "kandinsky" , "sdxl",  "gigagan" ]

def compute_kid(real_embeddings, fake_embeddings, gamma=gamma):

    gamma_rbf = gamma
    k_real = kernel(real_embeddings, gamma=gamma_rbf)
    k_fake = kernel(fake_embeddings, gamma=gamma_rbf)
    k_cross = kernel(real_embeddings, fake_embeddings, gamma=gamma_rbf)
    
    mmd2 = k_real.mean() + k_fake.mean() - 2 * k_cross.mean()
    return mmd2

def ucb_radius_estimator(prompt_normalized_emb, observed_prompts, kernel): 
    bonus = np.zeros((model_count,))
    for model in range(model_count):

        reg_target_g = kernel(observed_prompts[model], prompt_normalized_emb, gamma=gamma).flatten()
        reg_bon_g = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma).fit(observed_prompts[model].reshape(-1, prompt_feature_count), reg_target_g.reshape(-1, ))
        bonus[model] = reg_bon_g.predict(prompt_normalized_emb)[0]

    c = kernel(X=prompt_normalized_emb.reshape(1, prompt_feature_count), Y=prompt_normalized_emb.reshape(1, prompt_feature_count), gamma = gamma ).flat[0]
    return np.sqrt(np.maximum(np.zeros((model_count,)), c * np.ones((model_count,)) - bonus))

divs = []
clips = []
dists = []

for run in range(runs):

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
    observed_images_per_model = [np.empty((1, image_feature_count )) for _ in range(model_count)]
    observed_targets_per_model = [np.empty((1, image_feature_count )) for _ in range(model_count)]
    clip_scores_per_model = [np.empty(1,) for _ in range(model_count)]
    div_scores_per_model = [[np.empty(1,) for _ in range(model_count)] for _ in range(model_count)]    
    pred_score_per_model = np.ones(model_count) * np.inf
    visits_per_model = np.zeros((model_count, ), dtype=int)

    for iteration in range(iterations):
         
        embedding = NotImplementedError   # sample a prompt(context) from prompts distribution and extract clip features
        prompt_normalized_emb = embedding.cpu().numpy().flatten().reshape(1, -1) / np.linalg.norm(embedding.cpu().numpy().flatten().reshape(1, -1), axis=1, keepdims=True)
        estimated_bounds = np.empty((model_count, ))

        if iteration < model_count: 
            selected_model = np.argmax(pred_score_per_model)

        else:
            estimated_clip_scores = np.empty((model_count, ))
            estimated_div_scores = np.empty((model_count, model_count ))
            estimated_div_scores_lin = np.empty((model_count))

            for model in range(model_count): 
                estimated_clip_scores[model] = clip_regressors[model].predict(prompt_normalized_emb)[0]
                estimated_div_scores_lin[model] = div_regressors[model][model].predict(prompt_normalized_emb)[0]
                for second_model in range(model_count):
                    estimated_div_scores[model][second_model] = div_regressors[model][second_model].predict(prompt_normalized_emb)[0]  

            estimated_bounds =  ucb_radius_estimator(prompt_normalized_emb, observed_prompts_per_model, kernel)
            alpha_init = np.ones(model_count) / model_count  
            D_sym = (estimated_div_scores + estimated_div_scores.T) / 2

            def objective(alpha):
                linear_term1 = estimated_clip_scores @ alpha
                linear_term2 = (eta * estimated_bounds) @ alpha
                linear_term3 = - lambd * estimated_div_scores_lin @ alpha 
                quadratic_term = - lambd * alpha.T @ D_sym @ alpha
                if MIX:
                    return - (quadratic_term + linear_term1 + linear_term2 )
                else :
                    return - (linear_term1 + linear_term2 + linear_term3)

            constraints = [
                {'type': 'eq', 'fun': lambda alpha: np.sum(alpha) - 1}
            ]
            bounds = [(0, 1) for _ in range(model_count)]
            result = minimize(objective, alpha_init, method='SLSQP', bounds=bounds, constraints=constraints)

            alpha_value = result.x
            if RANDOM:
                alphaa_value = np.ones(model_count) / model_count
            alpha_probabilities = np.maximum(alphaa_value, 0)
            alpha_probabilities /= np.sum(alpha_probabilities)
            selected_model = np.random.choice(len(alpha_probabilities), p=alpha_probabilities)
            if ONEARM:
                selected_model = oracleChoice

        if selected_model == 0:
            embedding = NotImplementedError   # generate an image with model 0 for observed prompt and extract dino embedding
            embedding_clip = NotImplementedError   # extract clip embedding for generated image 
            image_normalized_emb = embedding.reshape(1, -1) / np.linalg.norm(embedding.reshape(1, -1), axis=1, keepdims=True)

        elif selected_model == 1:
            embedding = NotImplementedError   # generate an image with model 1 for observed prompt and extract dino embedding
            embedding_clip = NotImplementedError   # extract clip embedding for generated image 
            image_normalized_emb = embedding.reshape(1, -1) / np.linalg.norm(embedding.reshape(1, -1), axis=1, keepdims=True)

        elif selected_model == 2:
            embedding = NotImplementedError   # generate an image with model 2 for observed prompt and extract dino embedding
            embedding_clip = NotImplementedError   # extract clip embedding for generated image 
            image_normalized_emb = embedding.reshape(1, -1) / np.linalg.norm(embedding.reshape(1, -1), axis=1, keepdims=True)                    
       
        target_normalized_emb = NotImplementedError  # choose the reference and extract the embedding
        fake_embeddings.append(image_normalized_emb)
        real_embeddings.append(target_normalized_emb)

        if iteration > 0 :
            s = 0
            for m in range(model_count):
                for sample in range(visits_per_model[m]):
                    c1 = kernel(X=prompt_normalized_emb.reshape(1, prompt_feature_count), Y=observed_prompts_per_model[m][sample].reshape(1, prompt_feature_count), gamma = gamma ).flat[0]
                    c2 = kernel(X=image_normalized_emb.reshape(1, image_feature_count), Y=observed_images_per_model[m][sample].reshape(1, image_feature_count), gamma = gamma ).flat[0]
                    s += (c1**2 * c2**2)
            running_div += (2*s + 1)
        all_joint_rke.append(running_div/((iteration+1)**2))

        cosine_angle = (embedding_clip.squeeze(0)).dot(prompt_normalized_emb.squeeze(0))
        score = max(0., 100. * cosine_angle)
        running_clip += score
        all_clips.append(running_clip/(iteration + 1))
        all_dists.append(compute_kid(np.array(real_embeddings).reshape((iteration+1, -1)), np.array(fake_embeddings).reshape((iteration+1, -1))))
 
        if visits_per_model[selected_model] == 0:
            observed_prompts_per_model[selected_model][0] = prompt_normalized_emb
            clip_scores_per_model[selected_model][0] = score / 100
            observed_images_per_model[selected_model][0] = image_normalized_emb

            if Objective == "I-JRKE":
                for i in range(model_count):
                    div = 0
                    for sample in range(visits_per_model[i]):
                        c1 = kernel(X=prompt_normalized_emb.reshape(1, prompt_feature_count), Y=observed_prompts_per_model[i][sample].reshape(1, prompt_feature_count), gamma = gamma ).flat[0]
                        c2 = kernel(X=image_normalized_emb.reshape(1, image_feature_count), Y=observed_images_per_model[i][sample].reshape(1, image_feature_count), gamma = gamma ).flat[0]
                        div += (c1**2 * c2**2)
                    bound = np.sqrt(constant)
                    if visits_per_model[i] != 0:
                        div_scores_per_model[selected_model][i][0] = div/visits_per_model[i] - bound
                    else:
                        div_scores_per_model[selected_model][i][0] = div - bound

            else:
                for i in range(model_count):
                    dist = 0
                    for sample in range(visits_per_model[i]):
                        k =  kernel(X=observed_images_per_model[i][sample].reshape(1, image_feature_count), Y=image_normalized_emb.reshape(1, image_feature_count), gamma = gamma ).flat[0] + kernel(X=observed_targets_per_model[i][sample].reshape(1, image_feature_count), Y=target_normalized_emb.reshape(1, image_feature_count), gamma = gamma ).flat[0] - kernel(X=observed_images_per_model[i][sample].reshape(1, image_feature_count), Y=target_normalized_emb.reshape(1,image_feature_count), gamma = gamma ).flat[0] - kernel(X=observed_targets_per_model[i][sample].reshape(1, image_feature_count), Y=image_normalized_emb.reshape(1, image_feature_count), gamma = gamma ).flat[0]
                        c1 = kernel(X=prompt_normalized_emb.reshape(1, prompt_feature_count), Y=observed_prompts_per_model[i][sample].reshape(1, prompt_feature_count), gamma = gamma ).flat[0]
                        dist += (c1 * k)
                    if visits_per_model[i] != 0:
                        dist = dist / (visits_per_model[i])
                    bound = np.sqrt(constant)
                    div_scores_per_model[selected_model][i][0] =  dist - bound

                observed_targets_per_model[selected_model][0] = target_normalized_emb

        else:
            observed_prompts_per_model[selected_model] = np.concatenate((observed_prompts_per_model[selected_model], prompt_normalized_emb), axis=0)
            clip_scores_per_model[selected_model] = np.concatenate((clip_scores_per_model[selected_model], np.array([score / 100])), axis=0)
            observed_images_per_model[selected_model] = np.concatenate((observed_images_per_model[selected_model], image_normalized_emb), axis=0)

            if Objective == 'I-JRKE':
                for i in range(model_count):
                    div = 0
                    for sample in range(visits_per_model[i]):
                        c1 = kernel(X=prompt_normalized_emb.reshape(1, prompt_feature_count), Y=observed_prompts_per_model[i][sample].reshape(1, prompt_feature_count), gamma = gamma ).flat[0]
                        c2 = kernel(X=image_normalized_emb.reshape(1, image_feature_count), Y=observed_images_per_model[i][sample].reshape(1, image_feature_count), gamma = gamma ).flat[0]
                        div += (c1**2 * c2**2)
                    bound = np.sqrt(constant/(visits_per_model[i]))
                    if visits_per_model[i] != 0:
                        div_scores_per_model[selected_model][i] = np.concatenate((div_scores_per_model[selected_model][i],  np.array([(div/visits_per_model[i]) - bound])), axis=0)
                    else:
                        div_scores_per_model[selected_model][i] = np.concatenate((div_scores_per_model[selected_model][i],  np.array([div - bound])), axis=0)

            else:
                for i in range(model_count):
                    dist = 0
                    for sample in range(visits_per_model[i]):
                        k =  kernel(X=observed_images_per_model[i][sample].reshape(1, image_feature_count), Y=image_normalized_emb.reshape(1, image_feature_count), gamma = gamma ).flat[0] + kernel(X=observed_targets_per_model[i][sample].reshape(1, image_feature_count), Y=target_normalized_emb.reshape(1, image_feature_count), gamma = gamma ).flat[0] - kernel(X=observed_images_per_model[i][sample].reshape(1, image_feature_count), Y=target_normalized_emb.reshape(1,image_feature_count), gamma = gamma ).flat[0] - kernel(X=observed_targets_per_model[i][sample].reshape(1, image_feature_count), Y=image_normalized_emb.reshape(1, image_feature_count), gamma = gamma ).flat[0]
                        c1 = kernel(X=prompt_normalized_emb.reshape(1, prompt_feature_count), Y=observed_prompts_per_model[i][sample].reshape(1, prompt_feature_count), gamma = gamma ).flat[0]
                        dist += (c1 * k)
                    dist = dist / (visits_per_model[i])
                    bound = np.sqrt(constant/(visits_per_model[i]))
                    div_scores_per_model[selected_model][i] = np.concatenate((div_scores_per_model[selected_model][i], np.array([dist - bound])), axis=0)

                observed_targets_per_model[selected_model] = np.concatenate((observed_targets_per_model[selected_model], target_normalized_emb), axis=0)


        pred_score_per_model[selected_model] = 0
        visits_per_model[selected_model] += 1

        y = clip_scores_per_model[selected_model].reshape(-1)
        x = observed_prompts_per_model[selected_model].reshape(-1, prompt_feature_count)
        
        clip_regressors[selected_model] = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma).fit(x,y)
        for i in range(model_count):
            div_regressors[selected_model][i] = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma).fit(x, div_scores_per_model[selected_model][i].reshape(-1))

    sorted_models = sorted(zip(all_models, visits_per_model), key=lambda x: x[1], reverse=True)

    print("\nModels sorted by selection frequency:")
    for model, visits in sorted_models:
        print(f"{model}: {int(visits)} selections")
    divs.append(all_joint_rke)
    clips.append(all_clips)
    dists.append(all_dists)

avg_div = np.mean(np.array(divs), axis=0)
avg_clip = np.mean(np.array(clips), axis=0)
avg_dist = np.mean(np.array(dists), axis=0)

with open(f"{Experiment}_{Algorithm}.txt", 'w') as f:

    if Objective == "I-JRKE":
        f.write("Average 1/div :\n")
        f.write(", ".join([str(x) for x in avg_div]) + "\n")

    else:
        f.write("Average dist :\n")
        f.write(", ".join([str(x) for x in avg_dist]) + "\n")

    f.write("Average clip :\n")
    f.write(", ".join([str(x) for x in avg_clip]) + "\n")
