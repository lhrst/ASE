import torch
import numpy as np
import libmr
import sklearn.metrics as sk


def distance_cal(model,train_loader,num_class=40):
    dist={}
    correct_classified_logit= [[] for _ in range(num_class)]
    mav = []
    with torch.no_grad():
        model.eval()
        for k,(data,target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            _, logit=model(data)
            for score, t in zip(logit,target):
                if torch.argmax(score)==t:
                    correct_classified_logit[t].append(score.detach().cpu().numpy())
        correct_classified_logit=np.asarray(correct_classified_logit, dtype=object)
        for i in range(num_class):
            mav.append(np.asarray(correct_classified_logit[i]).mean(axis=0))
        mav=np.asarray(mav)
        mav=np.repeat(mav[:,np.newaxis,:],len(mav),axis=1)
        for i in range(num_class) :
            dist[i]=0.005*sk.pairwise_distances(correct_classified_logit[i],mav[i],metric='euclidean').mean(axis=1)+\
                    sk.pairwise_distances(correct_classified_logit[i],mav[i],metric='cosine').mean(axis=1)
    return dist,mav[:,0,:]


def weibull_tailfitting(network,train_loader,eta,num_class=40):
    dist_cal, dist_mean = distance_cal(network, train_loader, num_class = num_class)
    weibull_model={}
    for category in range(num_class):
        weibull_model[category]={}
        weibull_model[category]['distance']=dist_cal[category]
        weibull_model[category]['mean_vec']=dist_mean[category]
        weibull_model[category]['weibull_model']=[]
        mr=libmr.MR()
        tailtofit=np.sort(dist_cal[category])[-eta:]
        mr.fit_high(tailtofit,len(tailtofit))
        weibull_model[category]['weibull_model'].append(mr)
    return weibull_model


def input_cal(model,ood_loader,alpha):
    input_score=[]
    top_class=[]
    with torch.no_grad():
        model.eval()
        for k, (data, target) in enumerate(ood_loader):
            data, target = data.cuda(), target.cuda()
            _, logit = model(data)
            top_=logit.sort()[1][:,-alpha:]
            input_score.extend(logit.detach().cpu().numpy())
            top_class.extend(torch.flip(top_.detach().cpu(),[1]).numpy())
    return input_score,top_class


def query_weibull(category, weibull_model):
    return [weibull_model[category]['mean_vec'],
            weibull_model[category]['distance'],
            weibull_model[category]['weibull_model']]


def calc_distance(input_score,mean_vector):
    result = 0.005*sk.pairwise_distances(np.stack((input_score,input_score)), np.stack((mean_vector,mean_vector)), metric='euclidean')[0][0]+\
             sk.pairwise_distances(np.stack((input_score,input_score)), np.stack((mean_vector,mean_vector)), metric='cosine')[0][0]
    return result


def compute_openmax_prob(scores,scores_u):
    prob_scores,prob_unknowns =[],[]
    for s,su in zip(scores,scores_u):
        channel_scores=np.exp(s)
        channel_unknown =np.exp(np.sum(su))
        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append((channel_scores / total_denom).tolist())
        prob_unknowns.append([channel_unknown / total_denom])
    modified_scores=[]
    for i in range(len(prob_scores)):
        modified_scores.append(prob_scores[i]+prob_unknowns[i])
    return modified_scores


def openmax(model,loader,weibull_model,alpha,num_class=40,weight=True):
    input_score, top_class = input_cal(model, loader, alpha)
    omega = np.zeros((len(input_score), num_class))
    if weight:
        alpha_weights = [1.0-((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    else :
        alpha_weights = [1 for i in range(1, alpha + 1)]
    # print(alpha_weights)
    # print(omega.shape)
    for i in range(len(input_score)):
        omega[i][top_class[i]]=alpha_weights
    scores=[]
    score_u=[]
    for i in range(len(input_score)):
        wscore_list = []
        score_channel=input_score[i].copy()
        score_channel_u=np.zeros(num_class)
        for category in top_class[i]:
            mav,_, w_model = query_weibull(category, weibull_model)
            channel_dist = calc_distance(input_score[i],mav)
            wscore = w_model[0].w_score(channel_dist)
            modified_score=input_score[i][category]*(1-wscore*omega[i][category])
            score_channel[category]=modified_score
            score_channel_u[category]=input_score[i][category]-modified_score
            wscore_list.append(wscore)
        scores.append(score_channel)
        score_u.append(score_channel_u)
    scores=np.asarray(scores)
    score_u=np.asarray(score_u)
    openmax_prob = np.array(compute_openmax_prob(scores, score_u))
    return openmax_prob, np.asarray(input_score)
