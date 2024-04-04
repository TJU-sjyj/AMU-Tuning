
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize

def logit_normalize(logit):
    logits_std = torch.std(logit, dim=1, keepdim=True)
    logits_mean = torch.mean(logit, dim=1, keepdim=True)
    logit = (logit - logits_mean) / logits_std
    return logit

def uncertainty(logits, type, power):
    softmax_fun = nn.Softmax(dim=-1) # sofemax-norm to get probability distribution
    logits = softmax_fun(logits)
    if type == 'entropy':
        entropy = -torch.sum(logits * torch.log2(logits), dim=-1, keepdim=True) / torch.log2(torch.tensor(logits.shape[-1]).float())
        entropy =  (entropy * power).exp() 
        return entropy
    elif type == 'energy':
        max_values = logits.max(dim=-1, keepdim=True).values
        logits = logits - max_values
        tau = 2
        energy = tau * (torch.log(torch.sum(torch.exp(logits / tau), dim=-1, keepdim=True)) + max_values)
        return 1.0 / (energy ** power)
    elif type == 'max':
        max_values = logits.max(dim=-1, keepdim=True).values
        return 1.0 / (max_values) ** power
    elif type == 'max-min':
        diff = logits.max(dim=-1, keepdim=True).values - logits.min(dim=-1, keepdim=True).values
        return 1.0 / diff ** power 
    elif type == 'var':
        variance = torch.std(logits, dim=-1, keepdim=True)
        return variance
    elif type == 'top5':
        top2 = logits.topk(5, dim=-1).values
        confidence = (top2[:, 0] - top2[:, -1]).unsqueeze(-1)
        return 1.0 / (confidence) ** power
        
    elif type == 'moment':
        mu = torch.mean(logits, dim=-1, keepdim=True)
        sigma = torch.std(logits, dim=-1, keepdim=True)
        normalized_logits = (logits - mu) / sigma
        moment_4 = torch.mean(normalized_logits ** 4, dim=-1, keepdim=True)
        return 1 / ((moment_4 / 250) ** power)
        #return 1.5 - 0.12 * moment_4
        #return filp(moment_4)
        #return (- moment_4 * power).exp() 
    elif type == 'none':
        return torch.tensor(1.0)
    else:
        raise RuntimeError('Invalid uncertainty type.')

class Linear_Adapter(nn.Module):
    def __init__(self, feat_dim, class_num, sample_features=None):
        super().__init__()
        self.fc = nn.Linear(feat_dim, class_num, bias=False)
        # init
        if sample_features is not None:
            print('init adapter weight by training samples...')
            aux_features, aux_labels = sample_features[0], sample_features[1]
            aux_features = aux_features

            init_weight = torch.zeros(feat_dim, class_num, device=aux_features.device) 
            for i in range(len(aux_labels)):
                init_weight[:, aux_labels[i]] += aux_features[i]
    
            feat_per_class = len(aux_labels) / class_num
            init_weight = init_weight / feat_per_class
            self.fc.weight = nn.Parameter(init_weight.t())
        else:
            print('init adapter weight by random...')
        
    def forward(self, feat):
        return self.fc(feat)


tfm_clip = Compose([Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
tfm_aux = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class AMU_Model(nn.Module):
    def __init__(self, clip_model, aux_model, sample_features, clip_weights, feat_dim, class_num, lambda_merge, alpha,  uncent_type, uncent_power):
        super().__init__()
        self.clip_model = clip_model
        self.aux_model = aux_model
        self.clip_weights = clip_weights
        self.aux_adapter = Linear_Adapter(feat_dim, class_num, sample_features=sample_features)
        
        self.lambda_merge = lambda_merge
        self.uncent_type = uncent_type
        self.uncent_power = uncent_power
        self.alpha = alpha
    
    def forward(self, images=None, clip_features=None, aux_features=None, labels=None):
        
        if images is not None:
            clip_features, aux_features = self.forward_feature(images)
        clip_features /= clip_features.norm(dim=-1, keepdim=True)
        aux_features /= aux_features.norm(dim=-1, keepdim=True)
        clip_logits, aux_logits = self.forward_adapter(clip_features, aux_features)
        
        # fusion
        factor = uncertainty(
            clip_logits.float(),
            power=self.uncent_power,
            type=self.uncent_type
        )
        logits = clip_logits + factor * aux_logits * self.alpha
        
        # loss
        if labels is not None:
            loss_merge = F.cross_entropy(logits, labels)
            loss_aux = F.cross_entropy(aux_logits, labels)
            loss = self.lambda_merge * loss_merge + (1 - self.lambda_merge) * loss_aux
        else:
            loss_merge = None
            loss_aux = None
            loss = None
            
        return_dict = {
            "logits": logits,
            "clip_logits": clip_logits,
            "aux_logits": aux_logits,
            "loss": loss,
            "loss_merge": loss_merge,
            "loss_aux": loss_aux,            
        }
        
        return return_dict
        
    def forward_feature(self, images):
        # CLIP branch
        clip_features =self.clip_model.encode_image(tfm_clip(images))
        # AUX branch
        aux_feature = self.aux_model(tfm_aux(images))
        return clip_features, aux_feature
    
    def forward_adapter(self, clip_features, aux_features):
        # logits
        clip_logits = 100. * clip_features @ self.clip_weights
        aux_logits = self.aux_adapter(aux_features)
        aux_logits = logit_normalize(aux_logits)
        return clip_logits, aux_logits 