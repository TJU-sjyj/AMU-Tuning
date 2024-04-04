import logging
import datetime
from PIL import Image
from tqdm import tqdm

import clip
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor, RandomResizedCrop, RandomHorizontalFlip


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

tfm_train_base = Compose([
            RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=BICUBIC),
            RandomHorizontalFlip(p=0.5),
            ToTensor()
            ]
        )

tfm_test_base = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
    ])

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def gpt_clip_classifier(classnames, clip_model, template):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def load_aux_weight(args, model, train_loader_cache, tfm_norm):
    if args.load_aux_weight == False:
        aux_features = []
        aux_labels = []
        with torch.no_grad():
            for augment_idx in range(args.augment_epoch):
                aux_features_current = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, args.augment_epoch))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = model(tfm_norm(images))
                    aux_features_current.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        aux_labels.append(target)
                aux_features.append(torch.cat(aux_features_current, dim=0).unsqueeze(0))
         
        aux_features = torch.cat(aux_features, dim=0).mean(dim=0).cuda()
        aux_features /= aux_features.norm(dim=-1, keepdim=True)
        
        aux_labels = torch.cat(aux_labels).cuda()

        torch.save(aux_features, args.cache_dir + f'/aux_feature_' + str(args.shots) + "shots.pt")
        torch.save(aux_labels, args.cache_dir + f'/aux_labels_' + str(args.shots) + "shots.pt")

    else:
        aux_features = torch.load(args.cache_dir + f'/aux_feature_' + str(args.shots) + "shots.pt")
        aux_labels = torch.load(args.cache_dir + f'/aux_labels_' + str(args.shots) + "shots.pt")
    return aux_features, aux_labels

def load_test_features(args, split, model, loader, tfm_norm, model_name):
    if args.load_pre_feat == False:
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                    image_features = model.encode_image(tfm_norm(images)) # for clip model
                else:
                    image_features = model(tfm_norm(images))
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        features = features.cuda()
        torch.save(features, args.cache_dir + f"/{model_name}_" + split + "_f.pt")
        torch.save(labels, args.cache_dir + f"/{model_name}_" + split + "_l.pt")
        
    else:
        features = torch.load(args.cache_dir + f"/{model_name}_" + split + "_f.pt")
        labels = torch.load(args.cache_dir + f"/{model_name}_" + split + "_l.pt")
    return features, labels

def config_logging(args):
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M')
    now = datetime.datetime.now().strftime("%m-%d-%H_%M")
    # FileHandler
    fh = logging.FileHandler(f'result/{args.exp_name}_{now}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger 
