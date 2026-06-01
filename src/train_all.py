import os
import sys
import re
import math
import argparse
import json
import logging
import torch
from torch import optim
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Subset, ConcatDataset
from clip import clip
from torch.optim.lr_scheduler import LambdaLR

from utils.train_utils import EMA, log_result_table
from utils.dataset import PetWithFilename, CUBWithFilename, AugmentedDataset

logging.getLogger("PIL").setLevel(logging.WARNING)

id2label_cub = {0: 'black footed albatross', 1: 'laysan albatross', 2: 'sooty albatross', 3: 'groove billed ani', 4: 'crested auklet', 5: 'least auklet', 6: 'parakeet auklet', 7: 'rhinoceros auklet', 8: 'brewer blackbird', 9: 'red winged blackbird', 10: 'rusty blackbird', 11: 'yellow headed blackbird', 12: 'bobolink', 13: 'indigo bunting', 14: 'lazuli bunting', 15: 'painted bunting', 16: 'cardinal', 17: 'spotted catbird', 18: 'gray catbird', 19: 'yellow breasted chat', 20: 'eastern towhee', 21: 'chuck will widow', 22: 'brandt cormorant', 23: 'red faced cormorant', 24: 'pelagic cormorant', 25: 'bronzed cowbird', 26: 'shiny cowbird', 27: 'brown creeper', 28: 'american crow', 29: 'fish crow', 30: 'black billed cuckoo', 31: 'mangrove cuckoo', 32: 'yellow billed cuckoo', 33: 'gray crowned rosy finch', 34: 'purple finch', 35: 'northern flicker', 36: 'acadian flycatcher', 37: 'great crested flycatcher', 38: 'least flycatcher', 39: 'olive sided flycatcher', 40: 'scissor tailed flycatcher', 41: 'vermilion flycatcher', 42: 'yellow bellied flycatcher', 43: 'frigatebird', 44: 'northern fulmar', 45: 'gadwall', 46: 'american goldfinch', 47: 'european goldfinch', 48: 'boat tailed grackle', 49: 'eared grebe', 50: 'horned grebe', 51: 'pied billed grebe', 52: 'western grebe', 53: 'blue grosbeak', 54: 'evening grosbeak', 55: 'pine grosbeak', 56: 'rose breasted grosbeak', 57: 'pigeon guillemot', 58: 'california gull', 59: 'glaucous winged gull', 60: 'heermann gull', 61: 'herring gull', 62: 'ivory gull', 63: 'ring billed gull', 64: 'slaty backed gull', 65: 'western gull', 66: 'anna hummingbird', 67: 'ruby throated hummingbird', 68: 'rufous hummingbird', 69: 'green violetear', 70: 'long tailed jaeger', 71: 'pomarine jaeger', 72: 'blue jay', 73: 'florida jay', 74: 'green jay', 75: 'dark eyed junco', 76: 'tropical kingbird', 77: 'gray kingbird', 78: 'belted kingfisher', 79: 'green kingfisher', 80: 'pied kingfisher', 81: 'ringed kingfisher', 82: 'white breasted kingfisher', 83: 'red legged kittiwake', 84: 'horned lark', 85: 'pacific loon', 86: 'mallard', 87: 'western meadowlark', 88: 'hooded merganser', 89: 'red breasted merganser', 90: 'mockingbird', 91: 'nighthawk', 92: 'clark nutcracker', 93: 'white breasted nuthatch', 94: 'baltimore oriole', 95: 'hooded oriole', 96: 'orchard oriole', 97: 'scott oriole', 98: 'ovenbird', 99: 'brown pelican', 100: 'white pelican', 101: 'western wood pewee', 102: 'sayornis', 103: 'american pipit', 104: 'whip poor will', 105: 'horned puffin', 106: 'common raven', 107: 'white necked raven', 108: 'american redstart', 109: 'geococcyx', 110: 'loggerhead shrike', 111: 'great grey shrike', 112: 'baird sparrow', 113: 'black throated sparrow', 114: 'brewer sparrow', 115: 'chipping sparrow', 116: 'clay colored sparrow', 117: 'house sparrow', 118: 'field sparrow', 119: 'fox sparrow', 120: 'grasshopper sparrow', 121: 'harris sparrow', 122: 'henslow sparrow', 123: 'le conte sparrow', 124: 'lincoln sparrow', 125: 'nelson sharp tailed sparrow', 126: 'savannah sparrow', 127: 'seaside sparrow', 128: 'song sparrow', 129: 'tree sparrow', 130: 'vesper sparrow', 131: 'white crowned sparrow', 132: 'white throated sparrow', 133: 'cape glossy starling', 134: 'bank swallow', 135: 'barn swallow', 136: 'cliff swallow', 137: 'tree swallow', 138: 'scarlet tanager', 139: 'summer tanager', 140: 'artic tern', 141: 'black tern', 142: 'caspian tern', 143: 'common tern', 144: 'elegant tern', 145: 'forsters tern', 146: 'least tern', 147: 'green tailed towhee', 148: 'brown thrasher', 149: 'sage thrasher', 150: 'black capped vireo', 151: 'blue headed vireo', 152: 'philadelphia vireo', 153: 'red eyed vireo', 154: 'warbling vireo', 155: 'white eyed vireo', 156: 'yellow throated vireo', 157: 'bay breasted warbler', 158: 'black and white warbler', 159: 'black throated blue warbler', 160: 'blue winged warbler', 161: 'canada warbler', 162: 'cape may warbler', 163: 'cerulean warbler', 164: 'chestnut sided warbler', 165: 'golden winged warbler', 166: 'hooded warbler', 167: 'kentucky warbler', 168: 'magnolia warbler', 169: 'mourning warbler', 170: 'myrtle warbler', 171: 'nashville warbler', 172: 'orange crowned warbler', 173: 'palm warbler', 174: 'pine warbler', 175: 'prairie warbler', 176: 'prothonotary warbler', 177: 'swainson warbler', 178: 'tennessee warbler', 179: 'wilson warbler', 180: 'worm eating warbler', 181: 'yellow warbler', 182: 'northern waterthrush', 183: 'louisiana waterthrush', 184: 'bohemian waxwing', 185: 'cedar waxwing', 186: 'american three toed woodpecker', 187: 'pileated woodpecker', 188: 'red bellied woodpecker', 189: 'red cockaded woodpecker', 190: 'red headed woodpecker', 191: 'downy woodpecker', 192: 'bewick wren', 193: 'cactus wren', 194: 'carolina wren', 195: 'house wren', 196: 'marsh wren', 197: 'rock wren', 198: 'winter wren', 199: 'common yellowthroat'}

idx_to_classname_pet = {0: 'Abyssinian', 1: 'American Bulldog', 2: 'American Pit Bull Terrier', 3: 'Basset Hound', 4: 'Beagle', 5: 'Bengal', 6: 'Birman', 7: 'Bombay', 8: 'Boxer', 9: 'British Shorthair', 10: 'Chihuahua', 11: 'Egyptian Mau', 12: 'English Cocker Spaniel', 13: 'English Setter', 14: 'German Shorthaired', 15: 'Great Pyrenees', 16: 'Havanese', 17: 'Japanese Chin', 18: 'Keeshond', 19: 'Leonberger', 20: 'Maine Coon', 21: 'Miniature Pinscher', 22: 'Newfoundland', 23: 'Persian', 24: 'Pomeranian', 25: 'Pug', 26: 'Ragdoll', 27: 'Russian Blue', 28: 'Saint Bernard', 29: 'Samoyed', 30: 'Scottish Terrier', 31: 'Shiba Inu', 32: 'Siamese', 33: 'Sphynx', 34: 'Staffordshire Bull Terrier', 35: 'Wheaten Terrier', 36: 'Yorkshire Terrier'}

CUSTOM_TEMPLATES = {
    'pets': 'a photo of a {}, a type of pet',
    'pet_visible': 'a photo of a {}, a type of pet',
    'flowers': 'a photo of a {}, a type of flower',
    'cars': 'a photo of a {}, a type of vehicle',
    'cub': 'a photo of a {}, a type of bird',
    'cub_auroc': 'a photo of a {}, a type of bird',
    'cub_visible': 'a photo of a {}, a type of bird',
}

def evaluate(model, dataloader, text_ids, device, temperature):
    alpha = 0.5 
    model.eval()
    all_text_features = []
    for i in range(text_ids.size(0)):
        text_features = model.encode_text(text_ids[i].to(device))
        text_features = text_features.detach().cpu()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_text_features.append(text_features.unsqueeze(0))
    text_features = torch.concat(all_text_features)

    num_true_cls = 0
    num_true_top2 = 0
    num_true_top3 = 0
    num_true_top5 = 0
    num_true_mean = 0
    num_true_score_top2 = 0
    num_true_score_top3 = 0
    num_true_global_softmax = 0
    num_true_global_softmax_top3 = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch[1]
            pixel_values = batch[0]
            image_features = model.encode_image(pixel_values.to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (1/temperature) * torch.einsum('bd,kpd->bkp', image_features.detach().cpu(), text_features)
            concept_logits = logits[:, :, 1:]          # (B, K, N)
            B, K, N = concept_logits.shape
            # flatten over (K, N)
            concept_logits_flat = concept_logits.reshape(B, -1)
            # global softmax over all classes & concepts
            concept_probs_flat = torch.softmax(concept_logits_flat, dim=-1)
            # reshape back
            concept_probs = concept_probs_flat.reshape(B, K, N)
            # class-wise average
            score_global_softmax = concept_probs.mean(dim=-1)  # (B, K)
            num_true_global_softmax += (
                score_global_softmax.argmax(-1) == labels
            ).sum()
            score_global_softmax_top3 = concept_probs.sort(-1, descending=True).values[:, :, :3].mean(dim=-1)  # (B, K)
            num_true_global_softmax_top3 += (
                score_global_softmax_top3.argmax(-1) == labels
            ).sum()
            num_true_cls += (logits[:, :, 0].argmax(-1) == labels).sum()
            # compute combined score (class token + concept mean)
            concept_logit_sorted = logits[:, :, 1:].sort(-1, descending=True).values
            score_top2 = alpha * logits[:, :, 0] + (1-alpha) * concept_logit_sorted[:, :, :2].mean(-1)
            score_top3 = alpha * logits[:, :, 0] + (1-alpha) * concept_logit_sorted[:, :, :3].mean(-1)
            num_true_score_top2 += (score_top2.argmax(-1) == labels).sum()
            num_true_score_top3 += (score_top3.argmax(-1) == labels).sum()
            # simple mean of top-K concepts
            num_true_top2 += (concept_logit_sorted[:, :, :2].mean(-1).argmax(-1) == labels).sum()
            num_true_top3 += (concept_logit_sorted[:, :, :3].mean(-1).argmax(-1) == labels).sum()
            num_true_top5 += (concept_logit_sorted[:, :, :5].mean(-1).argmax(-1) == labels).sum()
            num_true_mean += (concept_logit_sorted.mean(-1).argmax(-1) == labels).sum()
            total += labels.size(0)
    logging.info(f"""
    class prediction: {num_true_cls/total}
    top2 concept mean: {num_true_top2/total}
    top3 concept mean: {num_true_top3/total}
    top5 concept mean: {num_true_top5/total}
    all concept mean: {num_true_mean/total}
    top2 (alpha={alpha}): {num_true_score_top2/total}
    top3 (alpha={alpha}): {num_true_score_top3/total}
    global softmax mean (Eq.3): {num_true_global_softmax/total}
   global softmax mean (top3): {num_true_global_softmax_top3/total}
        """)
    return [num_true_cls/total, num_true_top2/total, num_true_top3/total, num_true_top5/total, num_true_mean/total, num_true_score_top2/total, num_true_score_top3/total, num_true_global_softmax/total, num_true_global_softmax_top3/total]

def main(args):

    # load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_type == 'vit32':
        logging.info('use ViT-B/32')
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    elif args.model_type == 'vit16':
        logging.info('use ViT-B/16')
        model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
    elif args.model_type == 'rn50':
        logging.info('use RN50')
        model, preprocess = clip.load("RN50", device=device, jit=False) 

    ema = EMA(model, momentum=0.98) #
    
    if args.clip_w:
        theta_v0 = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
    else:
        theta_v0 = None

    if args.dataset == 'pet_visible':
        dataset = PetWithFilename(args.data_dir + '/oxford_pet', split='trainval', mode='all', transform=preprocess)
        dataset_test = PetWithFilename(args.data_dir + '/oxford_pet', split='test', mode='all', transform=preprocess)
        new_descriptors = {}
        with open('./descriptors/pet_concepts.json', 'r') as f:
            descriptors = json.load(f)
        for path in descriptors:
            new_descriptors[path] = list(descriptors[path].values())
        descriptors = new_descriptors
        idx_to_class = idx_to_classname_pet
        seen_classes = dataset.seen_classes
        logging.info(f'seen_classes: {seen_classes}')
    elif args.dataset == 'cub_visible':
        print('cub_visible')
        dataset = CUBWithFilename(root=args.data_dir + '/cub/CUB_200_2011/images/train', transform=preprocess, mode='all')
        dataset_test = CUBWithFilename(root=args.data_dir + '/cub/CUB_200_2011/images/test', transform=preprocess, mode='all')

        seen_classes = dataset_test.seen_classes
        logging.info(f'seen_classes: {seen_classes}')
        with open('./descriptors/my_cub2-1.json', 'rb') as f:
            descriptors = json.load(f)
        idx_to_class = id2label_cub

    if args.lvlm_output is not None:
        with open(args.lvlm_output, 'r') as f:
            lvlm_output = json.load(f)
        pseudo_label = {}
        for x in lvlm_output:
            table = lvlm_output[x]['raw_output']
            if args.hard_label:
                pseudo_label[x] = [float(val) >= args.hard_threshold for val in re.findall(r'[0-9]+\.[0-9]+', table)]
            else:
                pseudo_label[x] = [float(val) for val in re.findall(r'[0-9]+\.[0-9]+', table)]

    
    cnt = 0
    if args.lvlm_output_aug is not None:
        # load pseudo labels for augmented images
        with open(args.lvlm_output_aug, 'r') as f:
            lvlm_output2 = json.load(f)
        for x in lvlm_output2:
            new_x = os.path.basename(x)
            table = lvlm_output2[x]['raw_output']

            if args.hard_label:
                values = [float(val) >= args.hard_threshold for val in re.findall(r'[0-9]+\.[0-9]+', table)]
            else:
                values = [float(val) for val in re.findall(r'[0-9]+\.[0-9]+', table)]

            # add to pseudo_label only if not all-False
            if args.hard_label:
                if any(values):
                    pseudo_label[new_x] = values
                else:
                    cnt += 1
            else:
                if sum(values) > 0.5:
                    pseudo_label[new_x] = values
                else:
                    cnt += 1
        logging.info(f"visible label: {values}")
        logging.info(f"lvlm_output_aug len: {len(lvlm_output2)}")
        logging.info(f"all-false data count: {cnt}")

    # randomly shuffle and split indices into train/val
    num_samples = len(dataset)
    num_train = int(0.8 * num_samples)
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    dataset_train = Subset(dataset, train_idx)
    dataset_val = Subset(dataset, val_idx)

    
    if args.data_aug and args.aug_data_path:
        # mix in augmented dataset
        dataset_aug = AugmentedDataset(root=args.aug_data_path, transform=preprocess, valid_files=pseudo_label if args.lvlm_output_aug else None)
        logging.info(f'loading additional augmented dataset, length: {len(dataset_aug)}')
        logging.info(f'normal dataset num: {len(dataset_train)}, aug dataset num: {len(dataset_aug)}')
        dataset_train = ConcatDataset([dataset_train, dataset_aug])
    logging.info(f'total train dataset num: {len(dataset_train)}')


    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val   = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False)

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size_val, shuffle=False)

    # build text token IDs
    text_ids_rich = []
    text_ids_normal = []
    text_ids_train_rich = []
    text_ids_train_normal = []
    texts_train_rich = []
    texts_train_normal = []
    for classname in idx_to_class.values():
        class_concept = descriptors[classname]
        class_concept_rich = [f'{classname} with {x}.' for x in class_concept]
        if args.dataset == 'pets' or args.dataset == 'pet_visible':
            class_concept_normal = [f'an animal with **{x}**' for x in class_concept]
        elif args.dataset == 'flowers':
            class_concept_normal = [f'a flower with **{x}**' for x in class_concept]
        elif args.dataset == 'cars':
            class_concept_normal = [f'a car with **{x}**' for x in class_concept]
        elif args.dataset == 'cub' or args.dataset == 'cub_visible':
            class_concept_normal = [f'a bird with **{x}**' for x in class_concept]
        class_concept_rich = [CUSTOM_TEMPLATES[args.dataset].format(classname)] + class_concept_rich
        class_concept_normal = [CUSTOM_TEMPLATES[args.dataset].format(classname)] + class_concept_normal
        texts_train_rich.append(class_concept_rich)
        texts_train_normal.append(class_concept_normal)
        text_ids_rich.append(clip.tokenize(class_concept_rich).unsqueeze(0))
        text_ids_normal.append(clip.tokenize(class_concept_normal).unsqueeze(0))
    
    for id in seen_classes:
        text_ids_train_rich.append(text_ids_rich[id])
        text_ids_train_normal.append(text_ids_normal[id])
    text_ids_rich = torch.concat(text_ids_rich)
    text_ids_normal = torch.concat(text_ids_normal)
    text_ids_train_rich = torch.concat(text_ids_train_rich)
    text_ids_train_normal = torch.concat(text_ids_train_normal)

    if args.rich_concept:
        text_ids = text_ids_rich
        text_ids_train = text_ids_train_rich
        text_train = texts_train_rich
    else:
        text_ids = text_ids_normal
        text_ids_train = text_ids_train_normal
        text_train = texts_train_normal
    logging.info(text_train[0])

    temperature = nn.Parameter(torch.tensor(args.tau))
    main_parameters = [param for name, param in model.named_parameters() if "proj" not in name and "text_projection" not in name]
    optimizer = optim.AdamW([
        {'params': main_parameters, 'lr': args.main_lr, 'weight_decay' : 1e-3},
        {'params': [model.text_projection, model.visual.proj], 'lr': args.proj_lr, 'weight_decay' : 1e-2},
        {'params': temperature, 'lr': args.temperature_lr, 'weight_decay' : 1e-6},
    ], betas=(0.9,0.98),eps=1e-6) 

    total_steps = int(len(dataloader_train) * args.epochs)
    warmup_steps = int(0.15 * total_steps)
    print(f"{warmup_steps=}")
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    if args.warm_up:
        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    logging.info(f"scheduler: {scheduler}")

    best_acc = 0.0
    logging.info('zero shot (test seen):')
    acc_seen = evaluate(ema.ema_model, dataloader_test, text_ids, device, temperature=temperature)
    results = {
        "seen": [x.item() for x in acc_seen],
    }
    header = ['class_prediction', 'top2_mean', 'top3_mean', 'top5_mean', 'all_mean_pred', 'top2_alpha05', 'top3_alpha05', 'global softmax mean', 'global softmax top3']
    log_result_table(
        header,
        results,
        title="Zero-Shot Test Accuracy Summary",
        save_dir=args.save_path,
        filename='zero-shot'
    )

    logging.info('start training...')
    for epoch in range(args.epochs):
        logging.info(f'epoch{epoch}')
        train_loss = train_on_epoch(args, model, temperature, ema, theta_v0, scheduler, dataloader_train, text_ids_train, device, optimizer=optimizer, epoch=epoch, total_steps=total_steps, pseudo_label=pseudo_label if args.lvlm_output else None)
        ave_acc = evaluate(ema.ema_model, dataloader_val, text_ids_train, device, temperature=temperature)
        ave_acc = max(ave_acc)
        if ave_acc > best_acc:
            logging.info(f'save best checkpoint: epoch {epoch}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema.ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, f"{args.save_path}/best.pt") #just change to your preferred folder/filename
            best_acc = ave_acc


    # =====================
    # Test
    # =====================
    logging.info(f"---------Seen-----------")
    # test_acc = evaluate(ema.ema_model, dataloader_test, text_ids_rich, device)
    test_acc = evaluate(ema.ema_model, dataloader_test, text_ids, device, temperature=temperature)
    results = {
        "seen": [x.item() for x in test_acc],
    }
    log_result_table(
        header,
        results,
        title="Final Test Accuracy Summary",
        save_dir=args.save_path,
        filename='final_epoch_results'
    )
        
    
    # =====================
    # Load BEST checkpoint
    # =====================
    model.to("cpu")
    torch.cuda.empty_cache()
    best_ckpt = torch.load(f"{args.save_path}/best.pt", map_location='cpu')
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.to(device)
    logging.info("Loaded best checkpoint. Start testing...")

    # =====================
    # Test
    # =====================

    header = ['class_prediction', 'top2_mean', 'top3_mean', 'top5_mean', 'all_mean_pred', 'top2_alpha05', 'top3_alpha05', 'global softmax mean', 'global softmax top3']

    # logging.info(f"---------Seen-----------")
    # test_acc_seen = evaluate(model, dataloader_test_seen, text_ids, device, temperature=temperature)
    # logging.info(f"---------Seen only-----------")
    # # test_acc_seen_only = evaluate(model, dataloader_test_seen, text_ids_train, device, temperature=temperature)
    # logging.info(f"---------UnSeen-----------")
    # test_acc_unseen = evaluate(model, dataloader_test_unseen, text_ids, device, temperature=temperature)
    logging.info(f"---------Seen-----------")
    test_acc_seen = evaluate(model, dataloader_test, text_ids, device, temperature=temperature)
    # acc_unseen_only = evaluate_only_unseen(ema.ema_model, dataloader_test_unseen, text_ids, device, seen_classes, temperature=temperature)
    results = {
        "seen": [x.item() for x in test_acc_seen],
    }
    logging.info(f"Final Test Accuracy (BEST model) seen: {test_acc_seen}")
    logging.info(f"---------UnSeen ONLY -----------")
    # test_acc_unseen_only = evaluate_only_unseen(model, dataloader_test_unseen, text_ids, device, seen_classes, temperature=temperature)
    
    results = {
        "seen": [x.item() for x in test_acc_seen],
    }
    log_result_table(
        header,
        results,
        title="Final Test Accuracy Summary",
        save_dir=args.save_path
    )


def loss_concept_ours(img_feat, concept_feat, visible_mask, device, tau_cpt=0.1, neg_param=1.0):
    valid_mask = visible_mask.sum(1) > 0
    B, D = img_feat.shape
    _, C, D_ = concept_feat.shape
    tokens = concept_feat.reshape(B * C, D)
    S = img_feat @ tokens.T / tau_cpt  # (B, B*C)

    idx_per_class = torch.arange(B * C, device=device).reshape(B, C)
    visible_mask = visible_mask.to(device)

    target_mask = torch.zeros(B, B*C, device=device).scatter_(1, idx_per_class, 1).bool()
    pos_mask = torch.zeros(B, B*C, device=device).scatter_(1, idx_per_class, visible_mask.float()).bool()

    s_target = S[target_mask].reshape(B, C)
    weight = (target_mask != pos_mask) * neg_param + (target_mask == pos_mask) * 1
    neg_logsumexp = torch.logsumexp(S.masked_fill(pos_mask, -float('inf')) * weight, dim=1)

    eps = 1e-7
    ctr = []
    for i in range(C):
        log_num = s_target[:, i]
        log_denom = torch.logsumexp(torch.stack([s_target[:, i], neg_logsumexp]), dim=0)
        val = (log_num - log_denom) * visible_mask[:, i]
        ctr.append(val)
    ctr = torch.stack(ctr, dim=1)
    ctr = -ctr[valid_mask].sum(1) / (visible_mask.sum(-1)[valid_mask] + eps)
    return ctr.mean()

def loss_concept_ours_pos(img_feat, concept_feat, visible_mask, device, tau_cpt=0.1):
    # excludes invisible concepts from denominator; does not add them as negatives
    valid_mask = visible_mask.sum(1) > 0
    B, D = img_feat.shape
    _, C, D_ = concept_feat.shape
    tokens = concept_feat.reshape(B * C, D)
    S = img_feat @ tokens.T / tau_cpt  # (B, B*C)

    idx_per_class = torch.arange(B * C, device=device).reshape(B, C)
    visible_mask = visible_mask.to(device)

    target_mask = torch.zeros(B, B*C, device=device).scatter_(1, idx_per_class, 1).bool()

    s_target = S[target_mask].reshape(B, C)
    self_mask = torch.zeros(B, B*C, device=device)\
        .scatter_(1, idx_per_class, 1).bool()

    visible_flat = visible_mask.reshape(B * C)
    visible_global = visible_flat.unsqueeze(0).expand(B, -1)

    denom_exclude_mask = self_mask | (~visible_global)

    neg_logsumexp = torch.logsumexp(
        S.masked_fill(denom_exclude_mask, -float('inf')),
        dim=1
    )

    eps = 1e-7
    ctr = []
    for i in range(C):
        log_num = s_target[:, i]
        log_denom = torch.logsumexp(torch.stack([s_target[:, i], neg_logsumexp]), dim=0)
        val = (log_num - log_denom) * visible_mask[:, i]
        ctr.append(val)
    ctr = torch.stack(ctr, dim=1)

    ctr = -ctr[valid_mask].sum(1) / (visible_mask.sum(-1)[valid_mask] + eps)
    return ctr.mean()

loss_bcel = nn.BCEWithLogitsLoss()
def loss_bce(image_feat, text_feat, visible_mask, device):
    bsz = image_feat.size(0)
    batch_loss = 0
    for i in range(bsz):
        S = (image_feat[i] @ text_feat[i].T) / 0.07
        batch_loss += loss_bcel(S, (visible_mask[i]*1).to(dtype=S.dtype).to(device))
    return batch_loss / bsz

def compute_l2sp_loss(model, theta_0):
    loss = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        loss += torch.sum((param - theta_0[name]) ** 2)
    return loss

def loss_concept_ours_t2i(img_feat, concept_feat, visible_mask, device, tau_cpt=0.1, neg_param=1.0):
    valid_mask = visible_mask.sum(1) > 0

    B, D = img_feat.shape
    _, C, D_ = concept_feat.shape

    tokens = concept_feat.reshape(B * C, D)              # (B*C, D)
    S = tokens @ img_feat.T / tau_cpt                    # (B*C, B)

    visible_mask = visible_mask.to(device)
    visible_flat = visible_mask.reshape(-1)              # (B*C)

    # target image index for each concept token
    target_img_idx = torch.arange(B, device=device).repeat_interleave(C)  # (B*C)

    # build masks
    idx_per_class = target_img_idx.unsqueeze(1)          # (B*C, 1)

    target_mask = torch.zeros(B*C, B, device=device).scatter_(1, idx_per_class, 1).bool()
    pos_mask = torch.zeros(B*C, B, device=device).scatter_(1, idx_per_class, visible_flat.float().unsqueeze(1)).bool()

    s_target = S[target_mask].reshape(B*C)               # (B*C)

    weight = (target_mask != pos_mask) * neg_param + (target_mask == pos_mask) * 1
    neg_logsumexp = torch.logsumexp(S.masked_fill(pos_mask, -float('inf')) * weight, dim=1)

    eps = 1e-7
    ctr = []

    for i in range(B*C):
        log_num = s_target[i]
        log_denom = torch.logsumexp(torch.stack([log_num, neg_logsumexp[i]]), dim=0)
        val = (log_num - log_denom) * visible_flat[i]
        ctr.append(val)

    ctr = torch.stack(ctr)                               # (B*C)

    # aggregate per image (symmetric to i2t)
    ctr_img = ctr.reshape(B, C)

    ctr_img = -ctr_img[valid_mask].sum(1) / (visible_mask.sum(-1)[valid_mask] + eps)

    return ctr_img.mean()

def loss_concept_ours_soft(
    img_feat,
    concept_feat,
    p_visible,     # ← p(i,c) ∈ [0,1]
    device,
    tau_cpt=0.1,
    lambda_inv=1.0,
    eps=1e-7
):
    B, D = img_feat.shape
    _, C, _ = concept_feat.shape

    tokens = concept_feat.reshape(B * C, D)
    S = img_feat @ tokens.T / tau_cpt   # (B, B*C)

    idx_per_class = torch.arange(B * C, device=device).reshape(B, C)

    # target mask
    target_mask = torch.zeros(B, B*C, device=device)
    target_mask.scatter_(1, idx_per_class, 1)

    # reshape similarity for target class
    s_target = S[target_mask.bool()].reshape(B, C)

    # ----- positive weights -----
    p_visible = p_visible.to(device)     # (B,C)

    # Z_i normalization
    Z = p_visible.sum(1, keepdim=True) + eps

    # ----- inter-class negatives -----
    S_inter = S.masked_fill(target_mask.bool(), -float('inf'))
    inter_logsumexp = torch.logsumexp(S_inter, dim=1)

    # ----- intra-class invisible negatives -----
    inv_weight = (1 - p_visible) * lambda_inv
    intra_term = torch.logsumexp(
        s_target + torch.log(inv_weight + eps), dim=1
    )

    neg_logsumexp = torch.logsumexp(
        torch.stack([inter_logsumexp, intra_term]), dim=0
    )

    # ----- contrastive objective -----
    log_num = s_target
    log_denom = torch.logsumexp(
        torch.stack([s_target, neg_logsumexp[:, None].expand(-1, C)]),
        dim=0
    )

    ctr = p_visible * (log_num - log_denom)

    loss = -(ctr.sum(1) / Z.squeeze()).mean()

    return loss


def get_lambda(step, total_steps, max_lambda=3.0):
    """Linearly ramp lambda up to max_lambda over the first half of training."""
    progress = step / total_steps
    if progress < 0.5:
        return max_lambda * (progress / 0.5)
    return max_lambda


global_steps = 0


def train_on_epoch(args, model, temperature, ema, theta_v0, scheduler, dataloader, concept_ids_train, device, optimizer=None, epoch=None, total_steps=None, pseudo_label=None):
    global global_steps
    num_class, num_concept, _ = concept_ids_train.shape
    num_concept = num_concept - 1
    total_loss_epochs = []
    model.train()
    with torch.enable_grad():
        for batch in dataloader:
            pixel_values = batch[0]
            labels = batch[1]
            paths = batch[2]
            # deduplicate: keep one sample per class in the batch
            new_pixel_values = []
            new_labels = []
            new_paths = []
            seen_class = set()
            for i in range(len(labels)):
                cls_id = labels[i].item()
                if cls_id not in seen_class:
                    seen_class.add(cls_id)
                    new_pixel_values.append(pixel_values[i].unsqueeze(0))
                    new_labels.append(cls_id)
                    new_paths.append(paths[i])
            if len(new_labels) < 4:
                print(len(new_labels))
                continue
            pixel_values = torch.concat(new_pixel_values)
            labels = torch.tensor(new_labels)
            paths = new_paths

            # build visible mask
            if args.use_vis_label:
                visible_mask = []
                for x in paths:
                    mask = pseudo_label[x]
                    if len(mask) != num_concept:
                        mask = [0.0 for _ in range(num_concept)]
                    visible_mask.append(mask)
                visible_mask = torch.tensor(visible_mask)
            else:
                visible_mask = torch.ones((len(labels), num_concept)).to(torch.float)

            with torch.no_grad():
                temperature.clamp_(min=3e-3)
    
            text_feat = model.encode_text(concept_ids_train[labels].view(-1, 77).cuda())
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.view(len(labels), -1, 512) #(bsz, 1+num_concept, dim)

            image_feat = model.encode_image(pixel_values.cuda())
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True) #(bsz, dim)

            stats = {}
            if args.use_concept:
                # concept contrastive loss
                if args.bi_ctr:
                    # print('simple', concept_corr_mask)
                    concept_loss_i2t = loss_concept_ours(image_feat, text_feat[:, 1:], visible_mask, device=device, tau_cpt=temperature, neg_param=args.ctr_neg_weight)
                    concept_loss_t2i = loss_concept_ours_t2i(image_feat, text_feat[:, 1:], visible_mask, device=device, tau_cpt=temperature, neg_param=args.ctr_neg_weight)
                    concept_loss = (concept_loss_i2t + concept_loss_t2i)/2
                    stats['ctr_loss (concept) (i2t+t2i)'] = float(concept_loss.item())
                elif args.ctr_only_pos:
                    concept_loss = loss_concept_ours_pos(image_feat, text_feat[:, 1:], visible_mask, device=device, tau_cpt=temperature)
                    stats['ctr_loss (concept+pos_only)'] = float(concept_loss.item())                  
                else:
                    if args.warm_up_lambda:
                        lambda_inv = get_lambda(global_steps, total_steps, max_lambda=args.ctr_neg_weight)
                    else:
                        lambda_inv = args.ctr_neg_weight
                    concept_loss = loss_concept_ours_soft(image_feat, text_feat[:, 1:], visible_mask, device=device, tau_cpt=temperature, lambda_inv=lambda_inv)
                    stats[f'ctr_loss_soft(concept)_neg:{args.ctr_neg_weight}/ {lambda_inv}'] = float(concept_loss.item())
                
                total_loss = concept_loss
            else:
                # class-level contrastive loss
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_feat @ text_feat[:, 0].t()
                logits_per_text = logit_scale * text_feat[:, 0] @ image_feat.t()
                loss_i2t = F.cross_entropy(logits_per_image, torch.arange(len(labels)).to(device))
                loss_t2i = F.cross_entropy(logits_per_text, torch.arange(len(labels)).to(device))
                batch_ctr = (loss_i2t + loss_t2i) / 2
                total_loss = args.class_ctr_lr * batch_ctr 
                stats['ctr_loss (class)'] = float(total_loss.item())
            
            if args.use_bce:
                vis_bce_loss = loss_bce(image_feat, text_feat[:, 1:], visible_mask, device)
                stats['bce loss (concept)'] = float(vis_bce_loss.item())
                total_loss += args.bce_lr * vis_bce_loss

            # L2-SP regularization
            if args.clip_w:
                loss_l2sp = compute_l2sp_loss(model, theta_v0)
                stats['l2sp loss'] = float(loss_l2sp.item())
                total_loss = total_loss + args.l2sp_lr*loss_l2sp
            stats['lr'] = optimizer.param_groups[0]['lr']

            stats['total_loss'] = float(total_loss.item())
            stats['tau'] = float(temperature.item())

            if global_steps % args.save_freq == 0:
                logging.info(f'total_step:{global_steps}, total_loss:{total_loss.item()}')
                logging.info(stats)
            total_loss_epochs.append(total_loss.item())
            

            total_loss.backward()
            optimizer.step()
            if args.warm_up:
                scheduler.step()
            optimizer.zero_grad()
            ema.update()
            global_steps += 1
    return sum(total_loss_epochs) / len(total_loss_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/user/Develop/datasets')

    parser.add_argument('--main_lr', default=1e-6, type=float)
    parser.add_argument('--proj_lr', default=5e-6, type=float)
    parser.add_argument('--temperature_lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size (train)')
    parser.add_argument('--batch_size_val', default=64, type=int, help='batch_size (val)')
    parser.add_argument('--epochs', default=30, type=int, help='maximum training epochs.')
    parser.add_argument('--save_freq', default=50, type=int, help='saving frequency (steps).')
    parser.add_argument('--dataset', type=str, default='pets', help='select dataset such as pets, cars, ... etc.')
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--use_vis_label', action='store_true')
    parser.add_argument('--use_concept', action='store_true')
    parser.add_argument('--lvlm_output', type=str)
    parser.add_argument('--model_type', type=str, default='vit32')
    parser.add_argument('--lvlm_output_aug', type=str)
    parser.add_argument('--aug_data_path', type=str)
    parser.add_argument('--rich_concept', action='store_true')
    parser.add_argument('--use_bce', action='store_true')
    parser.add_argument('--warm_up', action='store_true')
    parser.add_argument('--clip_w', action='store_true')
    parser.add_argument('--bi_ctr', action='store_true')
    parser.add_argument('--l2sp_lr', default=0.05, type=float)
    parser.add_argument('--bce_lr', default=0.05, type=float)
    parser.add_argument('--class_ctr_lr', default=1.0, type=float)
    parser.add_argument('--ctr_only_pos', action='store_true')
    parser.add_argument('--warm_up_lambda', action='store_true')
    parser.add_argument('--ctr_neg_weight', type=float, default=1.0)
    parser.add_argument('--tau', type=float, help="temperature", default=1.0)
    parser.add_argument('--hard_threshold', type=float, default=0.5, help='threshold for hard labeling in pseudo label')
    parser.add_argument('--hard_label', action='store_true', help='whether to use hard labeling for pseudo label')

    parser.add_argument('--save_path', default="checkpoints_cars/vanilla_ft_test", type=str, help='path to folder saving the checkpoints.')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file log handler
    file_handler = logging.FileHandler(f'{args.save_path}/training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # console log handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    # register handlers with root logger
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

    logging.info(args)
    main(args)