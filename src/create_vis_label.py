"""
Generate LVLM-based part visibility labels for original training images.

Usage:
    python create_vis_label02.py --dataset cub_auroc --model Qwen/Qwen3-VL-4B-Instruct
    python create_vis_label02.py --dataset pets_auroc --model Qwen/Qwen3-VL-4B-Instruct
"""

import argparse
import json
import os

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

from utils.dataset import PetWithFilename, CUBWithFilename

CLASSNAMES_CUB = ['black footed albatross', 'laysan albatross', 'sooty albatross', 'groove billed ani', 'crested auklet', 'least auklet', 'parakeet auklet', 'rhinoceros auklet', 'brewer blackbird', 'red winged blackbird', 'rusty blackbird', 'yellow headed blackbird', 'bobolink', 'indigo bunting', 'lazuli bunting', 'painted bunting', 'cardinal', 'spotted catbird', 'gray catbird', 'yellow breasted chat', 'eastern towhee', 'chuck will widow', 'brandt cormorant', 'red faced cormorant', 'pelagic cormorant', 'bronzed cowbird', 'shiny cowbird', 'brown creeper', 'american crow', 'fish crow', 'black billed cuckoo', 'mangrove cuckoo', 'yellow billed cuckoo', 'gray crowned rosy finch', 'purple finch', 'northern flicker', 'acadian flycatcher', 'great crested flycatcher', 'least flycatcher', 'olive sided flycatcher', 'scissor tailed flycatcher', 'vermilion flycatcher', 'yellow bellied flycatcher', 'frigatebird', 'northern fulmar', 'gadwall', 'american goldfinch', 'european goldfinch', 'boat tailed grackle', 'eared grebe', 'horned grebe', 'pied billed grebe', 'western grebe', 'blue grosbeak', 'evening grosbeak', 'pine grosbeak', 'rose breasted grosbeak', 'pigeon guillemot', 'california gull', 'glaucous winged gull', 'heermann gull', 'herring gull', 'ivory gull', 'ring billed gull', 'slaty backed gull', 'western gull', 'anna hummingbird', 'ruby throated hummingbird', 'rufous hummingbird', 'green violetear', 'long tailed jaeger', 'pomarine jaeger', 'blue jay', 'florida jay', 'green jay', 'dark eyed junco', 'tropical kingbird', 'gray kingbird', 'belted kingfisher', 'green kingfisher', 'pied kingfisher', 'ringed kingfisher', 'white breasted kingfisher', 'red legged kittiwake', 'horned lark', 'pacific loon', 'mallard', 'western meadowlark', 'hooded merganser', 'red breasted merganser', 'mockingbird', 'nighthawk', 'clark nutcracker', 'white breasted nuthatch', 'baltimore oriole', 'hooded oriole', 'orchard oriole', 'scott oriole', 'ovenbird', 'brown pelican', 'white pelican', 'western wood pewee', 'sayornis', 'american pipit', 'whip poor will', 'horned puffin', 'common raven', 'white necked raven', 'american redstart', 'geococcyx', 'loggerhead shrike', 'great grey shrike', 'baird sparrow', 'black throated sparrow', 'brewer sparrow', 'chipping sparrow', 'clay colored sparrow', 'house sparrow', 'field sparrow', 'fox sparrow', 'grasshopper sparrow', 'harris sparrow', 'henslow sparrow', 'le conte sparrow', 'lincoln sparrow', 'nelson sharp tailed sparrow', 'savannah sparrow', 'seaside sparrow', 'song sparrow', 'tree sparrow', 'vesper sparrow', 'white crowned sparrow', 'white throated sparrow', 'cape glossy starling', 'bank swallow', 'barn swallow', 'cliff swallow', 'tree swallow', 'scarlet tanager', 'summer tanager', 'artic tern', 'black tern', 'caspian tern', 'common tern', 'elegant tern', 'forsters tern', 'least tern', 'green tailed towhee', 'brown thrasher', 'sage thrasher', 'black capped vireo', 'blue headed vireo', 'philadelphia vireo', 'red eyed vireo', 'warbling vireo', 'white eyed vireo', 'yellow throated vireo', 'bay breasted warbler', 'black and white warbler', 'black throated blue warbler', 'blue winged warbler', 'canada warbler', 'cape may warbler', 'cerulean warbler', 'chestnut sided warbler', 'golden winged warbler', 'hooded warbler', 'kentucky warbler', 'magnolia warbler', 'mourning warbler', 'myrtle warbler', 'nashville warbler', 'orange crowned warbler', 'palm warbler', 'pine warbler', 'prairie warbler', 'prothonotary warbler', 'swainson warbler', 'tennessee warbler', 'wilson warbler', 'worm eating warbler', 'yellow warbler', 'northern waterthrush', 'louisiana waterthrush', 'bohemian waxwing', 'cedar waxwing', 'american three toed woodpecker', 'pileated woodpecker', 'red bellied woodpecker', 'red cockaded woodpecker', 'red headed woodpecker', 'downy woodpecker', 'bewick wren', 'cactus wren', 'carolina wren', 'house wren', 'marsh wren', 'rock wren', 'winter wren', 'common yellowthroat']
ID2LABEL_CUB = {i: name for i, name in enumerate(CLASSNAMES_CUB)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['cub_auroc', 'cub', 'pets_auroc'],
                        help='Dataset to process')
    parser.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct',
                        help='Qwen3-VL model name')
    parser.add_argument('--data_dir', default='/home/user/Develop/datasets',
                        help='Root directory of datasets')
    parser.add_argument('--descriptors_dir', default='descriptors',
                        help='Directory containing descriptor JSON files')
    parser.add_argument('--output_dir', default='vis_labels',
                        help='Output directory for vis_label JSONs')
    parser.add_argument('--save_interval', type=int, default=40,
                        help='Save checkpoint every N images')
    return parser.parse_args()


def parse_table(raw: str) -> dict:
    """Parse LVLM markdown table output into {attr: float} dict."""
    result = {}
    for line in raw.strip().split('\n'):
        if not line.startswith('|') or line.startswith('|--') or line.startswith('| --'):
            continue
        cols = [c.strip() for c in line.split('|')[1:3]]
        if len(cols) != 2:
            continue
        attr, val = cols
        if attr in ('Attribute', 'P(visible)', ''):
            continue
        try:
            result[attr] = float(val) >= 0.5
        except ValueError:
            result[attr] = val.lower() == 'yes'
    return result


def create_prompt(concept_list: list) -> str:
    concepts = '\n'.join(f'- {c}' for c in concept_list)
    return f"""You are given an image and a list of possible visual attributes:

{concepts}

Your task is to estimate the probability that each attribute is **visibly present** in the image.

Base your decision strictly on visible pixels.
Do not guess based on prior knowledge about the object class.

### Visibility rules
- If the attribute is clearly visible → probability close to 1.0
- If the attribute is clearly not visible → probability close to 0.0
- If uncertain → intermediate probability.

### Output format (STRICT)

|Attribute|P(visible)|
|---|---|

- P(visible) must be a decimal between **0.0 and 1.0**
- Output **only the table**
"""


def build_dataset(args):
    if args.dataset == 'cub_auroc':
        dataset = CUBWithFilename(
            root=f'{args.data_dir}/cub/CUB_200_2011/images/train',
            transform=None, mode='all',
        )
        with open(f'{args.descriptors_dir}/my_cub2-1.json') as f:
            descriptors = json.load(f)
        idx_to_class = ID2LABEL_CUB

    elif args.dataset == 'cub':
        dataset = CUBWithFilename(
            root=f'{args.data_dir}/cub/CUB_200_2011/images/train',
            transform=None, mode='all',
        )
        with open(f'{args.descriptors_dir}/my_cub_gpt4.1.json') as f:
            descriptors = json.load(f)
        idx_to_class = ID2LABEL_CUB

    elif args.dataset == 'pets_auroc':
        dataset = PetWithFilename(
            root=f'{args.data_dir}/oxford_pet',
            split='trainval', transform=None, mode='all',
        )
        with open(f'{args.descriptors_dir}/pet_concepts.json') as f:
            raw = json.load(f)
        descriptors = {k: list(v.values()) for k, v in raw.items()}
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    return dataset, descriptors, idx_to_class


def main():
    args = parse_args()

    dataset, descriptors, idx_to_class = build_dataset(args)

    model_name_safe = args.model.replace('/', '_').lower()
    output_file = os.path.join(
        args.output_dir,
        f'{args.dataset}_03_concept-based_prob_{model_name_safe}.json',
    )
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with open(output_file) as f:
            output_dict = json.load(f)
        print(f'Resuming from {output_file} ({len(output_dict)} entries)')
    except FileNotFoundError:
        output_dict = {}

    lvlm = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto',
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        image, label_id, path = sample[0], sample[1], sample[2]
        filename = os.path.basename(path)
        if filename in output_dict:
            continue

        classname = idx_to_class[label_id]
        prompt = create_prompt(descriptors[classname])
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': prompt},
        ]}]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt',
        ).to(lvlm.device)

        generated_ids = lvlm.generate(**inputs, max_new_tokens=256)
        trimmed = [out[len(inp):] for inp, out in zip(inputs['input_ids'], generated_ids)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]

        output_dict[filename] = {'table': parse_table(response), 'raw_output': response}

        if idx % args.save_interval == 0:
            with open(output_file, 'w') as f:
                json.dump(output_dict, f, indent=2)

    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f'Saved {len(output_dict)} entries -> {output_file}')


if __name__ == '__main__':
    main()
