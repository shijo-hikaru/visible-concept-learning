import argparse
import pickle
import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from torch.utils.data import ConcatDataset
from utils.dataset import Flowers102WithFilename, PetWithFilename, StanfordCarsWithFilename, CUBWithFilename


def parse_args():
    parser = argparse.ArgumentParser(description="LLMDet inference script")
    parser.add_argument("--dataset", required=True, choices=["cub", "pet", "flower", "cars"],
                        help="Dataset to run inference on")
    parser.add_argument("--data_dir", default="/home/user/Develop/datasets",
                        help="Root directory of datasets")
    parser.add_argument("--split", default="train", choices=["train", "test", "trainval", "all"],
                        help="Dataset split (pet: test/trainval, others: train)")
    parser.add_argument("--output", default=None,
                        help="Output pkl path (default: output_llmdet_{dataset}.pkl)")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Detection score threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2,
                        help="Text score threshold")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save interval (number of images)")
    parser.add_argument("--descriptor_dir", default="../descriptors",
                        help="Directory containing descriptor JSON files")
    return parser.parse_args()


def build_dataset(args):
    if args.dataset == "cub":
        dataset = CUBWithFilename(
            root=os.path.join(args.data_dir, "cub/CUB_200_2011/images/train"),
            transform=None, mode="all"
        )
        text_labels = ["head", "eye", "beak", "neck", "breast", "belly", "back", "wing", "leg", "tail"]
        idx_to_class = None
        descriptors = None

    elif args.dataset == "pet":
        split = args.split if args.split in ("test", "trainval") else "trainval"
        dataset = PetWithFilename(
            root=os.path.join(args.data_dir, "oxford_pet"),
            split=split, transform=None, mode="all"
        )
        with open(os.path.join(args.descriptor_dir, "pet_concepts.json")) as f:
            descriptors = json.load(f)
        text_labels = ["ear", "muzzle", "face", "eye", "fur", "coat", "tail", "leg", "body"]
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    elif args.dataset == "flower":
        trainset = Flowers102WithFilename(root=args.data_dir, transform=None, split="train", mode="all")
        valset   = Flowers102WithFilename(root=args.data_dir, transform=None, split="val",   mode="all")
        dataset  = ConcatDataset([trainset, valset])
        with open(os.path.join(args.descriptor_dir, "my_flowers.json")) as f:
            descriptors = json.load(f)
        text_labels = None
        idx_to_class = {i: c for i, c in enumerate(trainset.classes)}

    elif args.dataset == "cars":
        dataset = StanfordCarsWithFilename(
            root=os.path.join(args.data_dir, "StanfordCars"),
            transform=None, split="train", mode="all"
        )
        with open(os.path.join(args.descriptor_dir, "my_cars.json")) as f:
            descriptors = json.load(f)
        text_labels = None
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    return dataset, text_labels, idx_to_class, descriptors


def get_concept(dataset_name, class_id, text_labels, idx_to_class, descriptors):
    if dataset_name in ("cub", "pet"):
        return text_labels
    return descriptors[idx_to_class[class_id]]


def main():
    args = parse_args()

    output_path = args.output or f"output_llmdet_{args.dataset}.pkl"

    model_id = "iSEE-Laboratory/llmdet_large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    dataset, text_labels, idx_to_class, descriptors = build_dataset(args)

    outputs_llmdet = {}
    for idx, batch in tqdm(enumerate(dataset), total=len(dataset)):
        image     = batch[0]
        class_id  = batch[1]
        path = batch[2]
        full_path = batch[3]

        concept = get_concept(args.dataset, class_id, text_labels, idx_to_class, descriptors)

        inputs = processor(
            images=image, text=concept,
            return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=args.threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(image.height, image.width)],
            text_labels=concept,
        )[0]
        results["scores"] = results["scores"].cpu()
        results["boxes"]  = results["boxes"].cpu()
        outputs_llmdet[str(path).lower()] = results

        if idx % args.save_interval == 0:
            with open(output_path, "wb") as f:
                pickle.dump(outputs_llmdet, f)

    with open(output_path, "wb") as f:
        pickle.dump(outputs_llmdet, f)

    print(f"Saved {len(outputs_llmdet)} entries to {output_path}")


if __name__ == "__main__":
    main()
