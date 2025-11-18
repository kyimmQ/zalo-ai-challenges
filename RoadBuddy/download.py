"""
Download pretrained models for simple-tad from HuggingFace.

This script downloads pretrained models from the tue-mps/simple-tad HuggingFace repository.
Models are organized into three categories:
- DAPT (Domain-Adaptive Pre-Training) models
- Fine-tuned models on DoTA dataset
- Fine-tuned models on DADA-2000 dataset
"""

import os
import argparse
import urllib.request
import sys


HUGGINGFACE_BASE_URL = "https://huggingface.co/tue-mps/simple-tad/resolve/main/models"

MODELS = {
    "dapt": {
        "dapt-k700_vm-s": "DAPT/simpletad_dapt-k700_videomae-s_ep12.pth",
        "dapt-onlybdd_vm-s": "DAPT/simpletad_dapt-onlybdd_videomae-s_ep12.pth",
        "dapt_vm-s": "DAPT/simpletad_dapt_videomae-s_ep12.pth",
        "dapt_vm-b": "DAPT/simpletad_dapt_videomae-b_ep12.pth",
        "dapt_vm-l": "DAPT/simpletad_dapt_videomae-l_ep12.pth",
    },
    "dota": {
        "ft-dota_vm1-s_auroc": "Finetune_DoTA/simpletad_ft-dota_vm1-s_auroc.pth",
        "ft-dota_vm1-s_aumcc": "Finetune_DoTA/simpletad_ft-dota_vm1-s_aumcc.pth",
        "ft-dota_vm1-b-1600_auroc": "Finetune_DoTA/simpletad_ft-dota_vm1-b-1600_auroc.pth",
        "ft-dota_vm1-b-1600_aumcc": "Finetune_DoTA/simpletad_ft-dota_vm1-b-1600_aumcc.pth",
        "ft-dota_vm1-l_auroc": "Finetune_DoTA/simpletad_ft-dota_vm1-l_auroc.pth",
        "ft-dota_vm1-l_aumcc": "Finetune_DoTA/simpletad_ft-dota_vm1-l_aumcc.pth",
        "ft-dota_vm2-s_auroc": "Finetune_DoTA/simpletad_ft-dota_vm2-s_auroc.pth",
        "ft-dota_vm2-s_aumcc": "Finetune_DoTA/simpletad_ft-dota_vm2-s_aumcc.pth",
        "ft-dota_vm2-b_auroc": "Finetune_DoTA/simpletad_ft-dota_vm2-b_auroc.pth",
        "ft-dota_vm2-b_aumcc": "Finetune_DoTA/simpletad_ft-dota_vm2-b_aumcc.pth",
        "ft-dota_mvd-s-fromL_auroc": "Finetune_DoTA/simpletad_ft-dota_mvd-s-fromL_auroc.pth",
        "ft-dota_mvd-s-fromL_aumcc": "Finetune_DoTA/simpletad_ft-dota_mvd-s-fromL_aumcc.pth",
        "ft-dota_mvd-b-fromB_auroc": "Finetune_DoTA/simpletad_ft-dota_mvd-b-fromB_auroc.pth",
        "ft-dota_mvd-b-fromB_aumcc": "Finetune_DoTA/simpletad_ft-dota_mvd-b-fromB_aumcc.pth",
        "ft-dota_mvd-l-fromL_auroc": "Finetune_DoTA/simpletad_ft-dota_mvd-l-fromL_auroc.pth",
        "ft-dota_mvd-l-fromL_aumcc": "Finetune_DoTA/simpletad_ft-dota_mvd-l-fromL_aumcc.pth",
        "ft-dota_dapt-vm1-s_auroc": "Finetune_DoTA/simpletad_ft-dota_dapt-vm1-s_auroc.pth",
        "ft-dota_dapt-vm1-s_aumcc": "Finetune_DoTA/simpletad_ft-dota_dapt-vm1-s_aumcc.pth",
        "ft-dota_dapt-vm1-b_auroc": "Finetune_DoTA/simpletad_ft-dota_dapt-vm1-b_auroc.pth",
        "ft-dota_dapt-vm1-b_aumcc": "Finetune_DoTA/simpletad_ft-dota_dapt-vm1-b_aumcc.pth",
        "ft-dota_dapt-vm1-l_auroc": "Finetune_DoTA/simpletad_ft-dota_dapt-vm1-l_auroc.pth",
        "ft-dota_dapt-vm1-l_aumcc": "Finetune_DoTA/simpletad_ft-dota_dapt-vm1-l_aumcc.pth",
    },
    "dada": {
        "ft-dada_vm1-s_auroc": "Finetune_D2K/simpletad_ft-dada_vm1-s_auroc.pth",
        "ft-dada_vm1-s_aumcc": "Finetune_D2K/simpletad_ft-dada_vm1-s_aumcc.pth",
        "ft-dada_vm1-b-1600_auroc": "Finetune_D2K/simpletad_ft-dada_vm1-b-1600_auroc.pth",
        "ft-dada_vm1-b-1600_aumcc": "Finetune_D2K/simpletad_ft-dada_vm1-b-1600_aumcc.pth",
        "ft-dada_vm1-l_auroc": "Finetune_D2K/simpletad_ft-dada_vm1-l_auroc.pth",
        "ft-dada_vm1-l_aumcc": "Finetune_D2K/simpletad_ft-dada_vm1-l_aumcc.pth",
        "ft-dada_vm2-s_auroc": "Finetune_D2K/simpletad_ft-dada_vm2-s_auroc.pth",
        "ft-dada_vm2-s_aumcc": "Finetune_D2K/simpletad_ft-dada_vm2-s_aumcc.pth",
        "ft-dada_vm2-b_auroc": "Finetune_D2K/simpletad_ft-dada_vm2-b_auroc.pth",
        "ft-dada_vm2-b_aumcc": "Finetune_D2K/simpletad_ft-dada_vm2-b_aumcc.pth",
        "ft-dada_mvd-s-fromL_auroc": "Finetune_D2K/simpletad_ft-dada_mvd-s-fromL_auroc.pth",
        "ft-dada_mvd-s-fromL_aumcc": "Finetune_D2K/simpletad_ft-dada_mvd-s-fromL_aumcc.pth",
        "ft-dada_mvd-b-fromB_auroc": "Finetune_D2K/simpletad_ft-dada_mvd-b-fromB_auroc.pth",
        "ft-dada_mvd-b-fromB_aumcc": "Finetune_D2K/simpletad_ft-dada_mvd-b-fromB_aumcc.pth",
        "ft-dada_mvd-l-fromL_auroc": "Finetune_D2K/simpletad_ft-dada_mvd-l-fromL_auroc.pth",
        "ft-dada_mvd-l-fromL_aumcc": "Finetune_D2K/simpletad_ft-dada_mvd-l-fromL_aumcc.pth",
        "ft-dada_dapt-vm1-s_auroc": "Finetune_D2K/simpletad_ft-dada_dapt-vm1-s_auroc.pth",
        "ft-dada_dapt-vm1-s_aumcc": "Finetune_D2K/simpletad_ft-dada_dapt-vm1-s_aumcc.pth",
        "ft-dada_dapt-vm1-b_auroc": "Finetune_D2K/simpletad_ft-dada_dapt-vm1-b_auroc.pth",
        "ft-dada_dapt-vm1-b_aumcc": "Finetune_D2K/simpletad_ft-dada_dapt-vm1-b_aumcc.pth",
        "ft-dada_dapt-vm1-l_auroc": "Finetune_D2K/simpletad_ft-dada_dapt-vm1-l_auroc.pth",
        "ft-dada_dapt-vm1-l_aumcc": "Finetune_D2K/simpletad_ft-dada_dapt-vm1-l_aumcc.pth",
    },
}


def download_file(url, output_path):
    """Download a file from URL with progress bar."""
    try:
        print(f"Downloading from {url}")
        print(f"Saving to {output_path}")
        
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                sys.stdout.write(f"\rProgress: {percent}% [{count * block_size}/{total_size} bytes]")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        print("\nDownload completed!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False


def list_models():
    """Print all available models."""
    print("\n=== Available Models ===\n")
    
    print("DAPT Models (Domain-Adaptive Pre-Training):")
    for name in MODELS["dapt"].keys():
        print(f"  - {name}")
    
    print("\nFine-tuned Models on DoTA:")
    for name in MODELS["dota"].keys():
        print(f"  - {name}")
    
    print("\nFine-tuned Models on DADA-2000:")
    for name in MODELS["dada"].keys():
        print(f"  - {name}")
    print()


def download_model(model_name, output_dir="simple-tad/pretrained"):
    """Download a specific model."""
    model_path = None
    
    for category in MODELS.values():
        if model_name in category:
            model_path = category[model_name]
            break
    
    if model_path is None:
        print(f"Error: Model '{model_name}' not found.")
        print("Use --list to see all available models.")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    url = f"{HUGGINGFACE_BASE_URL}/{model_path}"
    filename = os.path.basename(model_path)
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return True
    
    return download_file(url, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained models for simple-tad from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python download.py --list
  
  # Download a specific DAPT model
  python download.py --model dapt_vm-s
  
  # Download a fine-tuned model for DoTA
  python download.py --model ft-dota_dapt-vm1-l_auroc
  
  # Download to a custom directory
  python download.py --model ft-dota_vm1-b-1600_aumcc --output models/checkpoints
  
  # Download all DAPT models
  python download.py --category dapt
  
  # Download all DoTA models
  python download.py --category dota
  
  # Download ALL models (52 total)
  python download.py --all
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model to download"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["dapt", "dota", "dada"],
        help="Download all models in a category (dapt, dota, or dada)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models (all categories)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="simple-tad/pretrained",
        help="Output directory for downloaded models (default: simple-tad/pretrained)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if args.all:
        print("\nDownloading ALL models (52 models total)\n")
        total_models = sum(len(models) for models in MODELS.values())
        current = 0
        
        for category_name, category_models in MODELS.items():
            print(f"\n{'='*60}")
            print(f"Category: {category_name.upper()}")
            print(f"{'='*60}\n")
            
            for model_name in category_models.keys():
                current += 1
                print(f"\n[{current}/{total_models}] Downloading {model_name}")
                success = download_model(model_name, args.output)
                if not success:
                    print(f"Failed to download {model_name}")
        
        print(f"\n{'='*60}")
        print(f"Finished downloading all models!")
        print(f"{'='*60}\n")
    
    elif args.category:
        print(f"\nDownloading all models in category: {args.category}\n")
        models_to_download = list(MODELS[args.category].keys())
        
        for i, model_name in enumerate(models_to_download, 1):
            print(f"\n[{i}/{len(models_to_download)}] Downloading {model_name}")
            success = download_model(model_name, args.output)
            if not success:
                print(f"Failed to download {model_name}")
        
        print(f"\nFinished downloading {args.category} models.")
    
    elif args.model:
        success = download_model(args.model, args.output)
        if not success:
            sys.exit(1)
    
    else:
        print("Error: Please specify --list, --model, --category, or --all")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()