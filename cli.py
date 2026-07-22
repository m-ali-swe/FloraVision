"""
Command Line Interface (CLI) tool for Flower Classification Model.
Supports single image prediction, directory batch processing, and formatted JSON output.
"""

import argparse
import sys
import json
from pathlib import Path

from src.classifier import FlowerClassifier


def main():
    parser = argparse.ArgumentParser(
        description="FloraVision CLI: Classify flower images using trained TensorFlow model."
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--image", type=str, help="Path to single image file for classification."
    )
    group.add_argument(
        "-d", "--dir", type=str, help="Path to directory containing images for batch classification."
    )
    
    parser.add_argument(
        "-k", "--top-k", type=int, default=3, help="Number of top candidates to report (default: 3)."
    )
    parser.add_argument(
        "--json", action="store_true", help="Format output as JSON."
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Save classification results to output JSON file."
    )

    args = parser.parse_args()

    # Load Model
    try:
        classifier = FlowerClassifier()
    except Exception as e:
        print(f"[ERROR] Failed to load flower classifier model: {e}", file=sys.stderr)
        sys.exit(1)

    results = None

    # Single Image Processing
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[ERROR] Image file does not exist: {img_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"[INFO] Classifying single image: {img_path} ...")
        res = classifier.predict_image(str(img_path), top_k=args.top_k)
        results = {"file": str(img_path), "prediction": res}

    # Directory Batch Processing
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"[ERROR] Directory does not exist: {dir_path}", file=sys.stderr)
            sys.exit(1)
        
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [
            f for f in dir_path.iterdir() if f.suffix.lower() in valid_extensions
        ]
        
        if not image_files:
            print(f"[WARNING] No valid image files found in {dir_path}")
            sys.exit(0)
            
        print(f"[INFO] Found {len(image_files)} image files in directory. Running batch inference...")
        batch_results = []
        for img_p in image_files:
            pred = classifier.predict_image(str(img_p), top_k=args.top_k)
            batch_results.append({"file": img_p.name, "predicted_class": pred["predicted_class"], "confidence": pred["confidence_percentage"]})
            
        results = {"directory": str(dir_path), "total_processed": len(image_files), "results": batch_results}

    # Output Formatting
    if args.json or args.output:
        formatted_json = json.dumps(results, indent=2)
        if args.json:
            print(formatted_json)
        if args.output:
            with open(args.output, "w") as f:
                f.write(formatted_json)
            print(f"[INFO] Saved results to {args.output}")
    else:
        # Standard Console Text Output
        print("\n" + "=" * 50)
        print(" FLARAVISION CLASSIFICATION REPORT")
        print("=" * 50)
        if "file" in results:
            pred = results["prediction"]
            print(f" File:       {results['file']}")
            print(f" Class:      {pred['predicted_class'].upper()}")
            print(f" Confidence: {pred['confidence_percentage']}%\n")
            print(" Top Candidate Probabilities:")
            for candidate in pred["top_k"]:
                print(f"   - {candidate['class_name']:<12}: {candidate['percentage']}%")
        elif "directory" in results:
            print(f" Directory: {results['directory']}")
            print(f" Total Processed: {results['total_processed']}\n")
            for item in results["results"]:
                print(f"   - {item['file']:<25} => {item['predicted_class'].title():<12} ({item['confidence']}%)")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
