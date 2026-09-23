"""
Hugging Face One-Click Upload Script for AgentJev-0.6B
=====================================================
Usage:
    python upload_to_hf.py --token hf_xxxx --repo your-username/agent-jev
"""

import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload AgentJev-0.6B to Hugging Face")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Write Token (starts with hf_)")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face Repo ID (e.g. username/agent-jev)")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    print(f"1. Checking repository: https://huggingface.co/{args.repo}...")
    try:
        info = api.repo_info(repo_id=args.repo)
        print(f"✓ Repository found: {info.id}")
    except Exception as e:
        print(f"Error checking repo: {e}")
        sys.exit(1)

    # Path to export directory
    export_dir = Path("/root/agentjev/export_hf")
    if not export_dir.exists():
        export_dir = Path(__file__).parent / "export_hf"

    if not export_dir.exists():
        print(f"Error: Export directory {export_dir} not found!")
        sys.exit(1)

    print(f"2. Uploading files from {export_dir} to https://huggingface.co/{args.repo}...")
    try:
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=args.repo,
            repo_type="model",
            commit_message="feat: initial release of AgentJev-0.6B model weights, config and card"
        )
        print("\n" + "="*65)
        print(f"🎉 成功上传到 Hugging Face！")
        print(f"🔗 模型地址: https://huggingface.co/{args.repo}")
        print("="*65)
    except Exception as e:
        print(f"\nUpload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
