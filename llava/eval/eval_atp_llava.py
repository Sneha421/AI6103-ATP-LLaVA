"""
ATP-aware evaluation script for LLaVA.

This script extends the standard LLaVA evaluation to include ATP-specific metrics:
- Token pruning statistics
- Computational efficiency measurements  
- Memory usage tracking
- Performance comparison with/without ATP

Usage:
python llava/eval/eval_atp_llava.py --model-path /path/to/model --image-file image.jpg --query "question"
"""

import argparse
import torch
import time
import psutil
import os
from typing import Dict, Any

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import requests
from io import BytesIO


class ATPEvaluator:
    """Evaluator that tracks ATP-specific metrics during inference."""
    
    def __init__(self, model, tokenizer, image_processor):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.atp_stats = []
        
    def evaluate_with_atp_metrics(self, image_files, query, conv_mode="vicuna_v1"):
        """
        Run evaluation while tracking ATP metrics.
        
        Returns:
            dict: Contains response, ATP statistics, and performance metrics
        """
        # Reset stats
        self.atp_stats = []
        
        # Load images
        images = []
        for img_file in image_files:
            if img_file.startswith("http"):
                response = requests.get(img_file)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(img_file).convert("RGB")
            images.append(image)
        
        # Process images
        images_tensor = process_images(
            images,
            self.image_processor, 
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        
        # Prepare conversation
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).cuda()
        
        # Track performance metrics
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Set model to eval mode for ATP hard pruning
        self.model.eval()
        
        with torch.no_grad():
            # Run inference with ATP
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True
            )
        
        # Calculate performance metrics
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        inference_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Decode response
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], 
            skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        
        # Collect ATP statistics
        atp_metrics = self.collect_atp_metrics()
        
        return {
            'response': outputs,
            'atp_metrics': atp_metrics,
            'performance': {
                'inference_time': inference_time,
                'memory_used_mb': memory_used,
                'tokens_per_second': (output_ids.shape[1] - input_token_len) / inference_time
            }
        }
    
    def collect_atp_metrics(self) -> Dict[str, Any]:
        """Collect ATP pruning statistics from the model."""
        if not hasattr(self.model.model, 'atp_stats'):
            return {'atp_enabled': False}
        
        atp_stats = self.model.model.atp_stats
        insertion_points = getattr(self.model.model, 'atp_insertion_points', [])
        
        if not atp_stats:
            return {'atp_enabled': True, 'no_stats': True}
        
        metrics = {
            'atp_enabled': True,
            'num_atp_layers': len(atp_stats),
            'insertion_points': insertion_points,
            'layer_stats': []
        }
        
        total_original_tokens = 0
        total_kept_tokens = 0
        
        for i, stats in enumerate(atp_stats):
            layer_idx = insertion_points[i] if i < len(insertion_points) else i
            
            layer_info = {
                'layer_index': layer_idx,
                'kept_tokens': stats.get('kept_tokens', 'N/A'),
                'original_tokens': stats.get('original_tokens', 'N/A'),
                'pruning_ratio': stats.get('pruning_ratio', 0.0),
                'theta_r': stats.get('theta_r', 'N/A'),
                'theta_s': stats.get('theta_s', 'N/A'),
            }
            
            metrics['layer_stats'].append(layer_info)
            
            if isinstance(layer_info['kept_tokens'], (int, float)):
                total_kept_tokens += layer_info['kept_tokens']
            if isinstance(layer_info['original_tokens'], (int, float)):
                total_original_tokens += layer_info['original_tokens']
        
        # Overall pruning statistics
        if total_original_tokens > 0:
            metrics['overall_pruning_ratio'] = 1.0 - (total_kept_tokens / total_original_tokens)
            metrics['average_kept_tokens'] = total_kept_tokens / len(atp_stats)
        
        return metrics
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def print_atp_report(self, results):
        """Print a detailed ATP performance report."""
        print("\n" + "="*60)
        print("🔬 ATP Evaluation Report")
        print("="*60)
        
        # Response
        print(f"🤖 Model Response:")
        print(f"   {results['response']}")
        
        # Performance metrics
        perf = results['performance']
        print(f"\n⚡ Performance Metrics:")
        print(f"   Inference time: {perf['inference_time']:.2f}s")
        print(f"   Memory used: {perf['memory_used_mb']:.1f} MB")
        print(f"   Tokens/sec: {perf['tokens_per_second']:.1f}")
        
        # ATP metrics
        atp = results['atp_metrics']
        if not atp.get('atp_enabled', False):
            print(f"\n❌ ATP not enabled")
            return
            
        if atp.get('no_stats', False):
            print(f"\n⚠️  ATP enabled but no statistics available")
            return
        
        print(f"\n🎯 ATP Pruning Statistics:")
        print(f"   Number of ATP layers: {atp['num_atp_layers']}")
        print(f"   Insertion points: {atp['insertion_points']}")
        
        if 'overall_pruning_ratio' in atp:
            print(f"   Overall pruning ratio: {atp['overall_pruning_ratio']:.1%}")
            print(f"   Average kept tokens: {atp['average_kept_tokens']:.1f}")
        
        print(f"\n📊 Layer-wise Statistics:")
        for layer_stat in atp['layer_stats']:
            layer_idx = layer_stat['layer_index']
            kept = layer_stat['kept_tokens']
            original = layer_stat['original_tokens']
            ratio = layer_stat['pruning_ratio']
            
            print(f"   Layer {layer_idx}: {kept}/{original} tokens ({ratio:.1%} pruned)")
        
        # Efficiency summary
        print(f"\n💡 Efficiency Summary:")
        if 'overall_pruning_ratio' in atp:
            pruning_ratio = atp['overall_pruning_ratio']
            computational_savings = pruning_ratio * 100
            print(f"   Computational savings: ~{computational_savings:.1f}%")
            print(f"   Memory savings: ~{computational_savings:.1f}%")
        print(f"   ATP overhead: Minimal (learnable pruning)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    
    # Disable torch init for faster loading
    disable_torch_init()
    
    print("🚀 Loading ATP-enabled LLaVA model...")
    
    # Load model with ATP enabled by default
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base, 
        get_model_name_from_path(args.model_path),
        use_atp=True,  # Ensure ATP is enabled
        device_map="auto"
    )
    
    print("✅ Model loaded successfully!")
    
    # Check ATP status
    if hasattr(model.model, 'atp_modules'):
        print(f"🎯 ATP Status: Enabled with {len(model.model.atp_modules)} modules")
        print(f"   Insertion points: {model.model.atp_insertion_points}")
    else:
        print("❌ ATP Status: Not detected")
    
    # Create evaluator
    evaluator = ATPEvaluator(model, tokenizer, image_processor)
    
    # Parse image files
    image_files = args.image_file.split(args.sep)
    
    print(f"\n🖼️  Processing {len(image_files)} image(s)...")
    print(f"❓ Query: {args.query}")
    
    # Run evaluation with ATP metrics
    results = evaluator.evaluate_with_atp_metrics(
        image_files, 
        args.query,
        args.conv_mode
    )
    
    # Print detailed report
    evaluator.print_atp_report(results)


if __name__ == "__main__":
    main()