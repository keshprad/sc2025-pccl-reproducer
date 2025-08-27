#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 Benchmark for GPT-style Transformer Models

Experiment Design:
- 7B and 13B parameter GPT-style transformer
- Global batch size: 4 million tokens  
- Sequence length: 2048
- Open Web Text dataset
- 10 training batches across 3 trials
- Average throughput over last 8 batches (skip first 2 for warmup)
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
import numpy as np
import tiktoken
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for the DeepSpeed ZeRO-3 experiment"""
    model_size: str  # "7B" or "13B"
    # target_tokens: int = 4_000_000  # Target 4 million tokens (will be adjusted)
    target_tokens: int = 100_000  # Target 4 million tokens (will be adjusted)
    seq_len: int = 2048
    num_batches: int = 10
    warmup_batches: int = 2  # Skip first 2 batches for averaging
    dataset_name: str = "openwebtext"
    max_position_embeddings: int = 2048
    vocab_size: int = 50257  # GPT-2 vocab size
    
    # These will be set after initialization
    world_size: int = None
    global_batch_size: int = None  # Adjusted global batch size (number of sequences)
    global_batch_size_tokens: int = None  # Adjusted token count
    local_batch_size: int = None
    
    def setup_batch_sizes(self, world_size: int):
        """Calculate and adjust batch sizes for proper divisibility"""
        self.world_size = world_size
        
        # Calculate raw global batch size from target tokens
        raw_global_batch = self.target_tokens // self.seq_len
        
        # Adjust global batch size to be divisible by world_size
        self.global_batch_size = (raw_global_batch // world_size) * world_size
        self.local_batch_size = self.global_batch_size // world_size
        
        # Ensure local batch size is at least 1
        if self.local_batch_size == 0:
            self.local_batch_size = 1
            self.global_batch_size = world_size
        
        # Update the global token count to match adjusted batch size
        self.global_batch_size_tokens = self.global_batch_size * self.seq_len
    
    @property 
    def model_config(self) -> Dict:
        """Get model configuration based on size"""
        configs = {
            "1B": {
                "n_embd": 2048,
                "n_layer": 24,
                "n_head": 16,
                "n_positions": self.max_position_embeddings,
                "vocab_size": self.vocab_size,
            },
            "7B": {
                "n_embd": 4096,
                "n_layer": 32,
                "n_head": 32,
                "n_positions": self.max_position_embeddings,
                "vocab_size": self.vocab_size,
            },
            "13B": {
                "n_embd": 5120,
                "n_layer": 40, 
                "n_head": 40,
                "n_positions": self.max_position_embeddings,
                "vocab_size": self.vocab_size,
            }
        }
        return configs[self.model_size]


class OpenWebTextDataset(torch.utils.data.Dataset):
    """OpenWebText dataset using pre-tokenized binary files"""
    
    def __init__(self, data_dir: str, split: str = "train", seq_len: int = 2048):
        self.seq_len = seq_len
        self.split = split
        
        # Load pre-tokenized binary data
        data_file = os.path.join(data_dir, f'{split}.bin')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Pre-tokenized data file not found: {data_file}")
            
        logger.info(f"Loading pre-tokenized data from {data_file}")
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        logger.info(f"Loaded {len(self.data):,} tokens from {split} split")
        
        # Calculate number of samples (non-overlapping sequences)
        self.num_samples = len(self.data) // seq_len
        logger.info(f"Dataset contains {self.num_samples:,} samples of length {seq_len}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get a sequence of seq_len tokens
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # Extract the sequence
        tokens = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
        
        return {
            'input_ids': tokens[:-1],  # All but last token
            'labels': tokens[1:],      # All but first token (shifted for next-token prediction)
            'attention_mask': torch.ones_like(tokens[:-1])
        }


def create_model(config: ExperimentConfig) -> nn.Module:
    """Create GPT-2 model with specified configuration"""
    model_config = GPT2Config(**config.model_config)
    model = GPT2LMHeadModel(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created {config.model_size} model with {total_params:,} parameters")
    
    return model


def create_deepspeed_config(experiment_config: ExperimentConfig, offload_params: bool = False) -> Dict:
    """Create DeepSpeed configuration for ZeRO-3"""
    config = {
        "train_batch_size": experiment_config.global_batch_size,
        "train_micro_batch_size_per_gpu": experiment_config.local_batch_size,
        "gradient_accumulation_steps": experiment_config.global_batch_size // (experiment_config.local_batch_size * experiment_config.world_size),
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 100
            }
        },
        
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "zero_optimization": {
            "stage": 3,
            # Try without optimizer offloading first
            "offload_optimizer": {
                "device": "cpu", 
                "pin_memory": True
            },
            # Try without parameter offloading first
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "allgather_bucket_size": "auto", 
            "reduce_bucket_size": "auto", 
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": False
        },
        
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        
        "wall_clock_breakdown": False,
        "steps_per_print": 1,
    }
    
    return config


def run_benchmark(model, dataloader, config: ExperimentConfig) -> Dict[str, float]:
    """Run a single benchmark trial"""
    
    logger.info("Running benchmark trial...")
    
    model.train()
    batch_times = []
    throughput_samples = []
    throughput_tokens = []
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= config.num_batches:
            break
            
        batch_start = time.time()
        
        # Move batch to device
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda() 
        attention_mask = batch['attention_mask'].cuda()
        
        # Forward pass
        outputs = model(input_ids=input_ids, 
                       attention_mask=attention_mask,
                       labels=labels)
        loss = outputs.loss
        
        # Backward pass
        model.backward(loss)
        model.step()
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        
        # Calculate throughput
        batch_size = input_ids.shape[0]
        tokens_per_batch = batch_size * config.seq_len
        
        samples_per_sec = batch_size / batch_time
        tokens_per_sec = tokens_per_batch / batch_time
        
        batch_times.append(batch_time)
        throughput_samples.append(samples_per_sec)
        throughput_tokens.append(tokens_per_sec)
        
        logger.info(f"Batch {batch_idx + 1}/{config.num_batches}: "
                   f"Loss={loss.item():.4f}, Time={batch_time:.2f}s, "
                   f"Throughput={samples_per_sec:.2f} samples/s, "
                   f"{tokens_per_sec:.2f} tokens/s")
    
    total_time = time.time() - start_time
    
    # Calculate averages over last 8 batches (excluding warmup)
    start_idx = config.warmup_batches
    avg_batch_time = sum(batch_times[start_idx:]) / len(batch_times[start_idx:])
    avg_samples_throughput = sum(throughput_samples[start_idx:]) / len(throughput_samples[start_idx:])
    avg_tokens_throughput = sum(throughput_tokens[start_idx:]) / len(throughput_tokens[start_idx:])
    
    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'avg_samples_throughput': avg_samples_throughput,
        'avg_tokens_throughput': avg_tokens_throughput,
        'all_batch_times': batch_times,
        'all_samples_throughput': throughput_samples,
        'all_tokens_throughput': throughput_tokens
    }


def main():
    parser = argparse.ArgumentParser(description='DeepSpeed ZeRO-3 Benchmark')
    parser.add_argument('--model_size', type=str, choices=['1B', '7B', '13B'], 
                       required=True, help='Model size to benchmark')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing pre-tokenized binary files (train.bin, val.bin)')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Directory to save results')
    parser.add_argument('--offload_params', action='store_true',
                       help='Enable parameter offloading to CPU (slower but saves GPU memory)')
    
    args = parser.parse_args()
    
    # Initialize distributed training
    deepspeed.init_distributed()
    
    # Get distributed info and check GPU
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    # Check GPU availability and setup device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            logger.info(f"Using GPU {local_rank} - {torch.cuda.get_device_name(local_rank)}")
            logger.info(f"Total GPUs: {torch.cuda.device_count()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB per GPU")
    else:
        logger.error(f"Rank {rank}: CUDA not available!")
        return
    
    # Setup experiment configuration
    config = ExperimentConfig(model_size=args.model_size)
    
    # Setup batch sizes for distributed training
    config.setup_batch_sizes(world_size)
    
    if torch.distributed.get_rank() == 0:
        logger.info(f"Experiment Configuration:")
        logger.info(f"  Model size: {config.model_size}")
        logger.info(f"  Global batch size: {config.global_batch_size} samples ({config.global_batch_size_tokens:,} tokens)")
        logger.info(f"  Local batch size: {config.local_batch_size}")
        logger.info(f"  Sequence length: {config.seq_len}")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Batches: {config.num_batches}")
        logger.info(f"  Warmup batches: {config.warmup_batches}")
        logger.info(f"  Data directory: {args.data_dir}")
    
    # Create model
    model = create_model(config)
    
    # Create dataset and dataloader using pre-tokenized data
    dataset = OpenWebTextDataset(
        data_dir=args.data_dir,
        split="train", 
        seq_len=config.seq_len
    )
    
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config.local_batch_size, 
                          sampler=sampler, num_workers=2, pin_memory=True)
    
    # Initialize DeepSpeed
    deepspeed_config = create_deepspeed_config(config)
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=deepspeed_config,
        dist_init_required=False
    )
    
    # Run single benchmark trial
    logger.info("Starting benchmark run...")
    trial_results = run_benchmark(model_engine, dataloader, config)
    
    if torch.distributed.get_rank() == 0:
        logger.info("Benchmark completed - "
                   f"Avg throughput: {trial_results['avg_samples_throughput']:.2f} samples/s, "
                   f"{trial_results['avg_tokens_throughput']:.2f} tokens/s")
        
        # Prepare final results with single trial
        final_results = {
            'model_size': config.model_size,
            'global_batch_size': config.global_batch_size,
            'global_batch_size_tokens': config.global_batch_size_tokens,
            'seq_len': config.seq_len,
            'world_size': world_size,
            'local_batch_size': config.local_batch_size,
            'num_batches': config.num_batches,
            'warmup_batches': config.warmup_batches,
            'avg_samples_throughput': trial_results['avg_samples_throughput'],
            'avg_tokens_throughput': trial_results['avg_tokens_throughput'],
            'avg_batch_time': trial_results['avg_batch_time'],
            'trial_data': trial_results
        }
        
        # Save results with timestamp to avoid overwrites
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f'deepspeed_zero3_{config.model_size}_{timestamp}_results.json')
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("BENCHMARK RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model: {config.model_size} parameters")
        logger.info(f"Global batch size: {config.global_batch_size:,} samples ({config.global_batch_size_tokens:,} tokens)")
        logger.info(f"Sequence length: {config.seq_len}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Batches: {config.num_batches} (warmup: {config.warmup_batches})")
        logger.info("-" * 40)
        logger.info(f"Throughput: {trial_results['avg_samples_throughput']:.2f} samples/sec")
        logger.info(f"Throughput: {trial_results['avg_tokens_throughput']:.2f} tokens/sec") 
        logger.info(f"Average batch time: {trial_results['avg_batch_time']:.3f} seconds")
        logger.info(f"Results saved to: {results_file}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
