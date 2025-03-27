"""
Generate large-scale dataset for ML training and validation.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os
import sys
import time
import argparse

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.batch_generator import (
    generate_uniform_dataset,
    generate_domain_specific_dataset,
    save_dataset
)
from core_model.domain_profiles import domain_profiles


def generate_dataset_chunk(chunk_id, n_samples, sampling_strategy, domain=None, output_dir=None):
    """
    Generate a chunk of the dataset for parallel processing.
    
    Parameters:
    -----------
    chunk_id : int
        Identifier for this chunk
    
    n_samples : int
        Number of samples in this chunk
    
    sampling_strategy : str
        'uniform' or 'domain_specific'
    
    domain : str, optional
        Domain to use if strategy is 'domain_specific'
    
    output_dir : str, optional
        Directory to save chunk file
    
    Returns:
    --------
    pandas.DataFrame or str
        Dataframe if output_dir is None, otherwise path to saved file
    """
    # Set seed based on chunk_id for reproducibility but different chunks
    random_seed = 42 + chunk_id
    
    # Generate data based on strategy
    if sampling_strategy == 'uniform':
        df = generate_uniform_dataset(n_samples=n_samples, random_seed=random_seed)
    elif sampling_strategy == 'domain_specific':
        if domain is None:
            # Randomly select a domain if none specified
            domain = np.random.choice(list(domain_profiles.keys()))
        df = generate_domain_specific_dataset(domain=domain, n_samples=n_samples, random_seed=random_seed)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    # Add chunk identifier
    df['chunk_id'] = chunk_id
    
    # Save chunk if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"chunk_{chunk_id:04d}.parquet")
        save_dataset(df, filepath)
        return filepath
    else:
        return df


def generate_large_dataset(n_samples=1000000, n_chunks=10, sampling_strategy='uniform', 
                          domain=None, output_dir='data/large_dataset', n_processes=None):
    """
    Generate large-scale dataset using parallel processing.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples to generate
    
    n_chunks : int
        Number of chunks to split data generation into
    
    sampling_strategy : str
        'uniform' or 'domain_specific'
    
    domain : str, optional
        Domain to use if strategy is 'domain_specific'
    
    output_dir : str
        Directory to save dataset chunks
    
    n_processes : int, optional
        Number of processes to use, defaults to number of CPU cores
    
    Returns:
    --------
    str
        Path to final combined dataset file
    """
    start_time = time.time()
    
    # Determine number of processes to use
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Calculate samples per chunk
    samples_per_chunk = n_samples // n_chunks
    
    # Prepare arguments for parallel processing
    chunk_args = [(i, samples_per_chunk, sampling_strategy, domain, output_dir) 
                  for i in range(n_chunks)]
    
    print(f"Generating {n_samples:,} samples in {n_chunks} chunks using {n_processes} processes...")
    print(f"Strategy: {sampling_strategy}, Domain: {domain or 'Multiple'}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate chunks in parallel
    with mp.Pool(processes=n_processes) as pool:
        chunk_files = list(tqdm(
            pool.starmap(generate_dataset_chunk, chunk_args),
            total=n_chunks,
            desc="Generating chunks"
        ))
    
    # Combine chunks into final dataset
    print("\nCombining chunks into final dataset...")
    combined_file = os.path.join(output_dir, "expertise_dataset_combined.parquet")
    
    # Read and combine in batches to reduce memory usage
    combined_df = pd.DataFrame()
    for chunk_file in tqdm(chunk_files, desc="Reading chunks"):
        chunk_df = pd.read_parquet(chunk_file)
        if combined_df.empty:
            combined_df = chunk_df
        else:
            combined_df = pd.concat([combined_df, chunk_df], ignore_index=True)
    
    # Save combined dataset
    save_dataset(combined_df, combined_file)
    
    # Generate metadata
    metadata = {
        "n_samples": len(combined_df),
        "sampling_strategy": sampling_strategy,
        "domain": domain,
        "generated_at": pd.Timestamp.now().isoformat(),
        "generation_time_seconds": time.time() - start_time,
        "mean_expertise": {col: combined_df[col].mean() 
                          for col in combined_df.columns if col.startswith('expertise')},
        "columns": list(combined_df.columns),
        "parameter_ranges": {
            "harm": [combined_df['harm'].min(), combined_df['harm'].max()],
            "complexity": [combined_df['complexity'].min(), combined_df['complexity'].max()],
            "error_rate": [combined_df['error_rate'].min(), combined_df['error_rate'].max()]
        }
    }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {len(combined_df):,}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Dataset saved to: {combined_file}")
    print(f"Metadata saved to: {metadata_file}")
    
    return combined_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate large-scale dataset for ML training")
    parser.add_argument("--samples", type=int, default=1000000, help="Total number of samples to generate")
    parser.add_argument("--chunks", type=int, default=10, help="Number of chunks for parallel processing")
    parser.add_argument("--strategy", type=str, default="uniform", choices=["uniform", "domain_specific"],
                       help="Sampling strategy")
    parser.add_argument("--domain", type=str, default=None, help="Domain to use for domain-specific sampling")
    parser.add_argument("--output-dir", type=str, default="data/large_dataset", help="Output directory")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use")
    
    args = parser.parse_args()
    
    # Verify domain if specified
    if args.domain is not None and args.domain not in domain_profiles:
        available_domains = ", ".join(domain_profiles.keys())
        print(f"Error: Domain '{args.domain}' not found. Available domains: {available_domains}")
        sys.exit(1)
    
    # Generate dataset
    generate_large_dataset(
        n_samples=args.samples,
        n_chunks=args.chunks,
        sampling_strategy=args.strategy,
        domain=args.domain,
        output_dir=args.output_dir,
        n_processes=args.processes
    )