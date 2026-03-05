"""
Test 3: Overall Accuracy Test on Custom Dataset
Tests models on the 'custom' part of the dataset (rows after the empty line separator)
and creates a single diagram for overall answer accuracy.
"""
import os
import time
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common import (
    load_questions,
    run_model_on_questions,
    stop_model,
    SYSTEM_MESSAGE,
    DEFAULT_MODELS
)
from benchmark.test3.report3 import build_report


def run_test(models=None, questions_csv='questions.csv'):
    """
    Runs Test 3: Overall accuracy benchmark on the 'custom' part of the dataset.
    
    Args:
        models: List of model names to test (uses DEFAULT_MODELS if None)
        questions_csv: Path to the questions CSV file
    
    Returns:
        Path to the results directory
    """
    if models is None:
        models = DEFAULT_MODELS
    
    all_results = []
    
    # Load and validate input CSV once (shared across all models)
    # Use only the 'custom' part of the dataset (rows after the empty line separator)
    df_full = load_questions(questions_csv, dataset_part='custom')
    
    if len(df_full) == 0:
        print("No records found in custom dataset. Test 3 cannot run.")
        return None
    
    # Create results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    
    test_dir = f'results/test3-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(test_dir)
    
    print(f"\n=== TEST 3: Overall Accuracy Test (Custom Dataset) ===")
    print(f"Testing {len(models)} models on {len(df_full)} questions")
    print(f"Results will be saved to: {test_dir}\n")
    
    for m in models:
        # Run benchmark for the current model
        model_results = run_model_on_questions(m, df_full, SYSTEM_MESSAGE)
        
        # Save individual model results to CSV
        model_df = pd.DataFrame(model_results)
        model_df.to_csv(f'{test_dir}/{m.replace(":", "_")}_results.csv', index=False)
        
        all_results.extend(model_results)
        
        # Forceful resource cleanup
        stop_model(m)
        time.sleep(3)  # Short cooldown for hardware stabilization
    
    # Build the report
    build_report(all_results, test_dir)
    
    print(f"\n=== TEST 3 COMPLETE ===")
    print(f"Results saved to: {test_dir}")
    
    return test_dir


if __name__ == '__main__':
    run_test()
