"""
Test 2: Category-based Accuracy Test
Groups questions by Category and creates a diagram showing accuracy per category for each model.
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
from benchmark.test2.report2 import build_report


def run_test(models=None, questions_csv='questions.csv', questions_per_category=None):
    """
    Runs Test 2: Category-based accuracy benchmark.
    
    Args:
        models: List of model names to test (uses DEFAULT_MODELS if None)
        questions_csv: Path to the questions CSV file
        questions_per_category: Number of questions to sample per category (None = all questions)
    
    Returns:
        Path to the results directory
    """
    if models is None:
        models = DEFAULT_MODELS
    
    all_results = []
    
    # Load and validate input CSV once (shared across all models)
    df_full = load_questions(questions_csv)

    # Drop rows where no category is specified
    df_full = df_full.dropna(subset=['Time period'])
    
    # Sample up to N questions per category
    if questions_per_category is None:
        df_sampled = df_full
    else:
        df_sampled = df_full.groupby('Time period').head(questions_per_category).reset_index(drop=True)
    
    # Create results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    
    test_dir = f'results/test2-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(test_dir)
    
    unique_categories = df_sampled['Time period'].nunique()
    
    print(f"\n=== TEST 2: Category-based Accuracy Test ===")
    print(f"Testing {len(models)} models")
    print(f"Questions: {len(df_sampled)} ({'' if questions_per_category is None else f'up to {questions_per_category} per category'})")
    print(f"Unique categories: {unique_categories}")
    print(f"Results will be saved to: {test_dir}\n")
    
    for m in models:
        # Run benchmark for the current model
        model_results = run_model_on_questions(m, df_sampled, SYSTEM_MESSAGE)
        
        # Save individual model results to CSV
        model_df = pd.DataFrame(model_results)
        model_df.to_csv(f'{test_dir}/{m.replace(":", "_")}_results.csv', index=False)
        
        all_results.extend(model_results)
        
        # Forceful resource cleanup
        stop_model(m)
        time.sleep(3)  # Short cooldown for hardware stabilization
    
    # Build the report
    build_report(all_results, test_dir)
    
    print(f"\n=== TEST 2 COMPLETE ===")
    print(f"Results saved to: {test_dir}")
    
    return test_dir


if __name__ == '__main__':
    run_test()
