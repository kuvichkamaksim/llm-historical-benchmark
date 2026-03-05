"""
Test 3: Category-based Accuracy on Custom Questions Set
Filters records without Time period and groups by Subject field.
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


def run_test(models=None, questions_csv='questions.csv', questions_per_category=None):
    """
    Runs Test 3: Category-based accuracy on custom questions set.
    
    Filters records without Time period and groups by Subject field.
    
    Args:
        models: List of model names to test (uses DEFAULT_MODELS if None)
        questions_csv: Path to the questions CSV file
        questions_per_category: Number of questions to sample per Subject category (None = all questions)
    
    Returns:
        Path to the results directory
    """
    if models is None:
        models = DEFAULT_MODELS
    
    all_results = []
    
    # Load and validate input CSV once (shared across all models)
    df_full = load_questions(questions_csv)
    
    # Filter records without Time period (empty/NaN Time period)
    df_custom = df_full[df_full['Time period'].isna() | (df_full['Time period'].astype(str).str.strip() == '')]
    
    if len(df_custom) == 0:
        print("No records found without Time period. Test 3 cannot run.")
        return None
    
    # Filter out records without Subject
    df_custom = df_custom.dropna(subset=['Subject'])
    df_custom = df_custom[df_custom['Subject'].astype(str).str.strip() != '']
    
    if len(df_custom) == 0:
        print("No records found with Subject field. Test 3 cannot run.")
        return None
    
    # Sample up to N questions per category
    if questions_per_category is None:
        df_sampled = df_custom
    else:
        df_sampled = df_custom.groupby('Subject').head(questions_per_category).reset_index(drop=True)
    
    # Create results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    
    test_dir = f'results/test3-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(test_dir)
    
    unique_categories = df_sampled['Subject'].nunique()
    
    print(f"\n=== TEST 3: Category-based Accuracy on Custom Questions Set ===")
    print(f"Testing {len(models)} models")
    print(f"Questions: {len(df_sampled)} ({'' if questions_per_category is None else f'up to {questions_per_category} per Subject category'})")
    print(f"Unique Subject categories: {unique_categories}")
    print(f"Results will be saved to: {test_dir}\n")
    
    for m in models:
        # Run benchmark for the current model
        model_results = run_model_on_questions(m, df_sampled, SYSTEM_MESSAGE, 'Subject')
        
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
