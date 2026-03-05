#!/usr/bin/env python3
"""
LLM Historical Benchmark Tool
Main entry point for running benchmarks and utilities.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='LLM Historical Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --test1              Run Test 1 (overall accuracy)
  python main.py --test2              Run Test 2 (topic-based accuracy)
  python main.py --test3              Run Test 3 (subject-based accuracy on custom questions)
  python main.py --all                Run all tests
  python main.py --check-answers      Check answer distribution in dataset
        """
    )
    
    parser.add_argument(
        '--test1',
        action='store_true',
        help='Run Test 1: Overall accuracy test on the whole dataset'
    )
    
    parser.add_argument(
        '--test2',
        action='store_true',
        help='Run Test 2: Topic-based accuracy test (grouped by Time period)'
    )
    
    parser.add_argument(
        '--test3',
        action='store_true',
        help='Run Test 3: Subject-based accuracy on custom questions set (records without Time period, grouped by Subject)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all available tests'
    )
    
    parser.add_argument(
        '--check-answers-distribution',
        action='store_true',
        help='Check and visualize the distribution of correct answers in the dataset'
    )
    
    parser.add_argument(
        '--questions-csv',
        type=str,
        default='questions.csv',
        help='Path to the questions CSV file (default: questions.csv)'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Handle check answers distribution
    if args.check_answers_distribution:
        from common import check_answers_distribution
        check_answers_distribution()
        return
    
    # Handle tests
    if args.all:
        args.test1 = True
        args.test2 = True
        args.test3 = True
    
    if args.test1:
        print("\n" + "="*60)
        print("RUNNING TEST 1: Overall Accuracy Test")
        print("="*60)
        from benchmark import run_test1
        run_test1(questions_csv=args.questions_csv)
    
    if args.test2:
        print("\n" + "="*60)
        print("RUNNING TEST 2: Topic-based Accuracy Test")
        print("="*60)
        from benchmark import run_test2
        run_test2(questions_csv=args.questions_csv)
    
    if args.test3:
        print("\n" + "="*60)
        print("RUNNING TEST 3: Category-based Accuracy on Custom Questions Set")
        print("="*60)
        from benchmark import run_test3
        run_test3(questions_csv=args.questions_csv)
    
    if not (args.test1 or args.test2 or args.test3):
        print("No test specified. Use --test1, --test2, --test3 or --all")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
