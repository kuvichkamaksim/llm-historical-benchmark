"""
Report module for Test 2: Category-based Accuracy Test
Generates a grouped bar chart showing accuracy per category for each model.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_results(results, dir_name):
    """Saves aggregated results to CSV file."""
    if not results:
        print("No results to save.")
        return None

    res_df = pd.DataFrame(results)
    res_df.to_csv(f'{dir_name}/final_aggregation.csv', index=False)
    print(f"--- Results saved to {dir_name}/final_aggregation.csv ---")
    return res_df


def build_category_accuracy_chart(results, dir_name):
    """Builds a grouped bar chart showing accuracy per category for each model."""
    if not results:
        print("No results to build category chart.")
        return

    res_df = pd.DataFrame(results)

    # Calculate accuracy per model and category
    category_summary = res_df.groupby(['model', 'category'])['is_correct'].mean() * 100
    category_summary = category_summary.reset_index()
    category_summary.columns = ['Model', 'Category', 'Accuracy']

    print("\n--- Accuracy by Category (%) ---")
    print(category_summary.pivot(index='Category', columns='Model', values='Accuracy'))

    # Create grouped bar chart
    plt.figure(figsize=(16, 10))
    sns.set_theme(style="whitegrid")
    
    # Use a palette with more distinguishable colors for better visual clarity
    plot = sns.barplot(
        data=category_summary,
        x='Category',
        y='Accuracy',
        hue='Model',
        palette='tab10'
    )

    plt.title('LLM Accuracy by Question Category', pad=20)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Category')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout(pad=2.0)
    plt.savefig(f'{dir_name}/category_accuracy_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--- Category accuracy chart saved to {dir_name}/category_accuracy_chart.png ---")


def build_report(results, dir_name):
    """Main function that builds the Test 2 report."""
    save_results(results, dir_name)
    build_category_accuracy_chart(results, dir_name)
