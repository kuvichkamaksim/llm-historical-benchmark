import ollama
import pandas as pd
import time
import re
import os
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

ANSWER_PATTERN = re.compile(r'\b([ABCD])\b', re.IGNORECASE | re.UNICODE)

SYSTEM_MESSAGE = """Ти — об'єктивний асистент.
Твоє завдання — відповідати на тестові питання, обираючи лише одну правильну літеру (A, B, C або D).
Не пиши нічого іншого, крім цієї літери."""

def extract_answer(text):
    """Extracts the answer from the text using Regular Expression."""
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None

def stop_model(model_name):
    """Forcefully unloads the model from Ollama memory via CLI."""
    try:
        subprocess.run(["ollama", "stop", model_name], check=True, capture_output=True)
        print(f"--- Model {model_name} forcefully unloaded ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Failed to stop model {model_name}: {e} ---")

def run_benchmark(model_name, input_file, system_prompt):
    """Runs the benchmark for a specific model."""
    # Load dataset with headers, limit to first 10 rows

    df = pd.read_csv(input_file, header=0, nrows=10)
    results = []

    print(f"\n>>> Starting benchmark for model: {model_name}")

    for index, row in df.iterrows():
        prompt = f"""
        Питання: {row['Question']}
        Варіанти:
        A) {row['Choice A']}
        B) {row['Choice B']}
        C) {row['Choice C']}
        D) {row['Choice D']}
        
        Твоя відповідь:
        """

        try:
            # Model call (stateless, temperature=0 for determinism)
            # We use ollama.generate() rather than ollama.chat() because ollama.generate() doesn't save any context between calls
            response = ollama.generate(
                model=model_name,
                system=system_prompt,
                prompt=prompt,
                options={
                  'temperature': 0,   # For deterministic output
                  # 'num_predict': 10,  # Limit response length
                  'top_k': 1          # Greediest sampling
                  }
            )

            thinking = response.get('thinking', None)
            raw_response = response.get('response', '')
            answer = extract_answer(raw_response)

            actual_answer = str(row['Correct Answer']).strip().upper()
            is_correct = (answer == actual_answer) if answer else False
            
            results.append({
                'id': row['ID'],
                'model': model_name,
                'category': row['Topic'],
                'raw_response': raw_response.strip() or 'N/A',
                'thinking': thinking or 'N/A',
                'predicted': answer or 'N/A',
                'actual': actual_answer,
                'is_correct': is_correct
            })

            print(f"[{model_name}] Question {row['ID']}. Answer: {answer}. Correct: {row['Correct Answer']}")

        except Exception as e:
            print(f"Error while running {model_name}: {e}")

    return results

def build_report(results, dir_name):
  """Aggregates results and generates visualizations."""
  if not results:
    print("No results to report.")
    return

  res_df = pd.DataFrame(results)
  summary = res_df.groupby('model')['is_correct'].mean() * 100
  print("\n--- Final Results (Accuracy %) ---")
  print(summary)

  # Save aggregated results
  res_df.to_csv(f'{dir_name}/final_aggregation.csv', index=False)

  # Build graph
  plt.figure(figsize=(10, 6))
  sns.set_theme(style="whitegrid")
  plot = sns.barplot(x=summary.index, y=summary.values, hue=summary.index, palette='viridis', legend=False)

  plt.title('LLM Accuracy Comparison in Ukrainian Cultural Context')
  plt.ylabel('Accuracy (%)')
  plt.xlabel('Model Name')
  plt.ylim(0, 100)

  # Add percentage labels on top of bars
  for p in plot.patches:
    plot.annotate(format(p.get_height(), '.1f'),
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha = 'center', va = 'center',
                  xytext = (0, 9),
                  textcoords = 'offset points')

  plt.savefig(f'{dir_name}/benchmark_accuracy_chart.png', dpi=300)
  print("\n--- Charts saved to results/ ---")

def main():
  # Ensure these models are already pulled via 'ollama pull <name>' or created via 'ollama create <name> -f models/<name>.Modelfile'
  models = ['gemma3:4b', 'lapa-v0.1.2-q4', 'mamay-9b-q4', 'mamay-4b-q4', 'deepseek-r1:8b']
  all_results = []

  # Create results directory if it doesn't exist
  if not os.path.exists('results'):
    os.makedirs('results')

  test_dir = f'results/test-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
  os.makedirs(test_dir)

  for m in models:
    # Run benchmark for the current model
    model_results = run_benchmark(m, 'questions.csv', SYSTEM_MESSAGE)

    # Save individual model results to CSV
    model_df = pd.DataFrame(model_results)
    model_df.to_csv(f'{test_dir}/{m.replace(":", "_")}_results.csv', index=False)

    all_results.extend(model_results)

    # Forceful resource cleanup
    stop_model(m)
    time.sleep(3) # Short cooldown for hardware stabilization

  build_report(all_results, test_dir)

if __name__ == '__main__':
  main()
