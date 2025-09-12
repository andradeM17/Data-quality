from dingo.config import InputArgs
from dingo.exec import Executor
import csv
import os

def run_eval(eval_group, input_path):
    input_data = {
        "input_path": input_path,
        "dataset": {
            "source": "local",
            "format": "plaintext"
        },
        "executor": {
            "eval_group": eval_group,
            "result_save": {"bad": True}
        }
    }
    input_args = InputArgs(**input_data)
    executor = Executor.exec_map["local"](input_args)
    return executor.execute()

def main():
    #datasets = ["HPLT", "NLLB", "Paracrawl", "CCMatrix", "XLEnt", "OpenSubtitles", "QED", "EUBookshops", "EUconst", "Tatoeba"]
    
    datasets = ["test"]
    
    results = []
    
    for d in datasets:
        dataset_path = f"Dingo/JSONL/{d}.jsonl"
        dataset_name = os.path.basename(dataset_path[:-6])  # e.g., "hplt.jsonl"

        # Run both evaluations
        for group in ["default", "pretrain"]:
            print(f"Running eval_group='{group}' on dataset='{dataset_name}'...")
            result = run_eval(group, dataset_path)
            print(result)
            results.append({
                "dataset": dataset_name,
                "eval_group": group,
                "score": result.score,
                "num_good": result.num_good,
                "num_bad": result.num_bad,
                "total": result.total
            })

        # Save to CSV
        output_csv = "Dingo/test_eval_results.csv"
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "eval_group", "score", "num_good", "num_bad", "total"])
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    main()
