from dingo.config import InputArgs
from dingo.exec import Executor
import csv
import os
import tempfile

# List of all rules to test
RULE_LIST = [
    "RuleCharNumber",
    "RuleContentShortMultiLan",
    "RuleWordNumber",
    "RuleSentenceNumber",
    "RuleLineEndWithTerminal",
    "RuleStopWord",
    "RuleContentShort",
    "RuleAlphaWords",
    "RuleInvisibleChar",
    "RuleAbnormalChar",
    "RuleMeanWordLength",
    "RuleEnterAndSpace",
    "RuleEnterRatioMore",
    "RuleCapitalWords",
    "RuleLineEndWithEllipsis",
    "RuleContentNull",
    "RuleSpecialCharacter",
    "RuleLineStartWithBulletpoint",
    "RuleDocRepeat",
    "RuleSymbolWordRatio",
    "RuleAbnormalHtml",
    "RuleHtmlEntity",
    "RuleUniqueWords",
    "RuleCurlyBracket",
    "RuleLoremIpsum",
    "RuleColonEnd",
    "RuleEnterMore",
    "RuleHtmlTag",
    "RuleLatexSpecialChar",
    "RuleLineJavascriptCount",
    "RuleSpaceMore",
    "RuleOnlyUrl"
]

def create_limited_dataset(input_path, max_lines=10000):
    """Create a temporary truncated dataset of max_lines lines."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                if i >= max_lines:
                    break
                temp_file.write(line)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        raise e

def run_eval(eval_group, input_path, rules):
    """Run Dingo Executor for a given dataset and rule(s)."""
    input_data = {
        "input_path": input_path,
        "dataset": {"source": "local", "format": "plaintext"},
        "executor": {
            "eval_group": eval_group,
            "rule_list": rules,
            "result_save": {"good": True}
        }
    }
    input_args = InputArgs(**input_data)
    executor = Executor.exec_map["local"](input_args)
    return executor.execute()

def main():
    datasets = ["Review", "Medical", "Literary", "Social media", "Website", "Subtitles", "News"]
    temp_files_to_cleanup = []
    summary_rows = []

    try:
        for d in datasets:
            dataset_path = f"Dingo-experiments/Domain-annotated data/{d}.jsonl"
            dataset_name = os.path.basename(dataset_path[:-6])

            print(f"\nCreating limited dataset (10,000 lines) for {dataset_name}...\n\n")
            limited_dataset_path = create_limited_dataset(dataset_path, max_lines=10000)
            temp_files_to_cleanup.append(limited_dataset_path)

            # Prepare one row per dataset with all rule scores
            dataset_result = {"Rule": dataset_name}

            for rule in RULE_LIST:
                print(f"Testing {rule} on {d}")
                try:
                    result = run_eval("default", limited_dataset_path, [rule])
                    dataset_result[rule] = round(result.score, 2)
                except Exception as e:
                    print(f"⚠️ Error running {rule} on {dataset_name}: {e}")
                    dataset_result[rule] = ""

            summary_rows.append(dataset_result)

        # Write summary CSV (tab-separated)
        output_csv = "Dingo-experiments/results/WMDQS-monolingual-summary.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        fieldnames = ["Rule"] + RULE_LIST

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\n✅ Saved summary results for all rules to {output_csv}")

    finally:
        # Clean up temp files
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

if __name__ == "__main__":
    main()
