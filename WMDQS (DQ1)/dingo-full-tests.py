from dingo.config import InputArgs
from dingo.exec import Executor
import csv
import os
import json
import tempfile

RULE_LIST = [
    "RuleLineEndWithEllipsis", "RuleLineEndWithTerminal", "RuleSentenceNumber", "RuleWordNumber",
    "RuleAbnormalChar", "RuleAbnormalHtml", "RuleAlphaWords", "RuleCharNumber", "RuleColonEnd",
    "RuleContentNull", "RuleContentShort", "RuleContentShortMultiLan", "RuleEnterAndSpace", "RuleEnterMore",
    "RuleEnterRatioMore", "RuleHtmlEntity", "RuleHtmlTag", "RuleInvisibleChar", "RuleLatexSpecialChar",
    "RuleLineJavascriptCount", "RuleLoremIpsum", "RuleMeanWordLength", "RuleSpaceMore",
    "RuleSpecialCharacter", "RuleStopWord", "RuleSymbolWordRatio", "RuleOnlyUrl", "RuleDocRepeat",
    "RuleCapitalWords", "RuleCurlyBracket", "RuleLineStartWithBulletpoint", "RuleUniqueWords"
]

def create_limited_dataset(input_path, max_lines=10000):
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
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
    input_data = {
        "input_path": input_path,
        "dataset": {
            "source": "local",
            "format": "plaintext"
        },
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
    datasets = ["h"]
    results = []
    temp_files_to_cleanup = []

    try:
        for d in datasets:
            dataset_path = f"full-datasets/{d}-en-ga.jsonl"
            dataset_name = os.path.basename(dataset_path[:-6])

            print(f"Creating limited dataset (10000 lines) for {dataset_name}...")
            limited_dataset_path = create_limited_dataset(dataset_path, max_lines=10000)
            temp_files_to_cleanup.append(limited_dataset_path)

            for group in ["default"]:
                for rule in RULE_LIST:
                    print(f"Running eval_group='{group}' on dataset='{dataset_name}' with rule='{rule}'...")
                    result = run_eval(group, limited_dataset_path, [rule])

                    row = {
                        "dataset": dataset_name,
                        "eval_group": group,
                        "rule_applied": rule,
                        "task_id": result.task_id,
                        "task_name": getattr(result, "task_name", ""),
                        "score": result.score,
                        "num_good": result.num_good,
                        "num_bad": result.num_bad,
                        "total": result.total,
                        "type_ratio": json.dumps(result.type_ratio),
                        "name_ratio": json.dumps(result.name_ratio)
                    }

                    for r in RULE_LIST:
                        row[r] = result.name_ratio.get(f'QUALITY_BAD_COMPLETENESS-{r}', 0.0)

                    results.append(row)

                    # Print the line to console after each rule check
                    print(row)

    finally:
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    fieldnames = [
        "dataset", "eval_group", "rule_applied", "task_id", "task_name",
        "score", "num_good", "num_bad", "total",
        "type_ratio", "name_ratio"
    ] + RULE_LIST

    output_csv = "HPLT_eval_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    main()
