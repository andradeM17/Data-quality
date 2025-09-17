from dingo.config import InputArgs
from dingo.exec import Executor
import csv
import os
import json

RULE_LIST = [
    "RuleLineEndWithEllipsis", "RuleLineEndWithTerminal", "RuleSentenceNumber", "RuleWordNumber",
    "RuleAbnormalChar", "RuleAbnormalHtml", "RuleAlphaWords", "RuleCharNumber", "RuleColonEnd",
    "RuleContentNull", "RuleContentShort", "RuleContentShortMultiLan", "RuleEnterAndSpace", "RuleEnterMore",
    "RuleEnterRatioMore", "RuleHtmlEntity", "RuleHtmlTag", "RuleInvisibleChar", "RuleLatexSpecialChar",
    "RuleLineJavascriptCount", "RuleLoremIpsum", "RuleMeanWordLength", "RuleSpaceMore",
    "RuleSpecialCharacter", "RuleStopWord", "RuleSymbolWordRatio", "RuleOnlyUrl", "RuleDocRepeat",
    "RuleCapitalWords", "RuleCurlyBracket", "RuleLineStartWithBulletpoint", "RuleUniqueWords"
]


def run_eval(eval_group, input_path):
    input_data = {
        "input_path": input_path,
        "dataset": {
            "source": "local",
            "format": "plaintext"
        },
        "executor": {
            "eval_group": eval_group,
            "rule_list": RULE_LIST,
            "result_save": {"good": True}
        }
    }
    input_args = InputArgs(**input_data)
    executor = Executor.exec_map["local"](input_args)
    return executor.execute()

def main():
    datasets = ["HPLT", "NLLB", "Paracrawl", "CCMatrix", "XLEnt", "OpenSubtitles", "QED", "EUBookshops", "EUconst", "Tatoeba"]

    results = []
    
    for d in datasets:
        dataset_path = f"Dingo/JSONL/{d}.jsonl"
        dataset_name = os.path.basename(dataset_path[:-6])

        for group in ["default"]:
            print(f"Running eval_group='{group}' on dataset='{dataset_name}'...")
            result = run_eval(group, dataset_path)
            print(result)
            
            # Start with basic fields
            row = {
                "dataset": dataset_name,
                "eval_group": group,
                "task_id": result.task_id,
                "task_name": getattr(result, "task_name", ""),
                "score": result.score,
                "num_good": result.num_good,
                "num_bad": result.num_bad,
                "total": result.total,
                "type_ratio": json.dumps(result.type_ratio),
                "name_ratio": json.dumps(result.name_ratio)
            }

            # Add a column for each rule
            for rule in RULE_LIST:
                # Default to 0 if rule not in name_ratio
                row[rule] = result.name_ratio.get(f'QUALITY_BAD_COMPLETENESS-{rule}', 0.0)

            results.append(row)

    # Prepare CSV fieldnames: basic fields + one column per rule
    fieldnames = [
        "dataset", "eval_group", "task_id", "task_name",
        "score", "num_good", "num_bad", "total",
        "type_ratio", "name_ratio"
    ] + RULE_LIST

    # Write all results to CSV once
    output_csv = "Dingo/eval_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    main()
