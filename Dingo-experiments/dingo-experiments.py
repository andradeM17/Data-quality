from dingo.config import InputArgs
from dingo.exec import Executor
import csv
import os
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

def get_rule_description(rule, result):
    """Return a description string for the rule based on the result."""
    match rule:
        case "RuleLineEndWithEllipsis":
            return f"Over 30% of sentences in {100-result.score:.0f}% of the samples end with ellipsis (...)"
        case "RuleLineEndWithTerminal":
            return f"Over 60% of sentences in {result.score:.0f}% of the samples end with terminal punctuation (.!?;)"
        case "RuleSentenceNumber":
            return f"{result.score:.0f}% of the samples have between 3 and 7,500 sentences."
        case "RuleWordNumber":
            return f"{result.score:.0f}% of the samples have between 20 and 100,000 words."
        case "RuleAlphaWords":
            return f"Over 60% of the words in {result.score:.0f}% of the samples have alphabetic characters."
        case "RuleCharNumber":
            return f"{result.score:.0f}% of the samples have more than 100 characters."
        case "RuleColonEnd":
            return f"{100-result.score:.0f}% of the samples finish in a colon (:)."
        case "RuleContentNull":
            return f"{100-result.score:.0f}% of the samples are empty."
        case "RuleHtmlEntity":
            return f"{100-result.score:.0f}% of the samples have HTML content."
        case "RuleLineJavascriptCount":
            return f'{100-result.score:.0f}% of the samples contain the word "javascript".'
        case "RuleLoremIpsum":
            return f"{100-result.score:.0f}% of the samples contain Lorem Ipsum text."
        case "RuleMeanWordLength":
            return f"{result.score:.0f}% of the samples have a mean word length between 3 and 10 characters."
        case "RuleSpecialCharacter":
            return f"{100-result.score:.0f}% of the samples have special characters."
        case "RuleStopWord":
            return f"Over 6% of words in {100-result.score:.0f}% of the samples are stop words."
        case "RuleSymbolWordRatio":
            return f"In {100-result.score:.0f}% of the samples, the ratio between symbols and words is > 0.4."
        case "RuleDocRepeat":
            return f"{100-result.score:.0f}% of the samples have repeating lines."
        case "RuleCapitalWords":
            return f"Over 20% of words in {100-result.score:.0f}% of the samples are capitalised."
        case "RuleCurlyBracket":
            return f"There is a ratio > 0.1 between curly brackets and other characters in {result.score:.0f}% of the samples."
        case "RuleLineStartWithBulletpoint":
            return f"{100-result.score:.0f}% of the samples start with a bullet point."
        case "RuleUniqueWords":
            return f"{result.score:.0f}% of the samples have a ratio > 0.1 of unique words."
        case _:
            return ""

def main():
    datasets = ["n-ga"]
    temp_files_to_cleanup = []

    try:
        for d in datasets:
            dataset_path = f"Dingo-experiments/data/{d}.txt"
            dataset_name = os.path.basename(dataset_path[:-6])

            print(f"\nCreating limited dataset (10000 lines) for {dataset_name}...\n\n")
            limited_dataset_path = create_limited_dataset(dataset_path, max_lines=10000)
            temp_files_to_cleanup.append(limited_dataset_path)

            results = []

            for group in ["default"]:
                for rule in RULE_LIST:
                    print(f"Testing {rule} on {d}")
                    result = run_eval(group, limited_dataset_path, [rule])
                    description = get_rule_description(rule, result)
                    if description and round(result.score) != 0 and round(result.score) != 100:
                        print(description + "\n")

                    row = {
                        "rule_applied": rule,
                        "score": result.score,
                        "num_good": result.num_good,
                        "num_bad": result.num_bad,
                        "total": result.total,
                        "description": description
                    }

                    results.append(row)

            # Write CSV for this dataset
            fieldnames = ["rule_applied", "score", "num_good", "num_bad", "total", "description"]
            output_csv = f"Dingo-experiments/results/{d}.csv"

            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            print(f"\nSaved results to {output_csv}")

    finally:
        # Clean up temporary files
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

if __name__ == "__main__":
    main()
