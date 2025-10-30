from dingo.config import InputArgs
from dingo.exec import Executor
import csv
import os
import tempfile

# ----------------------------------------------------------------------
# Add the description helper function
# ----------------------------------------------------------------------
def get_rule_description(rule, result):
    """Return a description string for the rule based on the result."""
    match rule:
        case "RuleLineEndWithEllipsis":
            return f"Over 30% of sentences in {100 - result.score:.0f}% of the samples end with ellipsis (...)."
        case "RuleLineEndWithTerminal":
            return f"Over 60% of sentences in {result.score:.0f}% of the samples end with terminal punctuation (.!?;)."
        case "RuleSentenceNumber":
            return f"{result.score:.0f}% of the samples have between 3 and 7,500 sentences."
        case "RuleWordNumber":
            return f"{result.score:.0f}% of the samples have between 20 and 100,000 words."
        case "RuleAlphaWords":
            return f"Over 60% of the words in {result.score:.0f}% of the samples have alphabetic characters."
        case "RuleCharNumber":
            return f"{result.score:.0f}% of the samples have more than 100 characters."
        case "RuleColonEnd":
            return f"{100 - result.score:.0f}% of the samples finish in a colon (:)."
        case "RuleContentNull":
            return f"{100 - result.score:.0f}% of the samples are empty."
        case "RuleHtmlEntity":
            return f"{100 - result.score:.0f}% of the samples have HTML content."
        case "RuleLineJavascriptCount":
            return f' {100 - result.score:.0f}% of the samples contain the word "javascript".'
        case "RuleLoremIpsum":
            return f"{100 - result.score:.0f}% of the samples contain Lorem Ipsum text."
        case "RuleMeanWordLength":
            return f"{result.score:.0f}% of the samples have a mean word length between 3 and 10 characters."
        case "RuleSpecialCharacter":
            return f"{100 - result.score:.0f}% of the samples have special characters."
        case "RuleStopWord":
            return f"Over 6% of words in {100 - result.score:.0f}% of the samples are stop words."
        case "RuleSymbolWordRatio":
            return f"In {100 - result.score:.0f}% of the samples, the ratio between symbols and words is > 0.4."
        case "RuleDocRepeat":
            return f"{100 - result.score:.0f}% of the samples have repeating lines."
        case "RuleCapitalWords":
            return f"Over 20% of words in {100 - result.score:.0f}% of the samples are capitalised."
        case "RuleCurlyBracket":
            return f"There is a ratio > 0.1 between curly brackets and other characters in {100 - result.score:.0f}% of the samples."
        case "RuleLineStartWithBulletpoint":
            return f"{100 - result.score:.0f}% of the samples start with a bullet point."
        case "RuleUniqueWords":
            return f"{100 - result.score:.0f}% of the samples have a ratio > 0.1 of unique words."
        case _:
            return ""

# ----------------------------------------------------------------------
# Rule list
# ----------------------------------------------------------------------
RULE_LIST = [
    "RuleCharNumber", "RuleContentShortMultiLan", "RuleWordNumber", "RuleSentenceNumber",
    "RuleLineEndWithTerminal", "RuleStopWord", "RuleContentShort", "RuleAlphaWords",
    "RuleInvisibleChar", "RuleAbnormalChar", "RuleMeanWordLength", "RuleEnterAndSpace",
    "RuleEnterRatioMore", "RuleCapitalWords", "RuleLineEndWithEllipsis", "RuleContentNull",
    "RuleSpecialCharacter", "RuleLineStartWithBulletpoint", "RuleDocRepeat",
    "RuleSymbolWordRatio", "RuleAbnormalHtml", "RuleHtmlEntity", "RuleUniqueWords",
    "RuleCurlyBracket", "RuleLoremIpsum", "RuleColonEnd", "RuleEnterMore", "RuleHtmlTag",
    "RuleLatexSpecialChar", "RuleLineJavascriptCount", "RuleSpaceMore", "RuleOnlyUrl"
]

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    datasets = ["Review", "Medical", "Literary", "Social media", "Website", "Subtitles", "News"]
    temp_files_to_cleanup = []
    summary_rows = []

    try:
        for d in datasets:
            dataset_path = f"Dingo-experiments/Domain-annotated data/{d}.txt"
            dataset_name = os.path.basename(dataset_path[:-6])

            print(f"\nCreating limited dataset (10,000 lines) for {dataset_name}...\n\n")
            limited_dataset_path = create_limited_dataset(dataset_path, max_lines=10000)
            temp_files_to_cleanup.append(limited_dataset_path)

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

        # ------------------------------------------------------------------
        # Write summary CSV (tab-separated) with rule descriptions as 2nd row
        # ------------------------------------------------------------------
        output_csv = "Dingo-experiments/Domain-annotated data/domains.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        fieldnames = ["Rule"] + RULE_LIST

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Create a dummy object with .score=0 for description generation
            class Dummy:
                def __init__(self, score=50): self.score = score

            description_row = {"Rule": "Description (In the case that the score is 50%...)"}
            for rule in RULE_LIST:
                description_row[rule] = get_rule_description(rule, Dummy())

            # Write the description row
            writer.writerow(description_row)

            # Write the dataset results
            writer.writerows(summary_rows)

        print(f"\n✅ Saved summary results (with descriptions) to {output_csv}")

    finally:
        # Clean up temp files
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

if __name__ == "__main__":
    main()
