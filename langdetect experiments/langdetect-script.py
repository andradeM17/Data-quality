from collections import namedtuple
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging, os

logger = logging.getLogger(__name__)
output_template = "{0:<40}\t{1:<10}\t{2:<80}"


DetectorFactory.seed = 0

DetectionResult = namedtuple("DetectionResult", ["language", "probability", "total", "tested"])

class LanguageDetector:

    DEFAULT_MIN_FILE_LINE_LENGTH = 0
    DEFAULT_MIN_INITIAL_LINES = 50
    DEFAULT_SAMPLING_INTERVAL = 100
    DEFAULT_LANGUAGE = ""
    DEFAULT_PROBABILITY = -1.0

    def detect(self, str):
        detected = self.detect_language(str)
        return detected.language


    def detect_in_file(self, input_path, config={}):
        logger.info("Detecting language in file {0}.".format(input_path))
        with open(input_path, encoding='utf-8') as input_file:
            detected = self.detect_language_in_file(input_file, config)
            return detected.language


    def detect_language(self, str):
        try:
            detected_langs = detect_langs(str.lower())
            logger.debug("Detected language(s) with probabilities {0}.".format(detected_langs))
            detected_lang = max(detected_langs, key=lambda x: x.prob)
            return DetectionResult(language=detected_lang.lang, probability=detected_lang.prob, total=1, tested=1)
        except LangDetectException as lde:
            # langdetect produces a lot of these, mostly for empty or otherwise irrelevant lines
            logger.warning("Language could not be detected, exception: {0}.".format(lde))
            return DetectionResult(language=self.DEFAULT_LANGUAGE, probability=self.DEFAULT_PROBABILITY, total=1, tested=0)
        except Exception as e:
            logger.exception("Language could not be detected, exception: {0}.".format(e))
            return DetectionResult(language=self.DEFAULT_LANGUAGE, probability=self.DEFAULT_PROBABILITY, total=1, tested=0)


    def detect_language_in_file(self, lines, config={}):
        min_file_line_length = int(config.get("min_file_line_length", self.DEFAULT_MIN_FILE_LINE_LENGTH))
        min_initial_lines = int(config.get("min_initial_lines", self.DEFAULT_MIN_INITIAL_LINES))
        sampling_interval = int(config.get("sampling_interval", self.DEFAULT_SAMPLING_INTERVAL))

        combined_probabilities = {}
        count = 0
        tested_count = 0

        for line in lines:
            count += 1

            if len(line) >= min_file_line_length and (count <= min_initial_lines or count % sampling_interval == 0):
                try:
                    detected_langs = detect_langs(line.lower())
                    tested_count += 1

                    for detected_lang in detected_langs:
                        code = detected_lang.lang
                        if not code in combined_probabilities:
                            combined_probabilities[code] = 0
                        combined_probabilities[code] += detected_lang.prob

                except LangDetectException as lde:
                    # langdetect produces a lot of these, mostly for empty or otherwise irrelevant lines;
                    logger.warning("Language not detected for line (line ignored), langdetect exception: \"{0}\".".format(str(lde)))
                    continue

                except Exception as e:
                    logger.exception("Unexpected exception (line ignored): {0}.".format(e))
                    continue

        language, total_probability = self.find_max_probability(combined_probabilities)
        normalized_probability = self.calculate_normalized_probability(total_probability, tested_count)
        return DetectionResult(language=language, probability=normalized_probability, total=count, tested=tested_count)


    def find_max_probability(self, combined_probabilities):
        return max(combined_probabilities.items(), key=lambda x: x[1], default=(self.DEFAULT_LANGUAGE, self.DEFAULT_PROBABILITY))


    def calculate_normalized_probability(self, total_probability, tested_count):
        if tested_count <= 0:
            return self.DEFAULT_PROBABILITY
        return total_probability / tested_count

    
    def print_detected(s):
        print(str(detect_langs(s)))
        return

if __name__ == "__main__":
    folder_path = "langdetect experiments/c4"
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isfile(full_path):
            print(f"Processing file: {file_name}")
            detector = LanguageDetector()
            print(detector.detect_in_file(full_path))
            
            with open(full_path, 'r', encoding="utf-8") as f:
                content = f.read()
                LanguageDetector.print_detected(content)
                print("\n +++++++++++++++++++++++++++ \n")