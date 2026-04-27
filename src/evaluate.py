from transformers import AutoModelForCausalLM, AutoTokenizer
import textstat


def readability_score(text: str):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
    }


if __name__ == "__main__":
    sample = "The dragon was afraid of fire. He learned to blow cold air instead."
    print(readability_score(sample))
