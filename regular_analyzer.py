import argparse
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def analyze_regularity_with_gemini(language_description):
    """
    Analyzes the language description using Gemini to determine regularity and get structured output.
    """
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set. Please set your Gemini API key.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

    prompt_template = PromptTemplate.from_template(
        """Analyze the following language description. Determine if it is a regular language. If it is regular, describe a Deterministic Finite Automaton (DFA) that recognizes it in a structured format. If it is not regular, explain why.

Language Description: {language_description}

Respond in JSON format.

For regular languages, use this JSON structure:
{{
  "regularity": "Regular",
  "alphabet": [ ...list of alphabet symbols... ],
  "states": [ ...list of state names... ],
  "initial_state": "...",
  "final_states": [ ...list of final state names... ],
  "transitions": [
    {{ "from_state": "...", "symbol": "...", "to_state": "..." }},
    ... more transitions ...
  ]
}}

For non-regular languages, use this JSON structure:
{{
  "regularity": "Non-regular",
  "reason": "..."
}}

Example Language Description: Language of all strings with an even number of 'a's over alphabet {{a, b}}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    regularity_response = chain.run(language_description=language_description)
    return regularity_response.strip()

def main():
    """
    Main function to set up argument parser and call the language analysis function.
    """
    parser = argparse.ArgumentParser(description="Regular Language Analyzer CLI")
    parser.add_argument("description", help="Description of the language to analyze")
    args = parser.parse_args()

    language_description = args.description
    print(f"Analyzing language: {language_description}")

    try:
        structured_output_json = analyze_regularity_with_gemini(language_description)
        print(structured_output_json)

        # --- Next steps would be to parse the JSON and use the fado library ---
        # --- (Code for fado integration will be added in subsequent steps) ---


    except EnvironmentError as e:
        print(f"Error: {e}")
        print("Please make sure you have set your GOOGLE_API_KEY environment variable.")
    except Exception as e:
        print(f"An error occurred during language analysis: {e}")

if __name__ == "__main__":
    main()