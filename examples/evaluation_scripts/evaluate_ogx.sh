#!/bin/bash

python3 --version
which python3
echo $CONDA_DEFAULT_ENV

_MMLU_SUBJECTS=("abstract_algebra" "anatomy" "astronomy" "business_ethics" "clinical_knowledge" "college_biology" "college_chemistry" "college_computer_science" "college_mathematics" "college_medicine" "college_physics" "computer_security" "conceptual_physics" "econometrics" "electrical_engineering" "elementary_mathematics" "formal_logic" "global_facts" "high_school_biology" "high_school_chemistry" "high_school_computer_science" "high_school_european_history" "high_school_geography" "high_school_government_and_politics" "high_school_macroeconomics" "high_school_mathematics" "high_school_microeconomics" "high_school_physics" "high_school_psychology" "high_school_statistics" "high_school_us_history" "high_school_world_history" "human_aging" "human_sexuality" "international_law" "jurisprudence" "logical_fallacies" "machine_learning" "management" "marketing" "medical_genetics" "miscellaneous" "moral_disputes" "moral_scenarios" "nutrition" "philosophy" "prehistory" "professional_accounting" "professional_law" "professional_medicine" "professional_psychology" "public_relations" "security_studies" "sociology" "us_foreign_policy" "virology" "world_religions")
LANGUAGES=("en" "de" "fr" "it" "es" "pt-pt" "ro" "cs" "da" "el" "et" "fi" "hu" "lt" "lv" "nl" "bg" "pl" "sk" "sl" "sv")
LC_TO_LANG=("eng_Latn" "bul_Cyrl" "dan_Latn" "deu_Latn" "est_Latn" "fin_Latn" "fra_Latn" "ell_Grek" "ita_Latn" "lvs_Latn" "lit_Latn" "nld_Latn" "pol_Latn" "por_Latn" "ron_Latn" "swe_Latn" "slk_Latn" "slv_Latn" "spa_Latn" "ces_Latn" "hun_Latn")

# Define the path to your Python script

############ yaml_multilingual_tasks_alex branch of OGX ################
SCRIPT="/raid/s3/opengptx/alexj/github_runner/evaluation/lm-evaluation-harness/lm_eval/__main__.py"
MODEL_NAME=baseline-7-8b_2-3t-tokens_llama
MODEL_DIR=TrustLLMeu/baseline-7-8b_2-3t-tokens_llama

## Output directory
OUTPUT_DIRECTORY="/raid/s3/opengptx/alexj/modalities_eval/lm-evaluation-harness/evaluation_scripts/results/$MODEL_NAME"
# Check if the directory does not exist
if [ ! -d "$OUTPUT_DIRECTORY" ]; then
  # Create the directory
  mkdir -p "$OUTPUT_DIRECTORY"
  echo "Directory '$OUTPUT_DIRECTORY' created."
else
  echo "Directory '$OUTPUT_DIRECTORY' already exists."
fi

########## Creating tasks################
# Initialize an empty string to hold the result
result=""
# Loop through each element in the list
for lang in "${LANGUAGES[@]}"; do
  for subject in "${_MMLU_SUBJECTS[@]}"; do
      if [ "$lang" != "en" ]; then
        formatted_string="ogx_mmlux_${lang}-${subject}"
      else
        formatted_string="mmlu_${subject}"
      fi
    result+="${formatted_string},"
  done
done
MMLU_TASKS=${result%,}


# Initialize an empty string to hold the result
result=""
# Loop through each element in the list
for lang in "${LANGUAGES[@]}"; do
    if [ "$lang" != "en" ]; then
      formatted_string="ogx_hellaswagx_${lang}"
    else
      formatted_string="hellaswag"
    fi
  result+="${formatted_string},"
done
HELLASWAG_TASKS=${result%,}


# Initialize an empty string to hold the result
result=""
# Loop through each element in the list
for lang in "${LANGUAGES[@]}"; do
    if [ "$lang" != "en" ]; then
      formatted_string="ogx_truthfulqax_mc2_${lang}"
    else
      formatted_string="truthfulqa_mc2"
    fi
  result+="${formatted_string},"
done
TRUTHFULQA_TASKS=${result%,}


# Initialize an empty string to hold the result
result=""
# Loop through each element in the list
for lang in "${LANGUAGES[@]}"; do
    if [ "$lang" != "en" ]; then
      formatted_string="ogx_gsm8kx_${lang}"
    else
      formatted_string="ogx_gsm8k"
    fi
  result+="${formatted_string},"
done
GSM8K_TASKS=${result%,}


# Initialize an empty string to hold the result
result=""
# Loop through each element in the list
for lang in "${LANGUAGES[@]}"; do
    if [ "$lang" != "en" ]; then
      formatted_string="ogx_arcx_easy_${lang}"
    else
      formatted_string="arc_easy"
    fi
  result+="${formatted_string},"
done
ARC_EASY_TASKS=${result%,}


# Initialize an empty string to hold the result
result=""
# Loop through each element in the list
for lang in "${LANGUAGES[@]}"; do
    if [ "$lang" != "en" ]; then
      formatted_string="ogx_arcx_challenge_${lang}"
    else
      formatted_string="arc_challenge"
    fi
    result+="${formatted_string},"
done
ARC_CHALLENGE_TASKS=${result%,}


# Initialize an empty string to hold the result
result=""
# Loop through each element in the list
for lang in "${LC_TO_LANG[@]}"; do
    formatted_string="belebele_${lang}"
    result+="${formatted_string},"
done
BELEBELE_TASKS=${result%,}

#################### FLORES TASKS ##########################

count=0
# Initialize an empty string to hold the result
result=""
# Loop through each element in the list
for lang1 in "${LC_TO_LANG[@]}"; do
  for lang2 in "${LC_TO_LANG[@]}"; do
    if [ "$lang1" != "$lang2" ]; then
      formatted_string="ogx_flores200-trans-${lang1}-${lang2}"
      result+="${formatted_string},"
      # Increment the counter
      count=$((count + 1))
    fi
  done
done
FLORES200_TASKS=${result%,}


FLORES200_TASKS_K_1=ogx_flores200-trans-bul_Cyrl-ces_Latn,ogx_flores200-trans-bul_Cyrl-dan_Latn,ogx_flores200-trans-bul_Cyrl-deu_Latn,ogx_flores200-trans-bul_Cyrl-ell_Grek,ogx_flores200-trans-bul_Cyrl-eng_Latn,ogx_flores200-trans-bul_Cyrl-est_Latn,ogx_flores200-trans-bul_Cyrl-fin_Latn,ogx_flores200-trans-bul_Cyrl-fra_Latn,ogx_flores200-trans-bul_Cyrl-hun_Latn,ogx_flores200-trans-bul_Cyrl-ita_Latn,ogx_flores200-trans-bul_Cyrl-lit_Latn,ogx_flores200-trans-bul_Cyrl-lvs_Latn,ogx_flores200-trans-bul_Cyrl-nld_Latn,ogx_flores200-trans-bul_Cyrl-pol_Latn,ogx_flores200-trans-bul_Cyrl-por_Latn,ogx_flores200-trans-bul_Cyrl-ron_Latn,ogx_flores200-trans-bul_Cyrl-slk_Latn,ogx_flores200-trans-bul_Cyrl-slv_Latn,ogx_flores200-trans-bul_Cyrl-spa_Latn,ogx_flores200-trans-bul_Cyrl-swe_Latn,ogx_flores200-trans-ces_Latn-bul_Cyrl,ogx_flores200-trans-ces_Latn-dan_Latn,ogx_flores200-trans-ces_Latn-deu_Latn,ogx_flores200-trans-ces_Latn-ell_Grek,ogx_flores200-trans-ces_Latn-eng_Latn,ogx_flores200-trans-ces_Latn-est_Latn,ogx_flores200-trans-ces_Latn-fin_Latn,ogx_flores200-trans-ces_Latn-fra_Latn,ogx_flores200-trans-ces_Latn-hun_Latn,ogx_flores200-trans-ces_Latn-ita_Latn,ogx_flores200-trans-ces_Latn-lit_Latn,ogx_flores200-trans-ces_Latn-lvs_Latn,ogx_flores200-trans-ces_Latn-nld_Latn,ogx_flores200-trans-ces_Latn-pol_Latn,ogx_flores200-trans-ces_Latn-por_Latn,ogx_flores200-trans-ces_Latn-ron_Latn,ogx_flores200-trans-ces_Latn-slk_Latn,ogx_flores200-trans-ces_Latn-slv_Latn,ogx_flores200-trans-ces_Latn-spa_Latn,ogx_flores200-trans-ces_Latn-swe_Latn,ogx_flores200-trans-dan_Latn-bul_Cyrl,ogx_flores200-trans-dan_Latn-ces_Latn,ogx_flores200-trans-dan_Latn-deu_Latn,ogx_flores200-trans-dan_Latn-ell_Grek,ogx_flores200-trans-dan_Latn-eng_Latn,ogx_flores200-trans-dan_Latn-est_Latn,ogx_flores200-trans-dan_Latn-fin_Latn,ogx_flores200-trans-dan_Latn-fra_Latn,ogx_flores200-trans-dan_Latn-hun_Latn,ogx_flores200-trans-dan_Latn-ita_Latn,ogx_flores200-trans-dan_Latn-lit_Latn,ogx_flores200-trans-dan_Latn-lvs_Latn,ogx_flores200-trans-dan_Latn-nld_Latn,ogx_flores200-trans-dan_Latn-pol_Latn,ogx_flores200-trans-dan_Latn-por_Latn,ogx_flores200-trans-dan_Latn-ron_Latn,ogx_flores200-trans-dan_Latn-slk_Latn,ogx_flores200-trans-dan_Latn-slv_Latn,ogx_flores200-trans-dan_Latn-spa_Latn,ogx_flores200-trans-dan_Latn-swe_Latn,ogx_flores200-trans-deu_Latn-bul_Cyrl,ogx_flores200-trans-deu_Latn-ces_Latn,ogx_flores200-trans-deu_Latn-dan_Latn,ogx_flores200-trans-deu_Latn-ell_Grek,ogx_flores200-trans-deu_Latn-eng_Latn,ogx_flores200-trans-deu_Latn-est_Latn,ogx_flores200-trans-deu_Latn-fin_Latn,ogx_flores200-trans-deu_Latn-fra_Latn,ogx_flores200-trans-deu_Latn-hun_Latn,ogx_flores200-trans-deu_Latn-ita_Latn,ogx_flores200-trans-deu_Latn-lit_Latn,ogx_flores200-trans-deu_Latn-lvs_Latn,ogx_flores200-trans-deu_Latn-nld_Latn,ogx_flores200-trans-deu_Latn-pol_Latn,ogx_flores200-trans-deu_Latn-por_Latn,ogx_flores200-trans-deu_Latn-ron_Latn,ogx_flores200-trans-deu_Latn-slk_Latn,ogx_flores200-trans-deu_Latn-slv_Latn,ogx_flores200-trans-deu_Latn-spa_Latn,ogx_flores200-trans-deu_Latn-swe_Latn,ogx_flores200-trans-ell_Grek-bul_Cyrl,ogx_flores200-trans-ell_Grek-ces_Latn,ogx_flores200-trans-ell_Grek-dan_Latn,ogx_flores200-trans-ell_Grek-deu_Latn,ogx_flores200-trans-ell_Grek-eng_Latn,ogx_flores200-trans-ell_Grek-est_Latn,ogx_flores200-trans-ell_Grek-fin_Latn,ogx_flores200-trans-ell_Grek-fra_Latn,ogx_flores200-trans-ell_Grek-hun_Latn,ogx_flores200-trans-ell_Grek-ita_Latn,ogx_flores200-trans-ell_Grek-lit_Latn,ogx_flores200-trans-ell_Grek-lvs_Latn,ogx_flores200-trans-ell_Grek-nld_Latn,ogx_flores200-trans-ell_Grek-pol_Latn,ogx_flores200-trans-ell_Grek-por_Latn,ogx_flores200-trans-ell_Grek-ron_Latn,ogx_flores200-trans-ell_Grek-slk_Latn,ogx_flores200-trans-ell_Grek-slv_Latn,ogx_flores200-trans-ell_Grek-spa_Latn,ogx_flores200-trans-ell_Grek-swe_Latn,ogx_flores200-trans-eng_Latn-bul_Cyrl,ogx_flores200-trans-eng_Latn-ces_Latn,ogx_flores200-trans-eng_Latn-dan_Latn,ogx_flores200-trans-eng_Latn-deu_Latn,ogx_flores200-trans-eng_Latn-ell_Grek,ogx_flores200-trans-eng_Latn-est_Latn,ogx_flores200-trans-eng_Latn-fin_Latn,ogx_flores200-trans-eng_Latn-fra_Latn,ogx_flores200-trans-eng_Latn-hun_Latn,ogx_flores200-trans-eng_Latn-ita_Latn,ogx_flores200-trans-eng_Latn-lit_Latn,ogx_flores200-trans-eng_Latn-lvs_Latn,ogx_flores200-trans-eng_Latn-nld_Latn,ogx_flores200-trans-eng_Latn-pol_Latn,ogx_flores200-trans-eng_Latn-por_Latn,ogx_flores200-trans-eng_Latn-ron_Latn,ogx_flores200-trans-eng_Latn-slk_Latn,ogx_flores200-trans-eng_Latn-slv_Latn,ogx_flores200-trans-eng_Latn-spa_Latn,ogx_flores200-trans-eng_Latn-swe_Latn,ogx_flores200-trans-est_Latn-bul_Cyrl,ogx_flores200-trans-est_Latn-ces_Latn,ogx_flores200-trans-est_Latn-dan_Latn,ogx_flores200-trans-est_Latn-deu_Latn,ogx_flores200-trans-est_Latn-ell_Grek,ogx_flores200-trans-est_Latn-eng_Latn,ogx_flores200-trans-est_Latn-fin_Latn,ogx_flores200-trans-est_Latn-fra_Latn,ogx_flores200-trans-est_Latn-hun_Latn,ogx_flores200-trans-est_Latn-ita_Latn,ogx_flores200-trans-est_Latn-lit_Latn,ogx_flores200-trans-est_Latn-lvs_Latn,ogx_flores200-trans-est_Latn-nld_Latn,ogx_flores200-trans-est_Latn-pol_Latn,ogx_flores200-trans-est_Latn-por_Latn,ogx_flores200-trans-est_Latn-ron_Latn,ogx_flores200-trans-est_Latn-slk_Latn,ogx_flores200-trans-est_Latn-slv_Latn,ogx_flores200-trans-est_Latn-spa_Latn,ogx_flores200-trans-est_Latn-swe_Latn
FLORES200_TASKS_K_2=ogx_flores200-trans-fin_Latn-bul_Cyrl,ogx_flores200-trans-fin_Latn-ces_Latn,ogx_flores200-trans-fin_Latn-dan_Latn,ogx_flores200-trans-fin_Latn-deu_Latn,ogx_flores200-trans-fin_Latn-ell_Grek,ogx_flores200-trans-fin_Latn-eng_Latn,ogx_flores200-trans-fin_Latn-est_Latn,ogx_flores200-trans-fin_Latn-fra_Latn,ogx_flores200-trans-fin_Latn-hun_Latn,ogx_flores200-trans-fin_Latn-ita_Latn,ogx_flores200-trans-fin_Latn-lit_Latn,ogx_flores200-trans-fin_Latn-lvs_Latn,ogx_flores200-trans-fin_Latn-nld_Latn,ogx_flores200-trans-fin_Latn-pol_Latn,ogx_flores200-trans-fin_Latn-por_Latn,ogx_flores200-trans-fin_Latn-ron_Latn,ogx_flores200-trans-fin_Latn-slk_Latn,ogx_flores200-trans-fin_Latn-slv_Latn,ogx_flores200-trans-fin_Latn-spa_Latn,ogx_flores200-trans-fin_Latn-swe_Latn,ogx_flores200-trans-fra_Latn-bul_Cyrl,ogx_flores200-trans-fra_Latn-ces_Latn,ogx_flores200-trans-fra_Latn-dan_Latn,ogx_flores200-trans-fra_Latn-deu_Latn,ogx_flores200-trans-fra_Latn-ell_Grek,ogx_flores200-trans-fra_Latn-eng_Latn,ogx_flores200-trans-fra_Latn-est_Latn,ogx_flores200-trans-fra_Latn-fin_Latn,ogx_flores200-trans-fra_Latn-hun_Latn,ogx_flores200-trans-fra_Latn-ita_Latn,ogx_flores200-trans-fra_Latn-lit_Latn,ogx_flores200-trans-fra_Latn-lvs_Latn,ogx_flores200-trans-fra_Latn-nld_Latn,ogx_flores200-trans-fra_Latn-pol_Latn,ogx_flores200-trans-fra_Latn-por_Latn,ogx_flores200-trans-fra_Latn-ron_Latn,ogx_flores200-trans-fra_Latn-slk_Latn,ogx_flores200-trans-fra_Latn-slv_Latn,ogx_flores200-trans-fra_Latn-spa_Latn,ogx_flores200-trans-fra_Latn-swe_Latn,ogx_flores200-trans-hun_Latn-bul_Cyrl,ogx_flores200-trans-hun_Latn-ces_Latn,ogx_flores200-trans-hun_Latn-dan_Latn,ogx_flores200-trans-hun_Latn-deu_Latn,ogx_flores200-trans-hun_Latn-ell_Grek,ogx_flores200-trans-hun_Latn-eng_Latn,ogx_flores200-trans-hun_Latn-est_Latn,ogx_flores200-trans-hun_Latn-fin_Latn,ogx_flores200-trans-hun_Latn-fra_Latn,ogx_flores200-trans-hun_Latn-ita_Latn,ogx_flores200-trans-hun_Latn-lit_Latn,ogx_flores200-trans-hun_Latn-lvs_Latn,ogx_flores200-trans-hun_Latn-nld_Latn,ogx_flores200-trans-hun_Latn-pol_Latn,ogx_flores200-trans-hun_Latn-por_Latn,ogx_flores200-trans-hun_Latn-ron_Latn,ogx_flores200-trans-hun_Latn-slk_Latn,ogx_flores200-trans-hun_Latn-slv_Latn,ogx_flores200-trans-hun_Latn-spa_Latn,ogx_flores200-trans-hun_Latn-swe_Latn,ogx_flores200-trans-ita_Latn-bul_Cyrl,ogx_flores200-trans-ita_Latn-ces_Latn,ogx_flores200-trans-ita_Latn-dan_Latn,ogx_flores200-trans-ita_Latn-deu_Latn,ogx_flores200-trans-ita_Latn-ell_Grek,ogx_flores200-trans-ita_Latn-eng_Latn,ogx_flores200-trans-ita_Latn-est_Latn,ogx_flores200-trans-ita_Latn-fin_Latn,ogx_flores200-trans-ita_Latn-fra_Latn,ogx_flores200-trans-ita_Latn-hun_Latn,ogx_flores200-trans-ita_Latn-lit_Latn,ogx_flores200-trans-ita_Latn-lvs_Latn,ogx_flores200-trans-ita_Latn-nld_Latn,ogx_flores200-trans-ita_Latn-pol_Latn,ogx_flores200-trans-ita_Latn-por_Latn,ogx_flores200-trans-ita_Latn-ron_Latn,ogx_flores200-trans-ita_Latn-slk_Latn,ogx_flores200-trans-ita_Latn-slv_Latn,ogx_flores200-trans-ita_Latn-spa_Latn,ogx_flores200-trans-ita_Latn-swe_Latn,ogx_flores200-trans-lit_Latn-bul_Cyrl,ogx_flores200-trans-lit_Latn-ces_Latn,ogx_flores200-trans-lit_Latn-dan_Latn,ogx_flores200-trans-lit_Latn-deu_Latn,ogx_flores200-trans-lit_Latn-ell_Grek,ogx_flores200-trans-lit_Latn-eng_Latn,ogx_flores200-trans-lit_Latn-est_Latn,ogx_flores200-trans-lit_Latn-fin_Latn,ogx_flores200-trans-lit_Latn-fra_Latn,ogx_flores200-trans-lit_Latn-hun_Latn,ogx_flores200-trans-lit_Latn-ita_Latn,ogx_flores200-trans-lit_Latn-lvs_Latn,ogx_flores200-trans-lit_Latn-nld_Latn,ogx_flores200-trans-lit_Latn-pol_Latn,ogx_flores200-trans-lit_Latn-por_Latn,ogx_flores200-trans-lit_Latn-ron_Latn,ogx_flores200-trans-lit_Latn-slk_Latn,ogx_flores200-trans-lit_Latn-slv_Latn,ogx_flores200-trans-lit_Latn-spa_Latn,ogx_flores200-trans-lit_Latn-swe_Latn,ogx_flores200-trans-lvs_Latn-bul_Cyrl,ogx_flores200-trans-lvs_Latn-ces_Latn,ogx_flores200-trans-lvs_Latn-dan_Latn,ogx_flores200-trans-lvs_Latn-deu_Latn,ogx_flores200-trans-lvs_Latn-ell_Grek,ogx_flores200-trans-lvs_Latn-eng_Latn,ogx_flores200-trans-lvs_Latn-est_Latn,ogx_flores200-trans-lvs_Latn-fin_Latn,ogx_flores200-trans-lvs_Latn-fra_Latn,ogx_flores200-trans-lvs_Latn-hun_Latn,ogx_flores200-trans-lvs_Latn-ita_Latn,ogx_flores200-trans-lvs_Latn-lit_Latn,ogx_flores200-trans-lvs_Latn-nld_Latn,ogx_flores200-trans-lvs_Latn-pol_Latn,ogx_flores200-trans-lvs_Latn-por_Latn,ogx_flores200-trans-lvs_Latn-ron_Latn,ogx_flores200-trans-lvs_Latn-slk_Latn,ogx_flores200-trans-lvs_Latn-slv_Latn,ogx_flores200-trans-lvs_Latn-spa_Latn,ogx_flores200-trans-lvs_Latn-swe_Latn,ogx_flores200-trans-nld_Latn-bul_Cyrl,ogx_flores200-trans-nld_Latn-ces_Latn,ogx_flores200-trans-nld_Latn-dan_Latn,ogx_flores200-trans-nld_Latn-deu_Latn,ogx_flores200-trans-nld_Latn-ell_Grek,ogx_flores200-trans-nld_Latn-eng_Latn,ogx_flores200-trans-nld_Latn-est_Latn,ogx_flores200-trans-nld_Latn-fin_Latn,ogx_flores200-trans-nld_Latn-fra_Latn,ogx_flores200-trans-nld_Latn-hun_Latn,ogx_flores200-trans-nld_Latn-ita_Latn,ogx_flores200-trans-nld_Latn-lit_Latn,ogx_flores200-trans-nld_Latn-lvs_Latn,ogx_flores200-trans-nld_Latn-pol_Latn,ogx_flores200-trans-nld_Latn-por_Latn,ogx_flores200-trans-nld_Latn-ron_Latn,ogx_flores200-trans-nld_Latn-slk_Latn,ogx_flores200-trans-nld_Latn-slv_Latn,ogx_flores200-trans-nld_Latn-spa_Latn,ogx_flores200-trans-nld_Latn-swe_Latn
FLORES200_TASKS_K_3=ogx_flores200-trans-pol_Latn-bul_Cyrl,ogx_flores200-trans-pol_Latn-ces_Latn,ogx_flores200-trans-pol_Latn-dan_Latn,ogx_flores200-trans-pol_Latn-deu_Latn,ogx_flores200-trans-pol_Latn-ell_Grek,ogx_flores200-trans-pol_Latn-eng_Latn,ogx_flores200-trans-pol_Latn-est_Latn,ogx_flores200-trans-pol_Latn-fin_Latn,ogx_flores200-trans-pol_Latn-fra_Latn,ogx_flores200-trans-pol_Latn-hun_Latn,ogx_flores200-trans-pol_Latn-ita_Latn,ogx_flores200-trans-pol_Latn-lit_Latn,ogx_flores200-trans-pol_Latn-lvs_Latn,ogx_flores200-trans-pol_Latn-nld_Latn,ogx_flores200-trans-pol_Latn-por_Latn,ogx_flores200-trans-pol_Latn-ron_Latn,ogx_flores200-trans-pol_Latn-slk_Latn,ogx_flores200-trans-pol_Latn-slv_Latn,ogx_flores200-trans-pol_Latn-spa_Latn,ogx_flores200-trans-pol_Latn-swe_Latn,ogx_flores200-trans-por_Latn-bul_Cyrl,ogx_flores200-trans-por_Latn-ces_Latn,ogx_flores200-trans-por_Latn-dan_Latn,ogx_flores200-trans-por_Latn-deu_Latn,ogx_flores200-trans-por_Latn-ell_Grek,ogx_flores200-trans-por_Latn-eng_Latn,ogx_flores200-trans-por_Latn-est_Latn,ogx_flores200-trans-por_Latn-fin_Latn,ogx_flores200-trans-por_Latn-fra_Latn,ogx_flores200-trans-por_Latn-hun_Latn,ogx_flores200-trans-por_Latn-ita_Latn,ogx_flores200-trans-por_Latn-lit_Latn,ogx_flores200-trans-por_Latn-lvs_Latn,ogx_flores200-trans-por_Latn-nld_Latn,ogx_flores200-trans-por_Latn-pol_Latn,ogx_flores200-trans-por_Latn-ron_Latn,ogx_flores200-trans-por_Latn-slk_Latn,ogx_flores200-trans-por_Latn-slv_Latn,ogx_flores200-trans-por_Latn-spa_Latn,ogx_flores200-trans-por_Latn-swe_Latn,ogx_flores200-trans-ron_Latn-bul_Cyrl,ogx_flores200-trans-ron_Latn-ces_Latn,ogx_flores200-trans-ron_Latn-dan_Latn,ogx_flores200-trans-ron_Latn-deu_Latn,ogx_flores200-trans-ron_Latn-ell_Grek,ogx_flores200-trans-ron_Latn-eng_Latn,ogx_flores200-trans-ron_Latn-est_Latn,ogx_flores200-trans-ron_Latn-fin_Latn,ogx_flores200-trans-ron_Latn-fra_Latn,ogx_flores200-trans-ron_Latn-hun_Latn,ogx_flores200-trans-ron_Latn-ita_Latn,ogx_flores200-trans-ron_Latn-lit_Latn,ogx_flores200-trans-ron_Latn-lvs_Latn,ogx_flores200-trans-ron_Latn-nld_Latn,ogx_flores200-trans-ron_Latn-pol_Latn,ogx_flores200-trans-ron_Latn-por_Latn,ogx_flores200-trans-ron_Latn-slk_Latn,ogx_flores200-trans-ron_Latn-slv_Latn,ogx_flores200-trans-ron_Latn-spa_Latn,ogx_flores200-trans-ron_Latn-swe_Latn,ogx_flores200-trans-slk_Latn-bul_Cyrl,ogx_flores200-trans-slk_Latn-ces_Latn,ogx_flores200-trans-slk_Latn-dan_Latn,ogx_flores200-trans-slk_Latn-deu_Latn,ogx_flores200-trans-slk_Latn-ell_Grek,ogx_flores200-trans-slk_Latn-eng_Latn,ogx_flores200-trans-slk_Latn-est_Latn,ogx_flores200-trans-slk_Latn-fin_Latn,ogx_flores200-trans-slk_Latn-fra_Latn,ogx_flores200-trans-slk_Latn-hun_Latn,ogx_flores200-trans-slk_Latn-ita_Latn,ogx_flores200-trans-slk_Latn-lit_Latn,ogx_flores200-trans-slk_Latn-lvs_Latn,ogx_flores200-trans-slk_Latn-nld_Latn,ogx_flores200-trans-slk_Latn-pol_Latn,ogx_flores200-trans-slk_Latn-por_Latn,ogx_flores200-trans-slk_Latn-ron_Latn,ogx_flores200-trans-slk_Latn-slv_Latn,ogx_flores200-trans-slk_Latn-spa_Latn,ogx_flores200-trans-slk_Latn-swe_Latn,ogx_flores200-trans-slv_Latn-bul_Cyrl,ogx_flores200-trans-slv_Latn-ces_Latn,ogx_flores200-trans-slv_Latn-dan_Latn,ogx_flores200-trans-slv_Latn-deu_Latn,ogx_flores200-trans-slv_Latn-ell_Grek,ogx_flores200-trans-slv_Latn-eng_Latn,ogx_flores200-trans-slv_Latn-est_Latn,ogx_flores200-trans-slv_Latn-fin_Latn,ogx_flores200-trans-slv_Latn-fra_Latn,ogx_flores200-trans-slv_Latn-hun_Latn,ogx_flores200-trans-slv_Latn-ita_Latn,ogx_flores200-trans-slv_Latn-lit_Latn,ogx_flores200-trans-slv_Latn-lvs_Latn,ogx_flores200-trans-slv_Latn-nld_Latn,ogx_flores200-trans-slv_Latn-pol_Latn,ogx_flores200-trans-slv_Latn-por_Latn,ogx_flores200-trans-slv_Latn-ron_Latn,ogx_flores200-trans-slv_Latn-slk_Latn,ogx_flores200-trans-slv_Latn-spa_Latn,ogx_flores200-trans-slv_Latn-swe_Latn,ogx_flores200-trans-spa_Latn-bul_Cyrl,ogx_flores200-trans-spa_Latn-ces_Latn,ogx_flores200-trans-spa_Latn-dan_Latn,ogx_flores200-trans-spa_Latn-deu_Latn,ogx_flores200-trans-spa_Latn-ell_Grek,ogx_flores200-trans-spa_Latn-eng_Latn,ogx_flores200-trans-spa_Latn-est_Latn,ogx_flores200-trans-spa_Latn-fin_Latn,ogx_flores200-trans-spa_Latn-fra_Latn,ogx_flores200-trans-spa_Latn-hun_Latn,ogx_flores200-trans-spa_Latn-ita_Latn,ogx_flores200-trans-spa_Latn-lit_Latn,ogx_flores200-trans-spa_Latn-lvs_Latn,ogx_flores200-trans-spa_Latn-nld_Latn,ogx_flores200-trans-spa_Latn-pol_Latn,ogx_flores200-trans-spa_Latn-por_Latn,ogx_flores200-trans-spa_Latn-ron_Latn,ogx_flores200-trans-spa_Latn-slk_Latn,ogx_flores200-trans-spa_Latn-slv_Latn,ogx_flores200-trans-spa_Latn-swe_Latn,ogx_flores200-trans-swe_Latn-bul_Cyrl,ogx_flores200-trans-swe_Latn-ces_Latn,ogx_flores200-trans-swe_Latn-dan_Latn,ogx_flores200-trans-swe_Latn-deu_Latn,ogx_flores200-trans-swe_Latn-ell_Grek,ogx_flores200-trans-swe_Latn-eng_Latn,ogx_flores200-trans-swe_Latn-est_Latn,ogx_flores200-trans-swe_Latn-fin_Latn,ogx_flores200-trans-swe_Latn-fra_Latn,ogx_flores200-trans-swe_Latn-hun_Latn,ogx_flores200-trans-swe_Latn-ita_Latn,ogx_flores200-trans-swe_Latn-lit_Latn,ogx_flores200-trans-swe_Latn-lvs_Latn,ogx_flores200-trans-swe_Latn-nld_Latn,ogx_flores200-trans-swe_Latn-pol_Latn,ogx_flores200-trans-swe_Latn-por_Latn,ogx_flores200-trans-swe_Latn-ron_Latn,ogx_flores200-trans-swe_Latn-slk_Latn,ogx_flores200-trans-swe_Latn-slv_Latn,ogx_flores200-trans-swe_Latn-spa_Latn

# Define the arguments for each execution
TRUTHFULQA_BELEBELE_TASK_RUN="--model hf --model_args=pretrained="$MODEL_DIR",trust_remote_code=True --batch_size "auto" --tasks $TRUTHFULQA_TASKS,$BELEBELE_TASKS --output_path $OUTPUT_DIRECTORY"
HELLASWAG_FEWSHOT_RUN="--model hf --model_args=pretrained="$MODEL_DIR",trust_remote_code=True --num_fewshot 10  --batch_size "auto" --tasks $HELLASWAG_TASKS --output_path $OUTPUT_DIRECTORY"
ARC_FEWSHOT_RUN="--model hf --model_args=pretrained="$MODEL_DIR",trust_remote_code=True --num_fewshot 25 --batch_size "auto" --tasks $ARC_EASY_TASKS,$ARC_CHALLENGE_TASKS --output_path $OUTPUT_DIRECTORY"
FEWSHOT_TASKS_RUN="--model hf --model_args=pretrained="$MODEL_DIR",trust_remote_code=True --num_fewshot 5 --batch_size "auto" --tasks $MMLU_TASKS,$GSM8K_TASKS --output_path $OUTPUT_DIRECTORY"

FLORES_RUN_1="--model hf --model_args=pretrained="$MODEL_DIR",trust_remote_code=True --batch_size "auto" --tasks $FLORES200_TASKS_K_1 --output_path $OUTPUT_DIRECTORY"
FLORES_RUN_2="--model hf --model_args=pretrained="$MODEL_DIR",trust_remote_code=True --batch_size "auto" --tasks $FLORES200_TASKS_K_2 --output_path $OUTPUT_DIRECTORY"
FLORES_RUN_3="--model hf --model_args=pretrained="$MODEL_DIR",trust_remote_code=True --batch_size "auto" --tasks $FLORES200_TASKS_K_3 --output_path $OUTPUT_DIRECTORY"

# Run the script with the first set of arguments
accelerate launch "$SCRIPT" $TRUTHFULQA_BELEBELE_TASK_RUN

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Run Truthfulqa/Belebele failed to execute."
  exit 1
fi

# Run the script with the second set of arguments
accelerate launch "$SCRIPT" $HELLASWAG_FEWSHOT_RUN

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Run Hellaswag failed to execute."
  exit 1
fi

# Run the script with the third set of arguments
accelerate launch "$SCRIPT" $ARC_FEWSHOT_RUN

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Run ARC failed to execute."
  exit 1
fi

# Run the script with the fourth set of arguments
accelerate launch "$SCRIPT" $FEWSHOT_TASKS_RUN

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Run MMLU,GSM8K failed to execute."
  exit 1
fi

# Run the script with the fourth set of arguments
accelerate launch "$SCRIPT" $FLORES_RUN_1

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Run Flores 1 failed to execute."
  exit 1
fi

# Run the script with the fourth set of arguments
accelerate launch "$SCRIPT" $FLORES_RUN_2

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Run Flores 2 failed to execute."
  exit 1
fi

# Run the script with the fourth set of arguments
accelerate launch "$SCRIPT" $FLORES_RUN_3

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Run Flores 3 failed to execute."
  exit 1
fi