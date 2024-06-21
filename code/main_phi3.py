# Imports
import pandas as pd
from transformers import pipeline
import sacrebleu
from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')

# Instantiate Pipeline
model_path = "Essacheez/Phi-3-mini-4k-instruct-finetune-translation-10k-alpaca-style"
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=200)

#pipe = pipeline("text-generation", model="Essacheez/Phi-3-mini-4k-instruct-finetune-translation-10k-alpaca-style", trust_remote_code=True, max_new_tokens=200)



# File Paths
input_file_path = "data/Translation_InTheDark_Book/output_sentences_all.xlsx"
output_file_path = "results/Translations/translation_results_book_phi3.tsv"
scores_file_path = "results/TranslationScore/translation_scores_book_phi3.txt"

# Read Testdata
data = pd.read_excel(input_file_path, header=None)
data = data.drop(index=0)
data.columns = ['English', 'German']

print(data.head())
grouped_data = data.groupby('English')['German'].apply(list).reset_index()

# Define Function to put Sentences into the pipeline and store the results
def translate_sentences(data):
    results = []
    for _, row in data.iterrows():
        print("working")
        english_sentence = row['English']
        expected_german_sentences = row['German']
        
        translated_sentence = pipe(english_sentence)[0]['translation_text']
        
        results.append({
            "English": english_sentence,
            "Expected German": expected_german_sentences,
            "Translated German": translated_sentence
        })
    
    return results

# Execution of Translation
results = translate_sentences(grouped_data)
results_df = pd.DataFrame(results)
results_df.to_csv(output_file_path, sep='\t', index=False)

print(results_df.head())

# Calculate BLEU score
references = [row['German'] for _, row in grouped_data.iterrows()]
hypotheses = [result['Translated German'] for result in results]

bleu_score = sacrebleu.corpus_bleu(hypotheses, references)
print(f"BLEU score: {bleu_score.score}")

#Tokenize Words
tokenized_references = [[word_tokenize(sent) for sent in ref] for ref in references]
tokenized_hypotheses = [word_tokenize(hyp) for hyp in hypotheses]

# Calculate NLTK BLEU score
smoothing_function = SmoothingFunction().method1
nltk_bleu_scores = []
for hyp, refs in zip(tokenized_hypotheses, tokenized_references):
    score = sentence_bleu(refs, hyp, smoothing_function=smoothing_function)
    nltk_bleu_scores.append(score)

average_nltk_bleu_score = sum(nltk_bleu_scores) / len(nltk_bleu_scores)
print(f"NLTK BLEU score: {average_nltk_bleu_score}")

# Calculate METEOR score
meteor_scores = []
for hyp, ref in zip(tokenized_hypotheses, tokenized_references):
    score = nltk.translate.meteor_score.meteor_score(ref, hyp)
    meteor_scores.append(score)

average_meteor_score = sum(meteor_scores) / len(meteor_scores)
print(f"METEOR score: {average_meteor_score}")

# Save Scores to File
with open(scores_file_path, 'w') as f:
    f.write(f"BLEU score: {bleu_score.score}\n")
    f.write(f"NLTK BLEU score: {average_nltk_bleu_score}\n")
    f.write(f"METEOR score: {average_meteor_score}\n")
