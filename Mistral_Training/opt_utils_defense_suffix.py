import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import openai
from tqdm import tqdm
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from collections import defaultdict, OrderedDict
from string_utils_HGA_self_reminder_suffix import autodan_PrefixManager
import sys
import time


def forward(*, model, input_ids, attention_mask, batch_size=512):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i + batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **kwargs
    ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

### HGA ###
def construct_momentum_word_dict(word_dict, individuals, score_list, K):
    # Initialize a dictionary to store scores for each word
    word_scores = {}
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    stop_words = set(stopwords.words('english'))
    # Iterate over each individual and their corresponding score
    for individual, score in zip(individuals, score_list):
        # Extract words from the individual, excluding filtered words
        words = [word for word in individual if word.lower() not in stop_words and word.lower() not in T]  # Assuming 'filtered' is defined elsewhere

        # Add the score to the corresponding word in word_scores
        for word in words:
            if word in word_scores:
                word_scores[word].append(score)
            else:
                word_scores[word] = [score]

    # Calculate the average score for each word and apply momentum update
    for word, scores in word_scores.items():
        avg_score = sum(scores) / len(scores)
        
        # Update the word score with momentum if the word exists in word_dict
        if word in word_dict:
            word_dict[word] = (word_dict[word] + avg_score) / 2
        else:
            word_dict[word] = avg_score

    # Sort the word_dict based on the scores in descending order and return top K items
    sorted_word_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
    
    return dict(list(sorted_word_dict.items())[:K])



def replace_with_synonym(prompt, word_dict):
    words = prompt.split()

    for i, word in enumerate(words):
        # Assuming the values in word_dict are now lists of synonyms
        synonyms_words_l = wordnet.synsets(word)
        synonyms_words_names = [synonyms_words.lemma_names() for synonyms_words in synonyms_words_l]
        synonyms_words_names = set(sum(synonyms_words_names, [])) #flatten a list
        synonyms=[]
        #pick out the synonyms that exists in the word dict
        for syn_word in synonyms_words_names:
            if syn_word in word_dict:
                synonyms.append((syn_word, word_dict[syn_word]))
        #synonyms = word_dict[word]
        #print(synonyms)
        if synonyms:
            total_score = sum([score for _, score in synonyms])
            #random_value = random.uniform(0, total_score)
            #cumulative_score = 0
    
            for synonym, score in synonyms:
                
                prob = score/(total_score + 1e-8)
                #cumulative_score += score
                if random.random() < prob:
                    words[i] = synonym  # Assuming synonym is a tuple (word, score)
                    break

    return ' '.join(words)

def apply_word_momentum_exchange(selected_control_prefixes, score_list, num_elites, init_word_dict):
    new_prefixes = []
    individuals=[]
    for i in range(0, len(selected_control_prefixes)):
        selected_prefix = selected_control_prefixes[i]
        individuals.append(selected_prefix.split())
    assert (len(individuals)==len(selected_control_prefixes))
    #apply to the momentum function
    momentum_word_dict = construct_momentum_word_dict(init_word_dict, individuals, score_list, 30)
    #print(momentum_word_dict)
    for j in range(0, len(selected_control_prefixes)):
        selected_prefix = selected_control_prefixes[j]
        new_prefix = replace_with_synonym(selected_prefix, momentum_word_dict)
        
        new_prefixes.append(new_prefix)
    return new_prefixes, momentum_word_dict
        
        

def autodan_hga_sample_control(selected_control_prefixes, score_list, num_elites, batch_size, word_dict):
    score_list = [-x for x in score_list]
    # Step 1: Sort the score_list and get corresponding control_prefixes
    sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
    sorted_control_prefixes = [selected_control_prefixes[i] for i in sorted_indices]

    elites = sorted_control_prefixes[:num_elites]

    # Step 3: Use roulette wheel selection for the remaining positions
    parents_list = roulette_wheel_selection(sorted_control_prefixes, score_list, batch_size - num_elites, if_softmax=True)

    # Combine elites with the mutated offspring
    
    new_prefixes, new_word_dict = apply_word_momentum_exchange(parents_list, sorted_indices, num_elites, word_dict)
    next_generation = elites + new_prefixes
    assert len(next_generation)==batch_size
    return next_generation, new_word_dict

    
    
### GA ###
def autodan_sample_control(control_prefixes, score_list, num_elites, batch_size, crossover=0.5,
                           num_points=5, mutation=0.01, API_key=None, reference=None, if_softmax=True, if_api=True):
    score_list = [-x for x in score_list]
    # Step 1: Sort the score_list and get corresponding control_prefixes
    sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
    sorted_control_prefixes = [control_prefixes[i] for i in sorted_indices]

    # Step 2: Select the elites
    elites = sorted_control_prefixes[:num_elites]

    # Step 3: Use roulette wheel selection for the remaining positions
    parents_list = roulette_wheel_selection(control_prefixes, score_list, batch_size - num_elites, if_softmax)

    # Step 4: Apply crossover and mutation to the selected parents
    mutated_offspring = apply_crossover_and_mutation(parents_list, crossover_probability=crossover,
                                                     num_points=num_points,
                                                     mutation_rate=mutation, API_key=API_key, reference=reference,
                                                     if_api=if_api)

    # Combine elites with the mutated offspring
    next_generation = elites + mutated_offspring

    assert len(next_generation) == batch_size
    return next_generation


def roulette_wheel_selection(data_list, score_list, num_selected, if_softmax=True):
    if if_softmax:
        selection_probs = np.exp(score_list - np.max(score_list))
        selection_probs = selection_probs / selection_probs.sum()
    else:
        total_score = sum(score_list)
        selection_probs = [score / total_score for score in score_list]

    selected_indices = np.random.choice(len(data_list), size=num_selected, p=selection_probs, replace=True)

    selected_data = [data_list[i] for i in selected_indices]
    return selected_data


def apply_crossover_and_mutation(selected_data, crossover_probability=0.5, num_points=3, mutation_rate=0.01,
                                 API_key=None,
                                 reference=None, if_api=True):
    offspring = []

    for i in range(0, len(selected_data), 2):
        parent1 = selected_data[i]
        parent2 = selected_data[i + 1] if (i + 1) < len(selected_data) else selected_data[0]

        if random.random() < crossover_probability:
            child1, child2 = crossover(parent1, parent2, num_points)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    mutated_offspring = apply_gpt_mutation(offspring, mutation_rate, API_key, reference, if_api)

    return mutated_offspring


def crossover(str1, str2, num_points):
    sentences1 = [s for s in re.split('(?<=[.!?])\s+', str1) if s]
    sentences2 = [s for s in re.split('(?<=[.!?])\s+', str2) if s]

    max_swaps = min(len(sentences1), len(sentences2))-1
    num_swaps = min(num_points, max_swaps)

    #print("crossover info")
    #print(num_swaps)
    #print(max_swaps)
    #print(sentences1)
    #print(sentences2)
    '''
    if num_swaps==max_swaps:
        num_swaps=max_swaps-1
    else:
        num_swaps=num_swaps
    '''
    swap_indices = sorted(random.sample(range(1, max_swaps), num_swaps))
    

    new_str1, new_str2 = [], []
    last_swap = 0
    for swap in swap_indices:
        if random.choice([True, False]):
            new_str1.extend(sentences1[last_swap:swap])
            new_str2.extend(sentences2[last_swap:swap])
        else:
            new_str1.extend(sentences2[last_swap:swap])
            new_str2.extend(sentences1[last_swap:swap])
        last_swap = swap

    if random.choice([True, False]):
        new_str1.extend(sentences1[last_swap:])
        new_str2.extend(sentences2[last_swap:])
    else:
        new_str1.extend(sentences2[last_swap:])
        new_str2.extend(sentences1[last_swap:])

    return ' '.join(new_str1), ' '.join(new_str2)


def gpt_mutate(sentence, API_key=None):
    openai.api_key = API_key
    system_msg = 'You are a helpful and creative assistant who writes well.'
    user_message = f"Please revise the following sentence with no changes to its length nor the punctuations and only output the revised version, the sentences are: \n '{sentence}'."
    revised_sentence = sentence
    received = False
    while not received:
        try:
            response = openai.ChatCompletion.create(model="gpt-4",
                                                    messages=[{"role": "system", "content": system_msg},
                                                              {"role": "user", "content": user_message}],
                                                    temperature=1, top_p=0.9)
            revised_sentence = response["choices"][0]["message"]["content"].replace('\n', '')
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:  # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError, Prompt error.")
                return None
            if error == AssertionError:
                print("Assert error:", sys.exc_info()[1])  # assert False
            else:
                print("API error:", error)
            time.sleep(1)
    if revised_sentence.startswith("'") or revised_sentence.startswith('"'):
        revised_sentence = revised_sentence[1:]
    if revised_sentence.endswith("'") or revised_sentence.endswith('"'):
        revised_sentence = revised_sentence[:-1]
    if revised_sentence.endswith("'.") or revised_sentence.endswith('".'):
        revised_sentence = revised_sentence[:-2]
    print(f'revised: {revised_sentence}')
    return revised_sentence


def apply_gpt_mutation(offspring, mutation_rate=0.01, API_key=None, reference=None, if_api=True):
    if if_api:
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                if API_key is None:
                    offspring[i] = random.choice(reference[len(offspring):])
                else:
                    offspring[i] = gpt_mutate(offspring[i], API_key)
    else:
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = replace_with_synonyms(offspring[i])
    return offspring


def apply_init_gpt_mutation(offspring, mutation_rate=0.01, API_key=None, if_api=True):
    for i in tqdm(range(len(offspring)), desc='initializing...'):
        if if_api:
            if random.random() < mutation_rate:
                offspring[i] = gpt_mutate(offspring[i], API_key)
                #print(offspring[i])
        else:
            if random.random() < mutation_rate:
                offspring[i] = replace_with_synonyms(offspring[i])
                #print(offspring[i])
    return offspring

def replace_with_synonyms(sentence, num=10):
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(sentence)
    uncommon_words = [word for word in words if word.lower() not in stop_words and word.lower() not in T]
    selected_words = random.sample(uncommon_words, min(num, len(uncommon_words)))
    for word in selected_words:
        synonyms = wordnet.synsets(word)
        if synonyms and synonyms[0].lemmas():
            synonym = synonyms[0].lemmas()[0].name()
            sentence = sentence.replace(word, synonym, 1)
    return sentence

def get_score_autodan(tokenizer, conv_template, conv_template_helpful, instruction, target, helpful_instruction, helpful_target, model, device, test_controls=None, crit=None, alpha=10):
    # Convert all test_controls to token ids and find the max length
    input_ids_list = []
    input_ids_list_helpful = []
    target_slices = []
    target_slices_helpful=[]
    for item in test_controls:
        #refusal
        prefix_manager = autodan_PrefixManager(tokenizer=tokenizer,
                                               conv_template=conv_template,
                                               instruction=instruction,
                                               target=target,
                                               adv_string=item)
        
        
        input_ids = prefix_manager.get_input_ids(adv_string=item).to(device)
        input_ids_list.append(input_ids)
        target_slices.append(prefix_manager._target_slice)

        #helpful
        prefix_manager_helpful = autodan_PrefixManager(tokenizer=tokenizer,
                                               conv_template=conv_template_helpful,
                                               instruction=helpful_instruction,
                                               target=helpful_target,
                                               adv_string=item)
        input_ids_helpful = prefix_manager_helpful.get_input_ids(adv_string=item).to(device)
        input_ids_list_helpful.append(input_ids_helpful)
        target_slices_helpful.append(prefix_manager_helpful._target_slice)

    # Pad all token ids to the max length
    pad_tok = 0
    for ids in input_ids_list:
        while pad_tok in ids:
            pad_tok += 1


    pad_tok_helpful = 0
    for ids in input_ids_list_helpful:
        while pad_tok_helpful in ids:
            pad_tok_helpful += 1

    # Find the maximum length of input_ids in the list
    max_input_length = max([ids.size(0) for ids in input_ids_list])

    max_input_length_helpful = max([ids.size(0) for ids in input_ids_list_helpful])

    # Pad each input_ids tensor to the maximum length
    padded_input_ids_list = []
    for ids in input_ids_list:
        pad_length = max_input_length - ids.size(0)
        padded_ids = torch.cat([ids, torch.full((pad_length,), pad_tok, device=device)], dim=0)
        padded_input_ids_list.append(padded_ids)

    padded_input_ids_list_helpful = []
    for ids in input_ids_list_helpful:
        pad_length_helpful = max_input_length_helpful - ids.size(0)
        padded_ids_helpful = torch.cat([ids, torch.full((pad_length_helpful,), pad_tok_helpful, device=device)], dim=0)
        padded_input_ids_list_helpful.append(padded_ids_helpful)

    # Stack the padded input_ids tensors
    input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)

    input_ids_tensor_helpful = torch.stack(padded_input_ids_list_helpful, dim=0)
    
    attn_mask = (input_ids_tensor != pad_tok).type(input_ids_tensor.dtype)

    attn_mask_helpful = (input_ids_tensor_helpful != pad_tok_helpful).type(input_ids_tensor_helpful.dtype)

    # Forward pass and compute loss
    logits = forward(model=model, input_ids=input_ids_tensor, attention_mask=attn_mask, batch_size=len(test_controls))

    logits_helpful = forward(model=model, input_ids=input_ids_tensor_helpful, attention_mask=attn_mask_helpful, batch_size=len(test_controls))
    
    losses = []
    for idx, target_slice in enumerate(target_slices):
        loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
        logits_slice = logits[idx, loss_slice, :].unsqueeze(0).transpose(1, 2)
        targets = input_ids_tensor[idx, target_slice].unsqueeze(0)
        loss = crit(logits_slice, targets)
        losses.append(loss)

    losses_helpful = []
    for idx, target_slice_helpful in enumerate(target_slices_helpful):
        loss_slice_helpful = slice(target_slice_helpful.start - 1, target_slice_helpful.stop - 1)
        logits_slice_helpful = logits_helpful[idx, loss_slice_helpful, :].unsqueeze(0).transpose(1, 2)
        targets_helpful = input_ids_tensor_helpful[idx, target_slice_helpful].unsqueeze(0)
        loss_helpful = crit(logits_slice_helpful, targets_helpful)
        losses_helpful.append(loss_helpful)
    assert(len(losses)==len(losses_helpful))

    final_losses = []
    for i in range(len(losses)):
        final_loss =alpha *losses[i] +  losses_helpful[i]
        final_losses.append(final_loss)
    

    del input_ids_list, target_slices, input_ids_tensor, attn_mask, input_ids_list_helpful, target_slices_helpful, input_ids_tensor_helpful, attn_mask_helpful
    gc.collect()
    return torch.stack(final_losses)


def get_score_autodan_low_memory(tokenizer, conv_template, conv_template_helpful, instruction, target, helpful_instruction, helpful_target, model, device, test_controls=None,
                                 crit=None, alpha=1):
    losses = []
    for item in test_controls:
        prefix_manager = autodan_PrefixManager(tokenizer=tokenizer,
                                               conv_template=conv_template,
                                               instruction=instruction,
                                               target=target,
                                               adv_string=item)
        input_ids = prefix_manager.get_input_ids(adv_string=item).to(device)
        input_ids_tensor = torch.stack([input_ids], dim=0)

        #helpful
        prefix_manager_helpful = autodan_PrefixManager(tokenizer=tokenizer,
                                               conv_template=conv_template_helpful,
                                               instruction=helpful_instruction,
                                               target=helpful_target,
                                               adv_string=item)
        input_ids_helpful = prefix_manager_helpful.get_input_ids(adv_string=item).to(device)
        input_ids_tensor_helpful = torch.stack([input_ids_helpful], dim=0)
    
        # Forward pass and compute loss
        logits = forward(model=model, input_ids=input_ids_tensor, attention_mask=None, batch_size=len(test_controls))
        logits_helpful = forward(model=model, input_ids=input_ids_tensor_helpful, attention_mask=None, batch_size=len(test_controls))
        
        target_slice = prefix_manager._target_slice
        loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
        logits_slice = logits[0, loss_slice, :].unsqueeze(0).transpose(1, 2)
        targets = input_ids_tensor[0, target_slice].unsqueeze(0)
        loss = crit(logits_slice, targets)

        target_slice_helpful = prefix_manager_helpful._target_slice
        loss_slice_helpful = slice(target_slice_helpful.start - 1, target_slice_helpful.stop - 1)
        logits_slice_helpful = logits_helpful[0, loss_slice_helpful, :].unsqueeze(0).transpose(1, 2)
        targets_helpful = input_ids_tensor_helpful[0, target_slice_helpful].unsqueeze(0)
        loss_helpful = crit(logits_slice_helpful, targets_helpful)
        total_loss = alpha*loss+loss_helpful
        losses.append(total_loss)

    del input_ids_tensor, input_ids_tensor_helpful
    gc.collect()
    return torch.stack(losses)
