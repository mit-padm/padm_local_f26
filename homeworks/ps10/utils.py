import re
import numpy as np

# Function to load the dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    labels = []
    messages = []
    for line in lines:  # Skip the header
        label, sms = line.strip().split('\t')
        labels.append(label)
        messages.append(sms)
    return labels, messages

# Function to replace punctuation in text
def replace_punctuation(text):
    """
    Replaces all punctuation in the text with spaces.
    """
    return re.sub(r'[^\w\s]', ' ', text)

# Function to clean SMS messages
def clean_data(labels, messages):
    cleaned_labels = labels  # Labels don't need cleaning
    cleaned_messages = [replace_punctuation(msg).lower() for msg in messages]
    return cleaned_labels, cleaned_messages

# Function to shuffle the dataset
def shuffle_data(labels, messages, seed=1):
    np.random.seed(seed)
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    shuffled_labels = [labels[i] for i in indices]
    shuffled_messages = [messages[i] for i in indices]
    return shuffled_labels, shuffled_messages

# Function to split the dataset into training and test sets
def split_data(labels, messages, train_ratio=0.8):
    split_index = int(len(labels) * train_ratio)
    training_labels = labels[:split_index]
    training_messages = messages[:split_index]
    test_labels = labels[split_index:]
    test_messages = messages[split_index:]
    return (training_labels, training_messages), (test_labels, test_messages)


# ---------------------------------------------------------------------------
# Problem 3: an LLM-written Naive Bayes classifier.
# ---------------------------------------------------------------------------

# Problem 3 uses a fixed split of SMS_data. The training set is the file SMS_llm_train (the first 80% of
# the lines of SMS_data); the test set is every line of SMS_data after those, built at runtime.
# Weighted accuracy: losing a ham message costs twice as much as letting a spam message through.
SPAM_WEIGHT, HAM_WEIGHT = 1/3, 2/3

# Weighted accuracy of the Problem 1 classifier (k = 1) trained and tested on the Problem 3 split.
REFERENCE_WEIGHTED_ACCURACY = 0.837

LLM_MAX_WORDS = 200
_LLM_WORD_PATTERN = re.compile(r'[a-z0-9]+')


def load_llm_test_set(data_path='SMS_data', training_path='SMS_llm_train'):
    """Builds the Problem 3 test set: the (cleaned) messages of SMS_data that come after the training set."""
    all_labels, all_messages = load_data(data_path)
    training_labels, training_messages = load_data(training_path)
    num_training = len(training_labels)
    assert (all_labels[:num_training], all_messages[:num_training]) == (training_labels, training_messages), \
        f"{training_path} should be exactly the first {num_training} lines of {data_path}."
    return clean_data(all_labels[num_training:], all_messages[num_training:])


def parse_llm_classifier(text, training_messages):
    """
    Parses an LLM-written classifier file (see the Problem 3 format) and checks the format rules.

    Inputs
    ------
    text : the contents of the classifier file, as a string.
    training_messages : the (cleaned) Problem 3 training messages. Every word in the file must occur in one of them.

    Output
    ------
    A dict with keys
      'prior'  : P(C = spam),
      'unseen' : (P(unseen | spam), P(unseen | ham)),
      'words'  : a dict mapping each listed word to (P(word | spam), P(word | ham)).
    If the file breaks any rule, prints every problem found (by line number) and returns None instead.
    """
    vocabulary = {w for message in training_messages for w in message.split()}
    prior, unseen, words = None, None, {}
    seen_keys = set()
    errors = []

    def parse_probabilities(line_number, values, expected):
        if len(values) != expected:
            errors.append(f"line {line_number}: expected {expected} number(s), found {len(values)}")
            return None
        try:
            probabilities = [float(v) for v in values]
        except ValueError:
            errors.append(f"line {line_number}: could not read {' '.join(values)!r} as numbers")
            return None
        if not all(0 < p < 1 for p in probabilities):
            errors.append(f"line {line_number}: every probability must be strictly between 0 and 1")
            return None
        return probabilities

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#') or line.startswith('```'):
            continue
        key, *values = line.split()
        if key == 'prior':
            if 'prior' in seen_keys:
                errors.append(f"line {line_number}: 'prior' appears more than once")
            probabilities = parse_probabilities(line_number, values, 1)
            seen_keys.add('prior')
            if probabilities is not None:
                prior = probabilities[0]
        elif key == 'unseen':
            if 'unseen' in seen_keys:
                errors.append(f"line {line_number}: 'unseen' appears more than once")
            probabilities = parse_probabilities(line_number, values, 2)
            seen_keys.add('unseen')
            if probabilities is not None:
                unseen = tuple(probabilities)
        else:
            if not _LLM_WORD_PATTERN.fullmatch(key):
                errors.append(f"line {line_number}: {key!r} is not a word of lowercase letters and digits")
                continue
            if key not in vocabulary:
                errors.append(f"line {line_number}: {key!r} does not appear in the training data")
                continue
            if key in words:
                errors.append(f"line {line_number}: {key!r} appears more than once")
                continue
            probabilities = parse_probabilities(line_number, values, 2)
            if probabilities is not None:
                words[key] = tuple(probabilities)

    if 'prior' not in seen_keys:
        errors.append("missing a 'prior' line")
    if 'unseen' not in seen_keys:
        errors.append("missing an 'unseen' line")
    if len(words) > LLM_MAX_WORDS:
        errors.append(f"{len(words)} word lines; at most {LLM_MAX_WORDS} are allowed")
    if errors:
        print("Problems in the classifier file:\n  " + "\n  ".join(errors))
        return None
    return {'prior': prior, 'unseen': unseen, 'words': words}


def classify_with_llm_classifier(message, classifier):
    """Classifies a message as "spam" or "ham" with the same MAP rule as classify_message in 1C:
    every word in the message contributes its listed log-probability, or the 'unseen' one if it is not listed."""
    log_prob_spam = np.log(classifier['prior'])
    log_prob_ham = np.log(1 - classifier['prior'])
    unseen_spam, unseen_ham = classifier['unseen']
    for word in message.split():
        p_spam, p_ham = classifier['words'].get(word, (unseen_spam, unseen_ham))
        log_prob_spam += np.log(p_spam)
        log_prob_ham += np.log(p_ham)
    return "spam" if log_prob_spam > log_prob_ham else "ham"


def weighted_accuracy(true_labels, predicted_labels):
    """Weighted average of the fraction of spam classified as spam (weight 1/3)
    and the fraction of ham classified as ham (weight 2/3)."""
    true_labels, predicted_labels = np.array(true_labels), np.array(predicted_labels)
    spam_recall = np.mean(predicted_labels[true_labels == "spam"] == "spam")
    ham_recall = np.mean(predicted_labels[true_labels == "ham"] == "ham")
    return SPAM_WEIGHT * spam_recall + HAM_WEIGHT * ham_recall


def evaluate_llm_classifier(classifier, labels, messages):
    """Classifies every message, prints the results, and returns the weighted accuracy."""
    if classifier is None:
        print("No valid classifier to evaluate. Fix the problems reported when parsing the file.")
        return None
    predictions = np.array([classify_with_llm_classifier(m, classifier) for m in messages])
    labels = np.array(labels)
    num_spam, num_ham = np.sum(labels == "spam"), np.sum(labels == "ham")
    spam_correct = np.sum((labels == "spam") & (predictions == "spam"))
    ham_correct = np.sum((labels == "ham") & (predictions == "ham"))
    score = weighted_accuracy(labels, predictions)
    print(f"{len(classifier['words'])} words listed.")
    print(f"Spam classified as spam: {spam_correct} / {num_spam}")
    print(f"Ham classified as ham:   {ham_correct} / {num_ham}")
    print(f"Accuracy:          {np.mean(predictions == labels):.3f}")
    print(f"Weighted accuracy: {score:.3f} (the Problem 1 classifier with k = 1 scores {REFERENCE_WEIGHTED_ACCURACY} on the test set)")
    return score
