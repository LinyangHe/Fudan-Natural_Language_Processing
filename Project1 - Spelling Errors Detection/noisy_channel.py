import numpy as np
import sys

Index = {chr(i+ord('a')): i for i in range(26)}
Index['-'] = 26
Index['@'] = 27

Index['.'] = 28
Index[','] = 29
Index['_'] = 30
Index['\''] = 31

epsilon = sys.float_info.epsilon


Del = np.zeros((32, 32))
Ins = np.zeros((32, 32))
Sub = np.zeros((32, 32))
Rev = np.zeros((32, 32))

Char = np.zeros((32, 1))
Chars = np.zeros((32, 32))

Del_prob = np.zeros((32, 32))
Ins_prob = np.zeros((32, 32))
Sub_prob = np.zeros((32, 32))
Rev_prob = np.zeros((32, 32))


def process(typo, right_word):
    typo = '@'+typo.lower()
    right_word = '@'+right_word.lower()

    # Delete
    if len(typo) < len(right_word):
        for i in range(len(typo)):
            if typo[i] != right_word[i]:
                Del[Index[right_word[i-1]], Index[right_word[i]]] += 1
        return

    # Insert
    if len(typo) > len(right_word):
        for i in range(len(right_word)):
            if typo[i] != right_word[i]:
                Ins[Index[typo[i-1]], Index[typo[i]]] += 1
        return

    for i in range(len(typo)):
        if typo[i] != right_word[i]:
            # Substitution
            if i == len(typo)-1 or typo[i+1] == right_word[i+1]:
                Sub[Index[right_word[i]], Index[typo[i]]] += 1
                return
            # Reversal
            else:
                Rev[Index[right_word[i]], Index[typo[i]]] += 1
                return


def char_freq(word):
    length = len(word)
    for i, char in enumerate(word):
        Char[Index[char]] += 1
        if i < length-1:
            Chars[Index[word[i]], Index[word[i+1]]] += 1


def non_zero_product(x, y):
    if x == 0 or y == 0:
        return epsilon
    return x/y


def noisy_prob():
    for i in range(32):
        for j in range(32):
            Del_prob[i, j] = non_zero_product(Del[i, j], Chars[i, j])
            Ins_prob[i, j] = non_zero_product(Ins[i, j], Char[i])
            Sub_prob[i, j] = non_zero_product(Sub[i, j], Char[j])
            Rev_prob[i, j] = non_zero_product(Rev[i, j], Chars[i, j])


def run():
    # with open('wiki_spell.txt', encoding='utf-8') as spell_errors:
    with open('spell_errors.txt', encoding='utf-8') as spell_errors:
        for line in spell_errors:
            line = line.strip()
            typo = line.split('->')[0].lower()
            right_words = [i.lower() for i in line.split('->')[1].split(', ')]
            # char_freq('@'+typo)
            for word in right_words:
                char_freq('@' + word)
                process(typo, word)
    noisy_prob()
    Index.pop('.')
    Index.pop(',')
    Index.pop('_')
    Index.pop('\'')


if __name__ == '__main__':
    run()
