import pickle
import HMM_decoder as DE


def trigger_train():
    zero_prob = {'O': 0, 'Business': 0, 'Personnel': 0, 'Conflict': 0,
                 'Movement': 0, 'Life': 0, 'Contact': 0, 'Transaction': 0, 'Justice': 0}
    '''
	We have counted the frequency in the HMM_encoder and used the pickle tool 
	to dump the data into a file. Every time we test different models, we don't
	need to count the frequency again, we just need to load the dumped file. 
	'''
    trigger_state_freq, trigger_emission_freq = pickle.load(
        open('trigger_HMM_encode.txt', 'rb'))
    '''
	First we compute the state transition probabolity when the current state is O. When current
	state is not 'O', the next state will always be 'O' according to the training data, so we 
	don't need to compute this kind of state transition prob.
	'''
    total_freq = sum(trigger_state_freq.values())
    state_transition_prob_O = {}
    for state in trigger_state_freq:
        state_transition_prob_O[state] = trigger_state_freq[state]/total_freq

    state_transition_prob = {'O': state_transition_prob_O}
    for state in zero_prob:
        if state == 'O':
            continue
        state_transition_prob[state] = zero_prob

    '''
	Second we compute the emission probobality of each state.
	'''
    emission_prob = {'O': {}, 'Business': {}, 'Personnel': {}, 'Conflict': {},
                     'Movement': {}, 'Life': {}, 'Contact': {}, 'Transaction': {}, 'Justice': {}}
    for state in trigger_emission_freq:
        state_total_freq = trigger_state_freq[state]
        emission_dict = trigger_emission_freq[state]
        for emission in emission_dict:
            emission_prob[state][emission] = emission_dict[
                emission]/state_total_freq

    return state_transition_prob, emission_prob


def trigger_test(state_transition_prob, emission_prob):
    '''
    We will use the HMM_decoder file to do sequence labeling. We provide several methods
    to find the hidden state of the HMM, including Viterbi algorithm and greedy algorithms.
    And the task parameter means trigger subtask or arguments subtask.
    '''
    DE.decoder(test_file_name='trigger_test.txt', output_file_name='trigger_result.txt',
               task='trigger', method='viterbi', state_transition_prob=state_transition_prob, emission_prob=emission_prob)


def argu_train():
    argu_state_freq, argu_emission_freq = pickle.load(
        open('argu_HMM_encode.txt', 'rb'))
    state_transition_prob, emission_prob = argu_state_freq, argu_emission_freq

    for state in argu_state_freq:
        state_total_freq = sum(argu_state_freq[state].values())
        transition_dict = argu_state_freq[state]
        for transition in transition_dict:
            state_transition_prob[state][transition] = transition_dict[
                transition]/state_total_freq

    initial_state_probs = {}
    for state in emission_prob:
        initial_state_probs[state] = 0.0

    new_state_transition_prob = {
        i: initial_state_probs.copy() for i in initial_state_probs}

    for state in state_transition_prob:
        for transition in state_transition_prob[state]:
            new_state_transition_prob[state][
                transition] = state_transition_prob[state][transition]

    for state in argu_emission_freq:
        state_total_freq = sum(argu_emission_freq[state].values())
        emission_dict = argu_emission_freq[state]
        for emission in emission_dict:
            emission_prob[state][emission] = emission_dict[
                emission]/state_total_freq
    return state_transition_prob, emission_prob, new_state_transition_prob


def argu_test(state_transition_prob, emission_prob):
    DE.decoder(test_file_name='argument_test.txt', output_file_name='argument_result.txt',
               task='argu', method='viterbi', state_transition_prob=state_transition_prob, emission_prob=emission_prob)

if __name__ == '__main__':
    state_transition_prob, emission_prob = trigger_train()
    trigger_test(state_transition_prob, emission_prob)

    # state_transition_prob, emission_prob, new = argu_train()
    # argu_test(new, emission_prob)
