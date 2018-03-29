
def decoder(test_file_name, output_file_name, task, method, state_transition_prob, emission_prob):
    initial_state_probs = {}
    for state in emission_prob:
        initial_state_probs[state] = 1.0
    if method == 'viterbi':
        viterbi_decode(test_file_name, output_file_name, task,
                       state_transition_prob, emission_prob, initial_state_probs)
    if method == 'greedy':
        greedy_decode(test_file_name, output_file_name, task,
                      state_transition_prob, emission_prob, initial_state_probs)


def greedy_decode(test_file_name, output_file_name, task, state_transition_prob, emission_prob, initial_state_probs):
    output_file = open(output_file_name, 'w', encoding='UTF-8')
    last_state_probs = initial_state_probs
    last_state = 'O'
    is_trigger = 0
    max_prob_current_state = 'O'
    with open(test_file_name, encoding='UTF-8') as test_file:
        start_flag = 1
        for line in test_file:
            line = line.strip()
            if not line:
                output_file.write('\n')
                start_flag = 1
                last_state_probs = initial_state_probs
                last_state = 'O'
                is_trigger = 0
                continue
            output_file.write(line)
            if is_trigger:
                output_file.write('\tO\n')
                continue

            line = line.split('\t')
            current_state_probs = {}
            for current_state in initial_state_probs:
                try:
                    current_emission_prob = emission_prob[
                        current_state][line[0]]
                    if not last_state_probs:
                        last_state_probs = initial_state_probs
                    current_state_probs[current_state] = max(last_state_probs[
                                                             i] for i in last_state_probs) * state_transition_prob[last_state][current_state] * current_emission_prob

                except KeyError:
                    pass

            # print(current_state_probs)
            if not current_state_probs:
                current_state_max_prob = max(
                    state_transition_prob[last_state].values())
                current_state = [i for i in state_transition_prob[
                    last_state] if state_transition_prob[last_state][i] == current_state_max_prob][0]
            else:
                max_prob = max(current_state_probs.values())
                max_prob_current_state = [
                    state for state in current_state_probs if current_state_probs[state] == max_prob][0]

            if max_prob_current_state == 'O':
                output_file.write('\t'+max_prob_current_state+'\n')
            elif task == 'trigger':
                output_file.write('\tT_'+max_prob_current_state+'\n')
                is_trigger = 1
            else:
                output_file.write('\t'+max_prob_current_state+'\n')

            last_state_probs = current_state_probs
            # print(max_prob_current_state,'&&&&&&&&&&&&')
            last_state = max_prob_current_state

    output_file.close()


def viterbi_decode(test_file_name, output_file_name, task, state_transition_prob, emission_prob, initial_state_probs):
    output_file = open(output_file_name, 'w', encoding='UTF-8')
    last_state_probs = initial_state_probs
    last_state = 'O'
    is_trigger = 0
    max_prob_current_state = 'O'
    with open(test_file_name, encoding='UTF-8') as test_file:
        start_flag = 1
        for line in test_file:
            line = line.strip()
            if not line:
                output_file.write('\n')
                start_flag = 1
                last_state_probs = initial_state_probs
                last_state = 'O'
                is_trigger = 0
                continue
            output_file.write(line)
            if is_trigger:
                output_file.write('\tO\n')
                continue

            line = line.split('\t')
            current_state_probs = {}
            for current_state in initial_state_probs:
                try:
                    current_emission_prob = emission_prob[
                        current_state][line[0]]
                    if not last_state_probs:
                        last_state_probs = initial_state_probs
                    current_state_probs[current_state] = max(last_state_probs[
                                                             i]*state_transition_prob[i][current_state] for i in last_state_probs) * current_emission_prob
                except KeyError:
                    pass

            # print(current_state_probs)
            if not current_state_probs:
                current_state_max_prob = max(
                    state_transition_prob[last_state].values())
                current_state = [i for i in state_transition_prob[
                    last_state] if state_transition_prob[last_state][i] == current_state_max_prob][0]
            else:
                max_prob = max(current_state_probs.values())
                max_prob_current_state = [
                    state for state in current_state_probs if current_state_probs[state] == max_prob][0]

            if max_prob_current_state == 'O':
                output_file.write('\t'+max_prob_current_state+'\n')
            elif task == 'trigger':
                output_file.write('\tT_'+max_prob_current_state+'\n')
                is_trigger = 1
            else:
                output_file.write('\t'+max_prob_current_state+'\n')

            last_state_probs = current_state_probs
            # print(max_prob_current_state,'&&&&&&&&&&&&')
            last_state = max_prob_current_state

    output_file.close()
