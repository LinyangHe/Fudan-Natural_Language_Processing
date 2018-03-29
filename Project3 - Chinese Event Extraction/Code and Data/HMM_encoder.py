import pickle

state_freq = {'O':0}
emission_freq = {'O':{},'Business':{},'Personnel':{},'Conflict':{},
				 'Movement':{},'Life':{},'Contact':{},'Transaction':{},'Justice':{}}

with open('trigger_train.txt',encoding = 'UTF-8') as trigger_train_file:
	for line in trigger_train_file:
		line = line.strip()
		if not line:
			continue
		line = line.split('\t')
		if line[1] == 'O':
			state_freq['O'] += 1
			try:
				emission_freq['O'][line[0]] += 1
			except KeyError:
				emission_freq['O'][line[0]] = 1
		else:
			type = line[1][2:]
			try:
				state_freq[type] += 1
			except KeyError:
				state_freq[type] = 0
			try:
				emission_freq[type][line[0]] += 1
			except KeyError:
				emission_freq[type][line[0]] = 1 

pickle.dump([state_freq,emission_freq], open('trigger_HMM_encode.txt', 'wb'))


'''
For the second task
'''
argu_state_freq = {}
argu_emission_freq = {}

with open('argument_train.txt',encoding = 'UTF-8') as trigger_train_file:
	last_type = ''
	start_flag = 1
	for line in trigger_train_file:
		line = line.strip()
		if not line:
			start_flag = 1
			continue
		line = line.split('\t')
		current_type = line[1]
		'''
		Compute the emission freq
		'''
		try:
			argu_emission_freq[current_type][line[0]] += 1
		except KeyError:
			try:
				argu_emission_freq[current_type][line[0]] = 1
			except KeyError:
				argu_emission_freq[current_type] = {}

		'''
		Compte the state transition
		'''
		if start_flag == 1:
			start_flag = 0
			last_type = current_type
			continue
		try:
			argu_state_freq[last_type][current_type] += 1
		except KeyError:
			try:
				argu_state_freq[last_type][current_type] = 1
			except KeyError:
				argu_state_freq[last_type] = {}

pickle.dump([argu_state_freq,argu_emission_freq], open('argu_HMM_encode.txt', 'wb'))