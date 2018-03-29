read_file1 = open('spell-errors_1.txt','r')
write_file1 = open('spell_errors.txt','w')

wrong_words = {}
for line in read_file1:
	line = line.strip()
	right_word = line.split(': ')[0]
	wrong_words = line.split(': ')[1].split(', ')
	for i in range(len(wrong_words)):
		times = 1
		if '*' in wrong_words[i]:
			times = wrong_words[i].split('*')[1]
			wrong_words[i] = wrong_words[i].split('*')[0]
		for j in range(int(times)):
			write_file1.write(right_word+'->'+wrong_words[i]+'\n')

read_file2 = open('testdata.txt')
read_file3 = open('ans.txt')
write_file2 = open('pair.txt','w')

data1 = []
data2 = []
for line1 in read_file2:
	line1 = line1.strip().split('\t')[2].split(' ')
	data1.append(line1)
for line2 in read_file3:
	line2 = line2.strip().split('\t')[1].split(' ')
	data2.append(line2)

for i in range(1000):
	write_file2.write(str(i+1) +'\t')
	for j in range(len(data1[i])):
		if data1[i][j] != data2[i][j]:
			write_file2.write(' ' + data1[i][j] + ' ' + data2[i][j])
	write_file2.write('\n')

def edit(str1, str2):  
      
    matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]  
  
    for i in range(1,len(str1)+1):  
        for j in range(1,len(str2)+1):  
            if str1[i-1] == str2[j-1]:  
                d = 0  
            else:  
                d = 1  
            matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)  
  
  
    return matrix[len(str1)][len(str2)]
