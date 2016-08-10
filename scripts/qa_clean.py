import os

dir = '../data/QA/test/'
a_file = open(dir+'a.toks', 'r')
b_file = open(dir+'b.toks', 'r')
id_file = open(dir+'id.txt', 'r')
sim_file = open(dir+'sim.txt', 'r')
bond_file = open(dir+'boundary.txt', 'r')
numrels_file = open(dir+'numrels.txt', 'r')

newdir = '../data/QA/ibm-test/'
a2_file = open(newdir+'a.toks','w')
b2_file = open(newdir+'b.toks', 'w')
id2_file = open(newdir+'id.txt', 'w')
sim2_file = open(newdir+'sim.txt', 'w')
bond2_file = open(newdir+'boundary.txt', 'w')
numrels2_file = open(newdir+'numrels.txt', 'w')

bond2_file.write('0\n')
line_counter = 0
pos_question, neg_question = 0, 0
begin = bond_file.readline()
for line in numrels_file.readlines():
  end = bond_file.readline()
  if int(line) == 0:
    neg_question = neg_question + 1
  if int(line) == int(end) - int(begin):
    pos_question = pos_question + 1
  # question with all pos answers or all neg answers
  if int(line) == 0 or int(line) == int(end) - int(begin):
    for i in range(0, int(end)-int(begin)):
      a_file.readline()
      b_file.readline()
      id_file.readline()
      sim_file.readline()
  else:
    for i in range(0, int(end)-int(begin)):
      a2_file.write(a_file.readline())
      b2_file.write(b_file.readline())
      id2_file.write(id_file.readline())
      sim2_file.write(sim_file.readline())
      line_counter = line_counter + 1
    numrels2_file.write(line)
    bond2_file.write(str(line_counter)+'\n')
  begin = end
      
print(pos_question, neg_question) 
