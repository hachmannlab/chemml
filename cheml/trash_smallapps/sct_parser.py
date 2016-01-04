## read script (*.cheml)
import glob
for filename in glob.glob('*.cheml'):
	script = open(filename,'r')
script = script.readlines()
	
## make to-do list
todo={}
todo_order=[]
for line in script:
	if '##' not in line:
		continue
	
	# function
	if '%%' in line:
		function = line[line.index('##')+2:line.index('%')].strip()
		line = line[line.index('%'):]
	elif '%' in line:
		function = line[line.index('##')+2:line.index('%')].strip()
		todo[function] = []
		todo_order.append(function)
		continue
	else:
		function = line[line.index('##')+2:].strip()
		todo[function] = []
		todo_order.append(function)
		continue
	
	# args	
	args=[]	
	while '%%' in line:
		line = line[line.index('%%')+2:].strip()
		if '%' in line:
			args.append(line[:line.index('%')].strip())
		else:
			args.append(line.strip())
	
	todo[function] = args
	todo_order.append(function)

print todo
print todo_order