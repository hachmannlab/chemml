## read script (*.cheml)
import glob

for filename in glob.glob('*.cheml'):
    script = open(filename,'r')
script = script.readlines()

def block_finder(script, blocks={}, it=-1):
    for line in script:
        if '##' in line:
            it += 1    
            blocks[it] = [line]
            continue
        elif '#' not in line and '%' in line:
            blocks[it].append(line)
            continue
    return blocks

def function_finder(line):
    if '%' in line:
        function = line[line.index('##')+2:line.index('%')].strip()
    else:
        function = line[line.index('##')+2:].strip()
    return function

def options(blocks, it=-1):
    cmlnb = []
    for i in xrange(len(blocks)):
        it += 1
        block = blocks[i]
        cmlnb.append({"function": function_finder(block[0]),
                     "parameters": {}})
        for line in block:
            while '%%' in line:
                line = line[line.index('%%')+2:].strip()
                if '%' in line:
                    args = line[:line.index('%')].strip()
                else:
                    args = line.strip()
                param = args[:args.index('=')].strip()
                val = args[args.index('=')+1:].strip()
                exec("cmlnb[it]['parameters']['%s']"%param+'='+'"%s"'%val)
    return cmlnb

def print_out(cmlnb):
    file = open('script.txt','w')
    for cmls in cmlnb:
        line = '%s [%s]\n' %(cmls['function'],type(cmls['function']))
        print '    function = '+line
        make_script(file,line,'f')
        for param in cmls['parameters']:
            line = '%s = %s [%s]\n'%(param,cmls['parameters'][param],type(cmls['parameters'][param]))
            print '        '+line 
            make_script(file,line,'p')
    file.close()

def make_script(file,line,type):
    if type == 'f':
        file.write('## '+line)
    if type == 'p':
        file.write('    %% '+line)


blocks = block_finder(script)
print blocks
print '\n'
print '\n'
cmlnb = options(blocks)
print cmlnb
print '\n'
print_out(cmlnb)