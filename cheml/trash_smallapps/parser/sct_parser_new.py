## read script (*.cheml)
import glob

for filename in glob.glob('*.cheml'):
    script = open(filename,'r')

script = script.readlines()
#script = [line.strip() for line in script]

def _block_finder(script):
    blocks={}
    item=-1
    a = [i for i,line in enumerate(script) if '#' in line]
    b = [i for i,line in enumerate(script) if '###' in line]
    if len(b)%2 != 0:
        msg = "one of the super functions has not been wrapped with '###' properly."
        raise ValueError(msg)
    for ind in b[1::2]:
        script[ind] = ''
    for i,j in zip(b,b[1:])[::2]:
        a = [ind for ind in a if ind<=i or ind>j]
    inactive = [a.index(i) for i in a if '##' not in script[i]]
    for i in xrange(len(a)-1):
        blocks[i] = script[a[i]:a[i+1]]
    blocks[len(a)-1] = script[a[-1]:]
    return blocks

def _functions(line):
    if '%' in line:
        function = line[line.index('##')+2:line.index('%')].strip()
    else:
        function = line[line.index('##')+2:].strip()
    return function

def _parameters(block):
    parameters = {}
    for line in block:
        while '%%' in line:
            line = line[line.index('%%')+2:].strip()
            if '%' in line:
                args = line[:line.index('%')].strip()
            else:
                args = line.strip()
            param = args[:args.index('=')].strip()
            val = args[args.index('=')+1:].strip()
            parameters[param] = "%s"%val
    return parameters

def _superfunctions(line):
    if '#' in line[line.index('###')+3:]:
        function = line[line.index('###')+3:line.index('#')].strip()
    else:
        function = line[line.index('###')+3:].strip()
    return function

def _superparameters(block):
    parameters = []
    block[0] = block[0][block[0].index('###')+3:]
    sub_blocks = _block_finder(block)
    for i in xrange(len(sub_blocks)):
        blk = sub_blocks[i]
        if '##' in blk[0]:
            parameters.append({"function": _functions(blk[0]),
                               "parameters": _parameters(blk)})
        else:
            continue        
    return parameters
    
def _options(blocks):
    cmls = []
    for item in xrange(len(blocks)):
        block = blocks[item]
        if '###' in block[0]:
            cmls.append({"function": _superfunctions(block[0]),
                         "parameters": _superparameters(block)})
        elif '##' in block[0]:
            cmls.append({"function": _functions(block[0]),
                         "parameters": _parameters(block)})
    return cmls

def _print_out(cmls):
    item = 0
    for block in cmls:
        item+=1
        line = '%s\n' %(block['function'])
        line = line.rstrip("\n")
        print '%i'%item+' '*(4-len(str(item)))+'function = '+line
        if type(block['parameters']) == dict:
            for param in block['parameters']:
                line = '%s = %s\n'%(param,block['parameters'][param])
                line = line.rstrip("\n")
                print '        '+line 
        elif type(block['parameters']) == list:
            for sub_block in block['parameters']:
                line = '%s\n' %(sub_block['function'])
                line = line.rstrip("\n")
                print '     *'+' '*2+'function = '+line
                for param in sub_block['parameters']:
                    line = '%s = %s\n'%(param,sub_block['parameters'][param])
                    line = line.rstrip("\n")
                    print '            '+line 
        else:
            msg = "script parser can not recognize the format of script"
            raise ValueError(msg)
            
"""
def print_out(cmlnb):
    file = open('script.txt','w')
    for cmls in cmlnb:
        line = '%s [%s]\n' %(cmls['function'],type(cmls['function']))
        line = line.rstrip("\n")
        print line
#         make_script(file,line,'f')
        for param in cmls['parameters']:
            line = '%s = %s [%s]\n'%(param,cmls['parameters'][param],type(cmls['parameters'][param]))
            line =  '        '+line 
            line = line.rstrip("\n")
            print line
#             make_script(file,line,'p')
    file.close()

def make_script(file,line,type):
    if type == 'f':
        file.write('## '+line)
    if type == 'p':
        file.write('    %% '+line)

def sub_function(block,line):
    line = line.split('__')
    imp = line[0]
    block['sub_function'] = line[0].split('.')[-1]
    block['sub_parameters'] = {}
    for arg in line[1:]:
        param = arg.split('=')[0].strip()
        val = arg.split('=')[1].strip()
        block['sub_parameters'][param] = '"%s"'%val
    return imp 
        
"""        
blocks = _block_finder(script)
print blocks
print '\n'
print '\n'
cmls = _options(blocks)
print cmls
print '\n'
_print_out(cmls)