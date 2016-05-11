def value(string):
    try:
        return eval(string)
    except NameError:
        return string

class Parser(object):
    """
    script: list of strings
        A list of lines in the cheml script file.
    """
    def __init__(self, script):
        self.script = script

    def run(self):
        """
        The main funtion for parsing cheml script.
        It starts with finding blocks and then runs other functions.

        :return:
        cmls: cheml script
        """
        blocks={}
        it = -1
        for line in self.script:
            if '##' in line:
                it += 1
                blocks[it] = [line]
                continue
            elif '#' not in line and ('<' in line or '>' in line):
                blocks[it].append(line)
                continue

        cmls = self._options(blocks)
        self._print_out(cmls)
        return cmls

    def _functions(self, line):
        if '<' in line:
            function = line[line.index('##')+2:line.index('<')].strip()
        elif '>' in line:
            function = line[line.index('##')+2:line.index('>')].strip()
        else:
            function = line[line.index('##')+2:].strip()
        return function

    def _parameters(self, block):
        parameters = {}
        send = {}
        for line in block:
            while '<<' in line:
                line = line[line.index('<<')+2:].strip()
                if '<' in line:
                    args = line[:line.index('<')].strip()
                else:
                    args = line.strip()
                param = args[:args.index('=')].strip()
                val = args[args.index('=')+1:].strip()
                parameters[param] = value(val) #val #"%s"%val
            while '>>' in line:
                line = line[line.index('>>') + 2:].strip()
                if '>' in line:
                    args = line[:line.index('>')].strip()
                else:
                    args = line.strip()
                param, val = args.split()
                send[param] = value(val) #val #"%s" % val
        return parameters, send

    def _options(self, blocks):
        cmls = []
        for item in xrange(len(blocks)):
            block = blocks[item]
            function = self._functions(block[0])
            parameters, send = self._parameters(block)
            cmls.append({"function": function,
                         "parameters": parameters,
                         "send": send})
        return cmls

    def _print_out(self, cmls):
        item = 0
        for block in cmls:
            item+=1
            line = '%s\n' %(block['function'])
            line = line.rstrip("\n")
            print '%i'%item+' '*(4-len(str(item)))+'function = '+line
            line = '<<<<<<<'
            line = line.rstrip("\n")
            print '        ' + line
            if len(block['parameters']) > 0 :
                for param in block['parameters']:
                    line = '%s = %s\n'%(param,block['parameters'][param])
                    line = line.rstrip("\n")
                    print '        '+line
            else:
                line = ' :no parameter passed: set to defaul values if available'
                line = line.rstrip("\n")
                print '        ' + line
            line = '>>>>>>>'
            line = line.rstrip("\n")
            print '        ' + line
            if len(block['send']) > 0:
                for param in block['send']:
                    line = '%s -> %s\n' % (param, block['send'][param])
                    line = line.rstrip("\n")
                    print '        ' + line
            else:
                line = ' :no token passed: nothing to send to other functions'
                line = line.rstrip("\n")
                print '        ' + line
            line = ''
            line = line.rstrip("\n")
            print '        ' + line

SCRIPT_NAME = "sample_script.txt"
script = open(SCRIPT_NAME, 'r')
script = script.readlines()
cmls = Parser(script).run()
print cmls


