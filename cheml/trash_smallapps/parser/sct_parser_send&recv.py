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

    def fit(self):
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
        ImpOrder,CompGraph = self.transform(cmls)
        self._print_out(cmls)
        return cmls, ImpOrder, CompGraph

    def _functions(self, line):
        if '<' in line:
            function = line[line.index('##')+2:line.index('<')].strip()
        elif '>' in line:
            function = line[line.index('##')+2:line.index('>')].strip()
        else:
            function = line[line.index('##')+2:].strip()
        return function

    def _parameters(self, block,item):
        parameters = {}
        send = {}
        recv = {}
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
                arg = args.split()
                if len(arg) == 2:
                    var, id = arg
                    send[(var, int(id))] = item
                elif len(arg) == 1:
                    recv[('recv%i'%item,int(arg[0]))] = item
                else:
                    msg = 'wrong format of send and receive in block #%i at %s (send: >> var id; recv: >> id)'%(item+1,args)
                    raise ValueError(msg)


        return parameters, send, recv

    def _options(self, blocks):
        cmls = []
        for item in xrange(len(blocks)):
            block = blocks[item]
            function = self._functions(block[0])
            parameters, send, recv = self._parameters(block,item)
            cmls.append({"SuperFunction": function,
                         "parameters": parameters,
                         "send": send,
                         "recv": recv})
        return cmls

    def _print_out(self, cmls):
        item = 0
        for block in cmls:
            item+=1
            line = '%s\n' %(block['SuperFunction'])
            line = line.rstrip("\n")
            print '%i'%item+' '*(4-len(str(item)))+'Task: '+line
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
                    line = '%s -> send\n' %str(param)
                    line = line.rstrip("\n")
                    print '        ' + line
            else:
                line = ' :no send:'
                line = line.rstrip("\n")
                print '        ' + line
            if len(block['recv']) > 0:
                for param in block['recv']:
                    line = '%s -> recv\n' %str(param)
                    line = line.rstrip("\n")
                    print '        ' + line
            else:
                line = ' :no receive:'
                line = line.rstrip("\n")
                print '        ' + line
            line = ''
            line = line.rstrip("\n")
            print '        ' + line

    def transform(self, cmls):
        """
        goals:
            - collect all sends and receives
            - check send and receive format.
            - make the computational graph
            - find the order of implementation of functions based on sends and receives

        :param cmls:
        :return implementation order:
        """
        send_all = {}
        recv_all = {}
        for block in cmls:
            send_all.update(block['send'])
            recv_all.update(block['recv'])
        # check send and recv
        if len(send_all) != len(recv_all):
            msg = 'not an equal number of send and receive has been provided'
            raise ValueError(msg)
        send_ids = [k[1] for k,v in send_all.items()]
        recv_ids = [k[1] for k,v in recv_all.items()]
        for id in send_ids:
            if send_ids.count(id)>1:
                msg = 'identified non unique send id (id#%i)'%id
                raise NameError(msg)
        for id in recv_ids:
            if recv_ids.count(id)>1:
                msg = 'identified non unique receive id (id#%i)'%id
                raise NameError(msg)
        if len(set(send_ids) - set(recv_ids))>0:
            msg = 'missing pairs of send and receive id'
            raise ValueError(msg)

        # make graph
        reformat_send = {k[1]:[v,-1,k[0]] for k,v in send_all.items()}
        for k, v in recv_all.items():
            reformat_send[k[1]][1] = v
            reformat_send[k[1]] = tuple(reformat_send[k[1]])
        CompGraph = tuple(reformat_send.values())

        # find orders
        ids_sent = []
        ImpOrder = []
        inf_checker = 0
        while len(ImpOrder)<len(cmls):
            inf_checker +=1
            for i in range(len(cmls)):
                if i not in ImpOrder:
                    ids_recvd = [k[1] for k,v in cmls[i]['recv'].items()]
                    if len(ids_recvd) == 0:
                        ids_sent += [k[1] for k,v in cmls[i]['send'].items()]
                        ImpOrder.append(i)
                    elif len(set(ids_recvd) - set(ids_sent))==0:
                        ids_sent += [k[1] for k,v in cmls[i]['send'].items()]
                        ImpOrder.append(i)
            if  inf_checker > len(cmls):
                msg = 'Your design of send and receive tokens makes a loop of interdependencies. We believe that you can avoid such loops with setting only one received input per input type.'
                raise IOError(msg)
        return tuple(ImpOrder),CompGraph


SCRIPT_NAME = "sample_script.txt"
script = open(SCRIPT_NAME, 'r')
script = script.readlines()
cmls, ImpOrder, CompGraph = Parser(script).fit()
print cmls
print ImpOrder
print CompGraph


