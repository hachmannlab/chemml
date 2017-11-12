import inspect

from cheml.wrappers.database.unhooked import sklearn_db

file = open('sklearndb.py','w')

fns = [klass[0] for klass in inspect.getmembers(sklearn_db)]
extras = ['np','__builtins__', '__doc__', '__file__', '__name__', '__package__','mask','Input', 'Output', 'Parameter', 'req', 'regression_types', 'cv_types']
fns = [c for c in fns if c not in extras]
def add_parameter(block,item,default,format,description,options):
    line = "        %s = Parameter('%s', %s,'%s'," % (item, item, default, format)
    block.append(line)
    line = '                        description = "%s",'%description
    block.append(line)
    line = "                        options = %s)" %options
    block.append(line)
    return block


file.write('import numpy as np\n')
file.write('from .containers import Input, Output, Parameter, req, regression_types, cv_types\n\n')
for f in fns:
    block = []
    F = getattr(sklearn_db, f)
    line = "class %s(object):"%F.function
    block.append(line)
    line = "    task = '%s'"%F.task
    block.append(line)
    line = "    subtask = '%s'"%F.subtask
    block.append(line)
    line = "    host = '%s'"%F.host
    block.append(line)
    line = "    function = '%s'"%F.function
    block.append(line)
    line = "    modules = ('%s','%s')"%(F.modules[0],F.modules[1])
    block.append(line)
    line = "    requirements = (req(0), req(2))"
    block.append(line)
    line = "    documentation = " + '"' + F.documentation + '"'
    block.append(line)
    line = ""
    block.append(line)
    line = "    class Inputs:"
    block.append(line)
    if len(vars(F.Inputs).keys())>2:
        for item in vars(F.Inputs).keys():
            if item not in ('__module__','__doc__'):
                line = '        %s = Input("%s","%s", ('%(item, item,vars(F.Inputs)[item].short_description)
                for typ in vars(F.Inputs)[item].types:
                    line += '"%s",'%typ
                line += '))'
                block.append(line)
    else:
        line = "        pass"
        block.append(line)
    line = "    class Outputs:"
    block.append(line)
    if len(vars(F.Outputs).keys())>2:
        for item in vars(F.Outputs).keys():
            if item not in ('__module__', '__doc__'):
                line = '        %s = Output("%s","%s", ('%(item, item,vars(F.Outputs)[item].short_description)
                for typ in vars(F.Outputs)[item].types:
                    line += '"%s",'%typ
                line += '))'
                block.append(line)
    else:
        line = "        pass"
        block.append(line)
    line = "    class WParameters:"
    block.append(line)
    if len(vars(F.WParameters).keys())>2:
        for item in vars(F.WParameters).keys():
            if item not in ('__module__', '__doc__'):
                dic = vars(F.WParameters)[item]
                line = "        %s = Parameter('%s','%s','%s'," % (item, item, dic.default, dic.format)
                block.append(line)
                line = "                        description = " + '"%s",'%dic.description
                block.append(line)
                line = "                        options = %s)"%str(dic.options)
                block.append(line)
    else:
        line = "        pass"
        block.append(line)
    # Adding track_header to all functions
    description = "if True, the input dataframe's header will be transformed to the output dataframe"
    options = '(True, False)'
    block = add_parameter(block, 'track_header', 'True', 'Boolean', description, options)
    line = "    class FParameters:"
    block.append(line)
    line = "        pass"
    block.append(line)
    line = ""
    block.append(line)
    for line in block:
        file.write(line+'\n')
file.close()


