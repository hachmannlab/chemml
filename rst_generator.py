import inspect
import pandas as pd
from tabulate import tabulate


databases = ['cheml', 'sklearn', ]
directory = 'docs/wrapper_docs'
direc = 'wrapper_docs'
extras = ['np','__builtins__', '__doc__', '__file__', '__name__', '__package__','mask','Input', 'Output', 'Parameter', 'req', 'regression_types', 'cv_types']
cols = ['task', 'subtask', 'host', 'function', 'input tokens', 'output tokens']
df = pd.DataFrame(columns = cols)
info = {'Enter':[], 'Prepare':[],'Model':[],'Search':[],'Mix':[],'Visualize':[],'Store':[]}
ind= 0

for h in databases:
    if h == 'sklearn':
        print h
        from cheml.wrappers.database import sklearn_db as db
        classes = [klass[0] for klass in inspect.getmembers(db)]
    elif h == 'cheml':
        print h
        from  cheml.wrappers.database import cheml_db as db
        classes = [klass[0] for klass in inspect.getmembers(db)]

    class_names = [c for c in classes if c not in extras]

    for n in class_names:
        ind += 1
        k = getattr(db, n)
        function = k.function
        host = k.host
        filename = '%s.%s'%(host,function)
        info[k.task].append('.. include:: %s/%s.rst'%(direc, filename))
        row = [k.task, k.subtask, host, ':ref:`%s`'%function]
        with open('%s/%s.rst'%(directory,filename), 'wb') as file:
            file.write('.. _%s:\n'%(function))
            file.write('\n')
            file.write('%s\n' % function)
            line = '=' * max(4,len(function)+1)
            file.write('%s\n'%line)
            file.write('\n')

            file.write(':task:\n')
            file.write('    | %s\n'%k.task)
            file.write('\n')

            file.write(':subtask:\n')
            file.write('    | %s\n'% k.subtask)
            file.write('\n')

            file.write(':host:\n')
            file.write('    | %s\n'%host)
            file.write('\n')

            file.write(':function:\n')
            file.write('    | %s\n'%function)
            file.write('\n')

            inputs = {}
            if len(vars(k.Inputs)) > 2:
                inputs = vars(k.Inputs)
                ins = ', '.join([i for i in inputs if i not in ('__module__', '__doc__')])
                row.append(ins)
                file.write(':input tokens (receivers):\n')
                for item in inputs:
                    if item not in ('__module__', '__doc__'):
                        file.write('    | ``%s`` : %s\n' % (inputs[item].name, inputs[item].short_description))
                        file.write('    |   %s\n'%str(inputs[item].types))
            else:
                row.append('-')
            file.write('\n')

            outputs = {}
            if len(vars(k.Outputs)) > 2:
                outputs = vars(k.Outputs)
                outs = ', '.join([i for i in outputs if i not in ('__module__', '__doc__')])
                row.append(outs)
                file.write(':output tokens (senders):\n')
                for item in outputs:
                    if item not in ('__module__', '__doc__'):
                        file.write('    | ``%s`` : %s\n' % (outputs[item].name, outputs[item].short_description))
                        file.write('    |   %s\n'%str(outputs[item].types))
            else:
                row.append('-')
            file.write('\n')

            wparams = {}
            if len(vars(k.WParameters)) > 2:
                wparams = vars(k.WParameters)
                file.write(':wrapper parameters:\n')
                for item in wparams:
                    if item not in ('__module__', '__doc__'):
                        file.write('    | ``%s`` : %s, (default:%s)\n' % (wparams[item].name, wparams[item].format, wparams[item].default))
                        file.write('    |   %s\n'%str(wparams[item].description))
                        file.write('    |   choose one of: %s\n' % str(wparams[item].options))
            file.write('\n')

            file.write(':required packages:\n')
            for r in k.requirements:
                file.write('    | %s, %s\n'%(r[0], r[1]))
            file.write('\n')

            file.write(':config file view:\n')
            file.write('    | ``## ``\n')
            file.write('    |   ``<< host = %s    << function = %s``\n' %(host,function))
            for item in wparams:
                if item not in ('__module__', '__doc__'):
                    file.write('    |   ``<< %s = %s``\n' % (wparams[item].name, wparams[item].default))

            if len(vars(k.FParameters)) > 2:
                fparams = vars(k.FParameters)
                for item in fparams:
                    if item not in ('__module__', '__doc__'):
                        file.write('    |   ``<< %s = %s``\n'%(fparams[item].name, fparams[item].default))
            for item in inputs:
                if item not in ('__module__', '__doc__'):
                    file.write('    |   ``>> id %s``\n'%inputs[item].name)
            for item in outputs:
                if item not in ('__module__', '__doc__'):
                    file.write('    |   ``>> id %s``\n'%outputs[item].name)
            file.write('    |\n')
            file.write('    .. note:: The documentation page for function parameters: %s'%k.documentation)
            df.loc[ind] = row

for i in info:
    print '*** %s:'%i
    for j in info[i]:
        print j
    print '\n'

print '********** contents tables **********'
df = df.sort('task')
print tabulate(df, headers='keys', tablefmt='psql')
# g = df.groupby('task',0)
# for f in g:
#     print tabulate(f[1], headers='keys', tablefmt='psql')