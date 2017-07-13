import pandas as pd
from tabulate import tabulate

df = pd.read_excel('contents.xlsx',sheetname=0,header=0) 
# print tabulate(df, headers='keys', tablefmt='psql')

###################
dir = 'CMLWR_ALL'

print df.shape
for i in range(df.shape[0]):
    i+=1
    function = df['function'][i][6:-1]
    filename = 'CMLWR.%s'%function
    print filename
    with open('%s/raw/%s.rst'%(dir,filename), 'wb') as file:
        file.write('.. _%s:\n'%function)
        file.write('\n')
        file.write('%s\n' % function)
        line = '=' * max(4,len(function)+1)
        file.write('%s\n'%line)
        file.write('\n')

        file.write(':task:\n')
        file.write('    | %s\n'%df['task'][i])
        file.write('\n')

        file.write(':subtask:\n')
        file.write('    | %s\n'% df['subtask'][i])
        file.write('\n')

        file.write(':host:\n')
        file.write('    | %s\n'%df['host'][i])
        file.write('\n')

        file.write(':function:\n')
        file.write('    | %s\n'%function)
        file.write('\n')

        file.write(':input tokens (receivers):\n')
        receivers_list = df['input tokens (receivers)'][i].split(',')
        for item in receivers_list:
            file.write('    | ``%s`` : pandas DataFrame, shape(n_samples, n_features), requied\n'%item.strip())
            file.write('    |   input DataFrame\n')
        file.write('\n')

        file.write(':output tokens (senders):\n')
        senders_list = df['output tokens (senders)'][i].split(',')
        for item in senders_list:
            file.write('    | ``%s`` : pandas DataFrame, shape(n_samples, n_features), requied\n'% item.strip())
            file.write('    |   output DataFrame\n')
        file.write('\n')

        file.write(':required parameters:\n')
        params_list = df['req_parameters'][i].split(',')
        for item in params_list:
            file.write('    | %s'% item.strip().split('=')[0])
            file.write(' (%s)\n'%item.strip().split('=')[1])
        file.write('    |\n')
        file.write('    .. note:: The documentation for this function can be found here_\n\n'\
                   '    .. _here: %s\n'% df['param_doc'][i])
        file.write('\n')

        file.write(':required packages:\n')
        requirements_list = df['requirements'][i].split(',')
        for ind in range(len(requirements_list)/2):
            file.write('    | %s, %s\n'%(requirements_list[2*ind].strip(),requirements_list[2*ind+1].strip()))
        file.write('\n')

        file.write(':input file view:\n')
        file.write('    | ``## %s``\n'% df['task'][i])
        file.write('    |   ``<< host = %s    << function = %s``\n' %(df['host'][i],function))
        params_list = df['parameters'][i].split(',')
        for item in params_list:
            file.write('    |   ``<< %s = %s``\n'%(item.strip().split('=')[0],item.strip().split('=')[1]))
        for item in receivers_list:
            if item.strip() == 'no receiver':
                pass
            else:
                file.write('    |   ``>> id %s``\n'%item.strip())
        for item in senders_list:
            if item.strip() == 'no sender':
                pass
            else:
                file.write('    |   ``>> %s id``\n'%item.strip())
        file.write('    |\n')
        file.write('    .. note:: The rest of parameters (if any) can be set the same way.')
