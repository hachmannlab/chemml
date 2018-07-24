from cheml.wrappers.database.TSHF import tshf, get_complete_meta


tasks, combination = tshf()
print 'tasks : ', tasks
print 'combination : ', combination

meta = get_complete_meta()
print 'Meta : ', meta
