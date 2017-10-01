#!/usr/bin/python
import MySQLdb as db

db1=db.connect("localhost","root","ankita123","ChemMLWrapper")

cur=db1.cursor()

sql="""INSERT INTO task(NAME)
VALUES("Prepare")"""
cur.execute(sql)
db1.commit()

sql="SELECT * FROM task"
cur.execute(sql)
info=cur.fetchone()


sql="""INSERT INTO subtask(NAME,task_id)
VALUES('%s','%d')""" % ("descriptor",info[0])
cur.execute(sql)
db1.commit()

sql="SELECT * FROM subtask"
cur.execute(sql)
info=cur.fetchone()


sql="""INSERT INTO host(NAME,subtask_id)
VALUES('%s','%d')""" % ("sklearn",info[0])
cur.execute(sql)
db1.commit()

sql="SELECT * FROM host"
cur.execute(sql)
info=cur.fetchone()


sql="""INSERT INTO function(NAME,host_id)
VALUES('%s','%d')""" % ("PolynomialFeatures",info[0])
cur.execute(sql)
db1.commit()

sql="SELECT * FROM function"
cur.execute(sql)
function_info=cur.fetchone()

lis=["Input","Output","Function_parameters","Wrapper_parameters"]
for i in lis:
	sql="""INSERT INTO type(NAME
	VALUES('%s') """ % (i)
	cur.execute(sql)
	db1.commit()

sql="SELECT * FROM type"
cur.execute(sql)
type_info=cur.fetall()
check =0
for i in type_info[0:3]:
	sql="""INSERT INTO IO(NAME,short_info,long_info,value,type,count,function_id,type_id)
	VALUES('%s','%s','%s','%s','%s','%d','%d','%d')""" % ("df","pandas dataframe","","","<class 'pandas.core.frame.DataFrame'>",0,function_info[0],i)
	cur.execute(sql)
	db1.commit()
	sql="""INSERT INTO IO(NAME,short_info,long_info,value,type,count,function_id,type_id)
	VALUES('%s','%s','%s','%s','%s','%d','%d','%d')""" % ("api","sklearn PolynomialFeatures class","","","<class 'sklearn.preprocessing.data.PolynomialFeatures'>",0,function_info[0],i)
	cur.execute(sql)
	db1.commit()



db1.close()

