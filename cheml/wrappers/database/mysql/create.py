#!/usr/bin/python
import MySQLdb as db

db1=db.connect("localhost","root","ankita123","ChemMLWrapper")

cur=db1.cursor()
#cur.execute("DROP TABLE IF EXISTS TASK")

# CREATING TABLE

sql="""CREATE TABLE task (
	task_id INT NOT NULL AUTO_INCREMENT,
	NAME CHAR(30) NOT NULL,
	PRIMARY KEY (task_id))"""
cur.execute(sql)
sql="""CREATE TABLE subtask (
	subtask_id INT NOT NULL AUTO_INCREMENT,
	NAME CHAR(30) NOT NULL,
	task_id INT NOT NULL,
	PRIMARY KEY (subtask_id),
	FOREIGN KEY (task_id) REFERENCES task(task_id))"""
cur.execute(sql)
sql="""CREATE TABLE host (
	host_id INT NOT NULL AUTO_INCREMENT,
	NAME CHAR(30) NOT NULL,
	subtask_id INT NOT NULL,
	PRIMARY KEY (host_id),
	FOREIGN KEY (subtask_id) REFERENCES subtask(subtask_id))"""
cur.execute(sql)
sql="""CREATE TABLE function (
	function_id INT NOT NULL AUTO_INCREMENT,
	NAME CHAR(30) NOT NULL,
	host_id INT NOT NULL,
	parameters_doc VARCHAR(300),
	requirements VARCHAR(200),
	PRIMARY KEY (function_id),
	FOREIGN KEY (host_id) REFERENCES host(host_id))"""
cur.execute(sql)
sql="""CREATE TABLE type (
	type_id INT NOT NULL AUTO_INCREMENT,
	NAME CHAR(30) NOT NULL,
	PRIMARY KEY (type_id))"""
cur.execute(sql)
sql="""CREATE TABLE parameters (
	parameter_id INT NOT NULL AUTO_INCREMENT,
	NAME CHAR(30) NOT NULL,
	value VARCHAR(100),
	required BOOLEAN,
	function_id INT NOT NULL,
	type_id INT NOT NULL,
	PRIMARY KEY (parameter_id),
	FOREIGN KEY (type_id) REFERENCES type(type_id),
	FOREIGN KEY (function_id) REFERENCES function(function_id))"""
cur.execute(sql)
sql="""CREATE TABLE IO (
	IO_id INT NOT NULL AUTO_INCREMENT,
	NAME CHAR(30) NOT NULL,
	short_info VARCHAR(100),
	long_info VARCHAR(200),
	value VARCHAR(100),
	type VARCHAR(100),
	sender VARCHAR(100),
	count INT,
	function_id INT NOT NULL,
	type_id INT NOT NULL,
	PRIMARY KEY (IO_id),
	FOREIGN KEY (type_id) REFERENCES type(type_id),
	FOREIGN KEY (function_id) REFERENCES function(function_id))"""

cur.execute(sql)
#add I/O table here
db1.close()