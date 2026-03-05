from app.init_nodes import *

if __name__ == '__main__':
	print("Which node is this instance? (1=master, 2=slave)")
	n = int(input("Type a number:"))
	if n == 1:
		InitNodes().master()
	elif n == 2:
		InitNodes().slave()
	else:
		print("Write a valid number")

