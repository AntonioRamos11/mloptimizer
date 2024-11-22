from app.init_nodes import *

if __name__ == '__main__':
	print("Which node is this instance? (1=master, 2=slave)")
	print("Initializing slave node ...")
	InitNodes().slave()
