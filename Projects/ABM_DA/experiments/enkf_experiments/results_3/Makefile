all: results_dir get_utils

results_dir:
	mkdir -p results/{data,figures,models}/{baseline,exp1,exp2,exp3}

get_utils:
	cp ../experiment_utils.py ./

clean:
	rm -rf results
	rm ./experiment_utils.py
