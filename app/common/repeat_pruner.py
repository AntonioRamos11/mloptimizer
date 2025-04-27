from optuna.pruners import BasePruner
from optuna.trial import TrialState
from app.common.socketCommunication import *

class RepeatPruner(BasePruner):
	#Based on https://github.com/Minyus/optkeras/blob/master/optkeras/optkeras.py
	def prune(self, study, trial):
		#Get all trials without current
		all_trials = study.trials
		del all_trials[-1]
		#Count completed trials
		n_trials = len([t for t in all_trials if t.state == TrialState.COMPLETE or t.state == TrialState.RUNNING])
		#If there are no previoud trials
		if n_trials == 0:
			return False
		#Assert that current trial is running
		assert trial.state == TrialState.RUNNING
		#Extract params from previously completed trials
		completed_params_list = [t.params for t in all_trials if t.state == TrialState.COMPLETE or t.state == TrialState.RUNNING]
		#Check if current trial is repeated
		if trial.params in completed_params_list:
			SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': "A trial was pruned"})
			return True
		return False