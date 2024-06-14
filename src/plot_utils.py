import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_runs(run_path_forward, run_path_backward, optimizers) :
  metrics = ["Loss","Acc","F1","Time"]
  train_run_epochs = {"forward" : {opt : {k:[] for k in metrics}for opt in optimizers.keys()} , "backward"  : {opt : {k:[] for k in metrics}for opt in optimizers.keys()}}
  train_run_its = {"forward" : {opt : {k:[] for k in metrics} for opt in optimizers.keys()}, "backward" : {opt : {k:[] for k in metrics}for opt in optimizers.keys()}}
  val_run_epochs = {"forward" : {opt : {k:[] for k in metrics} for opt in optimizers.keys()}, "backward" : { opt : {k:[] for k in metrics}for opt in optimizers.keys()}}
  val_run_its = {"forward" : {opt : {k:[] for k in metrics} for opt in optimizers.keys()}, "backward" : {opt : {k:[] for k in metrics}for opt in optimizers.keys()}}

  for opt in run_path_backward :
      for strat in ["forward", "backward"] :
          run_path = "logistic_reg_runs/forward/"+run_path_forward[opt] if strat=="forward" else "logistic_reg_runs/backward/"+run_path_backward[opt]
          for k in metrics:
            train_run_epochs[strat][opt][k] = pickle.load(open(run_path + "{}_train_run.pkl".format(k), "rb"))
            train_run_its[strat][opt][k] = pickle.load(open(run_path + "{}_train_run_its.pkl".format(k), "rb"))
            val_run_epochs[strat][opt][k] = pickle.load(open(run_path + "{}_val_run.pkl".format(k), "rb"))
            val_run_its[strat][opt][k] = pickle.load(open(run_path + "{}_val_run_its.pkl".format(k), "rb"))
            if k == "Time" :
              train_run_epochs[strat][opt]["CumTime"] = np.cumsum(train_run_epochs[strat][opt][k])
              val_run_epochs[strat][opt]["CumTime"] = np.cumsum(val_run_epochs[strat][opt][k])
              train_run_its[strat][opt]["CumTime"] = np.cumsum(train_run_its[strat][opt][k])
              val_run_its[strat][opt]["CumTime"] = np.cumsum(val_run_its[strat][opt][k])
  return train_run_epochs, train_run_its, val_run_epochs, val_run_its, metrics+["CumTime"]

def plot_comparison(train_data, val_data, optimizers):    
    metrics = ["Loss", "Acc", "F1"]
    opt_colors = {"SGD":"red", "MomentumSGD":"blue", "ClippedSGD":"green", "Adagrad":"orange", "Adam":"purple"}

    for k in range(len(metrics)):
        _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        metric = metrics[k]
        for opt in optimizers:
                axes[0].plot(train_data["forward"][opt][metric], label=f'forward {opt} train', alpha=0.6, color = opt_colors[opt] )
                #axes[k][0].plot(val_data[strats[i]][opt][metric], label=f'{strats[i][0]+opt} val', linestyle='dashed', alpha=0.6, color=strats[i][1])
                axes[0].set_title(f'Forward {metric} by Epochs')
                axes[0].set_xlabel("Epochs")
                axes[0].set_ylabel(metric)
                axes[0].legend()
                axes[0].grid(True)
                axes[1].plot(train_data["backward"][opt][metric], label=f'backward {opt} train', alpha=0.6, color = opt_colors[opt] )
                #axes[k][1].plot(val_data[strats[i][0]][opt]["CumTime"], val_data[strats[i][0]][opt][metric], label=f'{strats[i][0]+opt} val', linestyle='dashed', alpha=0.6, color=strats[i][1])
                axes[1].set_title(f'Backward {metric} by Epochs')
                axes[1].set_xlabel('Epochs')
                axes[1].set_ylabel(metric)
                axes[1].legend()
                axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()


def plot_train_valid(train_data, opt, x_axis,val_data=None):

    _, axes = plt.subplots(3, 2, figsize=(20, 10))
    
    strats = [("forward", "red"), ("backward","blue")]
    metrics = ["Loss", "Acc", "F1"]
    for k in range(len(metrics)):
        metric = metrics[k]
        for i in range(len(strats)):
            axes[k][0].plot(train_data[strats[i][0]][opt][metric], label=f'{strats[i][0] +opt} train', alpha=0.6 , color=strats[i][1])
            if val_data:
                axes[k][0].plot(val_data[strats[i][0]][opt][metric], label=f'{strats[i][0]+opt} val', linestyle='dashed', alpha=0.6, color=strats[i][1])
            axes[k][0].set_title(f'{strats[i][0]} {metric} by {x_axis}')
            axes[k][0].set_xlabel(x_axis)
            axes[k][0].set_ylabel(metric)
            if k == 0:
                axes[k][0].legend()
            axes[k][0].grid(True)
            axes[k][1].plot(train_data[strats[i][0]][opt][metric], label=f'{strats[i][0] +opt} train', alpha=0.6, color=strats[i][1])
            if val_data:
                axes[k][1].plot(val_data[strats[i][0]][opt][metric], label=f'{strats[i][0]+opt} val', linestyle='dashed', alpha=0.6, color=strats[i][1])
            axes[k][1].set_title(f'{strats[i][0]} {metric} by Time (s)')
            axes[k][1].set_xlabel('Time (s)')
            axes[k][1].set_ylabel(metric)
            if k == 0:
                axes[k][1].legend()
            axes[k][1].grid(True)

    plt.tight_layout()
    plt.show()