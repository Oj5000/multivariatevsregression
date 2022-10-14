import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from eval_funcs import *

from config.QSARFishToxicity import QSARFishToxicity
from config.GasEmission import GasEmission
from config.AirQuality import AirQuality
from config.CommunitiesAndCrime import CommunitiesAndCrime
from config.Superconductivity import Superconductivity
from config.CTSliceAxial import CTSliceAxial
from config.Accelerometer import Accelerometer
from config.BiasCorrection import BiasCorrection
from config.Cargo2000 import Cargo2000
from config.Chickenpox import Chickenpox
from config.CNNPred import CNNPred
from config.HouseholdPowerConsumption import HouseholdPowerConsumption
from config.OnlineNewsPopularity import OnlineNewsPopularity
from config.QueryAnalyticsWorkloads import QueryAnalyticsWorkloads
from config.WaveEnergyConverters import WaveEnergyConverters
from config.Synthetic import Synthetic

def plot_results_all_data(evaluators, name):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(evaluators)):
        mutation = []
        f1 = []

        for key in evaluators[i].results_all_data.keys():
            mutation.append(key)

            if i == 0:
                f1.append(np.mean(evaluators[i].results_all_data[key]['f1']))
            else:
                f1.append(float(evaluators[i].get_f1_result_all_data(key).split(" ")[1]))

        plt.plot(mutation, f1, label=evaluators[i].type)

    plt.legend(loc='lower right')
    ax.set_xlabel('Mutation')
    ax.set_ylabel('f1')
    plt.savefig('results/' + name + "_all_data.pdf")
    plt.close()

def plot_results_blind(evaluators, name):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(evaluators)):
        mutation = []
        f1 = []

        for key in evaluators[i].results_blind.keys():
            mutation.append(key)

            if i == 0:
                f1.append(np.mean(evaluators[i].results_blind[key]['f1']))
            else:
                f1.append(float(evaluators[i].get_f1_result_blind(key).split(" ")[1]))

        plt.plot(mutation, f1, label=evaluators[i].type)

    plt.legend(loc='lower right')
    ax.set_xlabel('Mutation')
    ax.set_ylabel('f1')
    plt.savefig('results/' + name + "_blind.pdf")
    plt.close()

# (str(mutation_perc), evaluators)
def write_f1_results_all_data(results, filename):
    # Open file and write header
    # mutation % | alg name #1 
    #     %      |     f1
    f = open("results/" + filename + "_f1_all_data.csv", "w")
    f.write("Outlier %|Mutation %|")

    t = ""
    for i in results[0][0]:
        t += i.type + "|"

    t = t[:-1] + "\n"
    f.write(t)

    for res in results:
        for key in res[0][0].results_all_data.keys():
            line = res[1] + "|"
            line += str(key) + "|"
            for result in res[0]:
                if type(result) == MultivariateDensity:
                    line += "0: " + result.get_f1_result_all_data(key) + "|"
                else:
                    line += result.get_f1_result_all_data(key) + "|"
                
            f.write(line[:-1] + "\n")

def write_all_tp_results_all_data(results, filename):
    # Open file and write header
    # mutation % | alg name #1 
    #     %      |     f1
    f = open("results/" + filename + "_all_tp_all_data.csv", "w")
    f.write("Outlier %|Mutation %|")

    t = ""
    for i in results[0][0]:
        t += i.type + "|"

    t = t[:-1] + "\n"
    f.write(t)

    for res in results:
        for key in res[0][0].results_all_data.keys():
            line = res[1] + "|"
            line += str(key) + "|"
            for result in res[0]:
                if type(result) == MultivariateDensity:
                    line += "0: " + result.get_all_tp_result_all_data(key) + "|"
                else:
                    line += result.get_all_tp_result_all_data(key) + "|"
                
            f.write(line[:-1] + "\n")

def write_f1_results_blind(results, filename):
    # Open file and write header
    # mutation % | alg name #1 
    #     %      |     f1
    f = open("results/" + filename + "_f1_blind.csv", "w")
    f.write("Outlier %|Mutation %|")

    t = ""
    for i in results[0][0]:
        t += i.type + "|"

    t = t[:-1] + "\n"
    f.write(t)

    for res in results:
        for key in res[0][0].results_blind.keys():
            line = res[1] + "|"
            line += str(key) + "|"
            for result in res[0]:
                if type(result) == MultivariateDensity:
                    line += "0: " + result.get_f1_result_blind(key) + "|"
                else:
                    line += result.get_f1_result_blind(key) + "|"
                
            f.write(line[:-1] + "\n")

def write_all_tp_results_blind(results, filename):
    # Open file and write header
    # mutation % | alg name #1 
    #     %      |     f1
    f = open("results/" + filename + "_all_tp_blind.csv", "w")
    f.write("Outlier %|Mutation %|")

    t = ""
    for i in results[0][0]:
        t += i.type + "|"

    t = t[:-1] + "\n"
    f.write(t)

    for res in results:
        for key in res[0][0].results_blind.keys():
            line = res[1] + "|"
            line += str(key) + "|"
            for result in res[0]:
                if type(result) == MultivariateDensity:
                    line += "0: " + result.get_all_tp_result_blind(key) + "|"
                else:
                    line += result.get_all_tp_result_blind(key) + "|"
                
            f.write(line[:-1] + "\n")

if __name__=="__main__":

    runs = 5
    mutation_perc_min = 0.05 # 5% mutation
    mutation_perc_max = 0.3 # 30% mutation
    mutation_min = 1.0
    mutation_max = 3.0
#    datasets = [QSARFishToxicity(),
#                GasEmission(),
#                AirQuality(),
#                CommunitiesAndCrime(),
#                Superconductivity(),
#                CTSliceAxial(),
#                Accelerometer(),
#                BiasCorrection(),
#                Cargo2000(),
#                Chickenpox(),
#                CNNPred(),
#                #HouseholdPowerConsumption(),
#                OnlineNewsPopularity(),
#                QueryAnalyticsWorkloads(),
#                WaveEnergyConverters(),
#                Synthetic(200, 1000000)] # 200 cols, 1million rows

    datasets = [Synthetic(10, 10000), Synthetic(50, 10000), Synthetic(100, 10000), Synthetic(200, 10000)]

    dataset_descriptions = {
        'Name' : [],
        'Total records' : [],
        'Total features' : [],
        'Total outliers' : []
    }

    table1 = {
        'Dataset mutation (*std)' : [],
        'Area under ROC' : []
    }

    table2 = {
        'Dataset mutation (*std)' : [],
        'False Positives' : []
    }

    midx1 = []
    midx2 = []

    data_sizes = []

    if not os.path.exists('data/'):
        os.makedirs('data/')

    if not os.path.exists('results/'):
        os.makedirs('results/')

    
    for dataset in datasets:
        name = type(dataset).__name__

        print("Loading %s data" % name)
        dataset.load_data()
        dataset.preprocess()
        
        results = []

        for mutation_perc in np.linspace(mutation_perc_min, mutation_perc_max, 10):
            data_sizes.append({"n": len(dataset.data), "features": len(dataset.columns)})
            
            evaluators = evaluate(dataset, runs, mutation_perc, mutation_min, mutation_max)

            # Dataset descriptions LaTeX table
            dataset_descriptions['Name'].append(name)
            dataset_descriptions['Total records'].append(len(dataset.data))
            dataset_descriptions['Total features'].append(len(dataset.columns))
            dataset_descriptions['Total outliers'].append(round(len(dataset.data)*mutation_perc))

            df0 = pd.DataFrame(dataset_descriptions)

            with open('results/results' + str(mutation_perc) + '.txt', 'w') as writer:
                writer.write(df0.to_latex(index=False))
                writer.close()

            plot_results_all_data(evaluators, name + "mutation_perc " + str(mutation_perc))
            plot_results_blind(evaluators, name + "mutation_perc " + str(mutation_perc))
            results.append( (evaluators, str(mutation_perc)) )

        write_f1_results_all_data(results, name)
        write_f1_results_blind(results, name)

        write_all_tp_results_all_data(results, name)
        write_all_tp_results_blind(results, name)

    print("Finished")