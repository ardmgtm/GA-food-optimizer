import plotille
import pandas as pd
import numpy as np
import random
import time
from tabulate import tabulate

class GeneticAlgorithm:
    #read csv
    df_nutrisi = pd.read_csv('data_nutrisi.csv')

    def __init__(self,target,pop_size,cr,mr,num_generation):
        self.target = target
        self.pop_size = pop_size
        self.cr = cr
        self.mr = mr
        self.num_generation = num_generation
        self.best_fitness = []
        self.avg_fitness = []

    #Initialize Population
    def popInit(self):
        self.pop = np.zeros((self.pop_size,len(GeneticAlgorithm.df_nutrisi.index)))
        for x in self.pop:
            x[random.randint(0,len(GeneticAlgorithm.df_nutrisi.index)-1)] = 1
        return self.pop

    #get fitness from chromosome
    def getFitness(self,chromosome):
        index = np.where(chromosome==1)[0]
        selected_food = GeneticAlgorithm.df_nutrisi.loc[index,['Protein','Lemak','Karbohidrat']].values

        return 10000/np.sum(np.abs(np.sum(selected_food,axis=0)-np.array(self.target)))

    #one-cut-point crossover
    def crossover(self,p1,p2):
        cut_point = random.randint(1,len(GeneticAlgorithm.df_nutrisi.index)-1)
        c1 = np.concatenate((p1[:cut_point],p2[cut_point:]))
        c2 = np.concatenate((p2[:cut_point],p1[cut_point:]))
        return c1,c2

    def mutation(self,p):
        c = p.copy()
        selected_point = random.randint(0,len(GeneticAlgorithm.df_nutrisi.index)-1)
        c[selected_point] = 1 if c[selected_point]==0 else 0
        return c

    #elitsm selection
    def selection(self):
        fitness = [self.getFitness(indv) for indv in self.pop]
        rank = sorted( [(x,i) for (i,x) in enumerate(fitness)], reverse=True )[:self.pop_size]
        selected_index = np.array(rank,dtype=np.int)[:,1]
        self.pop = self.pop[selected_index]

        self.best_indv = self.pop[0]
        self.best_fitness.append(fitness[selected_index[0]])
        self.avg_fitness.append(sum(fitness)/len(fitness))

    #print best solution / individu
    def printBest(self):
        index = np.where(self.best_indv==1)[0]
        selected_food = GeneticAlgorithm.df_nutrisi.loc[index,['Nama','Protein','Lemak','Karbohidrat']]
        tot_prot = round(selected_food['Protein'].sum(),2)
        tot_fat = round(selected_food['Lemak'].sum(),2)
        tot_carb = round(selected_food['Karbohidrat'].sum(),2)
        cal = lambda x:x[0]*4+x[1]*9+x[2]*4

        print(tabulate(
            selected_food,
            showindex=False,
            headers=selected_food.columns,
            tablefmt="psql",
            floatfmt=".2f"
        ))
        print("\nTotal \t= {0} = {1:.2f} kal".format( str([tot_prot,tot_fat,tot_carb]), cal([tot_prot,tot_fat,tot_carb]) ))
        print("Target \t= {0} = {1:.2f} kal".format( str(self.target), cal(self.target) ))

    #start genetic algorithm
    def optimize(self):
        self.popInit()
        print("\nstart evolving ...")
        for i in range(self.num_generation):
            offspring = []
            num_crossover = int(self.cr * self.pop_size // 2)
            num_mutation = int(self.mr * self.pop_size // 2)
            for n in range(num_crossover):
                p1 = self.pop[random.randint(0,self.pop_size-1)]
                p2 = self.pop[random.randint(0,self.pop_size-1)]
                c1,c2 = self.crossover(p1,p2)
                offspring.append(c1)
                offspring.append(c2)
            for m in range(num_mutation):
                p = self.pop[random.randint(0,self.pop_size-1)]
                c = self.mutation(p)
                offspring.append(c)
            self.pop = np.concatenate((self.pop,offspring))
            self.selection()
            progress = (i+1)/self.num_generation
            print("\rProgress: [{0:30s}] {1}/{2} generation".format(chr(10294) * int(progress * 30), i+1,self.num_generation), end="", flush=True)

def getAKG(isMale,weight,height,age,sportFreq):
    tot_kal = 0

    if(isMale):tot_kal=66+(13.7*weight)+(5*height)-(6.8*age)
    else:tot_kal=655+(9.6*weight)+(1.8*height)-(4.7*age)

    if(sportFreq<1):tot_kal*=1.2
    elif(sportFreq<4):tot_kal*=1.375
    elif(sportFreq<6):tot_kal*=1.55
    elif(sportFreq<8):tot_kal*=1.725
    else:tot_kal*=1.9

    prot = round(0.03*tot_kal,2)
    fat = round(0.04*tot_kal,2)
    carb = round(0.15*tot_kal,2)

    return [prot,fat,carb]
    
if __name__ == "__main__":
    gender = input("Jenis Kelamin (l/p)\t: ")
    weight = input("Berat Badan\t\t: ")
    height = input("Tinggi Badan\t\t: ")
    age = input("Umur\t\t\t: ")
    sport_freq = input("Frek. Olahraga/minggu\t: ")

    start_time = time.time()

    akg = getAKG(gender=="l",int(weight),int(height),int(age),int(sport_freq))

    pop_size = 100
    num_gen = 100
    cr = 0.8
    mr = 0.2

    ga = GeneticAlgorithm(akg,pop_size,cr,mr,num_gen)
    ga.optimize()
    print("\nResult\t:\n")
    ga.printBest()

    print("\n")
    fig = plotille.Figure()
    fig.width = 50
    fig.height = 20
    fig.x_label = "gen"
    fig.y_label = "fitness"
    fig.set_x_limits(min_=0 , max_=num_gen )
    fig.set_y_limits(min_=0 , max_= 100 + round(max(ga.best_fitness)) - round(max(ga.best_fitness))%-100 )
    fig.color_mode = 'byte'
    fig.plot(range(num_gen),ga.best_fitness, lc=25, label='Best Fitness')
    fig.plot(range(num_gen),ga.avg_fitness, lc=200, label='Average Fitness')
    print(fig.show(legend=True))

    print("\n(generation = {0}, population size = {1}, cr = {2}, mr = {3})".format(num_gen,pop_size,cr,mr))
    print("--- runtime finish in {0:.4f} seconds ---".format(time.time() - start_time))

