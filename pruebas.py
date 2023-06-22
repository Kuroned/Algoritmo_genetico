import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Etiquetas para las direcciones
directions = {0:'up', 1:'down', 2:'left', 3:'right', 4:'up_right', 5:'up_left', 6:'down_right', 7:'down_left'}

# Función que actualiza la posición según la dirección elegida
def move(pos, direction):
    if direction == 'up':
        return (pos[0], max(0, pos[1]-1))
    elif direction == 'down':
        return (pos[0], min(19, pos[1]+1))
    elif direction == 'left':
        return (max(0, pos[0]-1), pos[1])
    elif direction == 'right':
        return (min(19, pos[0]+1), pos[1])
    elif direction == 'up_right':
        return (min(19, pos[0]+1), max(0, pos[1]-1))
    elif direction == 'up_left':
        return (max(0, pos[0]-1), max(0, pos[1]-1))
    elif direction == 'down_right':
        return (min(19, pos[0]+1), min(19, pos[1]+1))
    elif direction == 'down_left':
        return (max(0, pos[0]-1), min(19, pos[1]+1))
# Función de aptitud: premia a aquellos individuos que llegan más cerca del extremo derecho de la matriz
def fitness(pos):
    return pos[0]

# Función de reproducción: crea un nuevo individuo a partir de dos padres
def reproduce(parent1, parent2):
    # Genera una máscara aleatoria de booleanos
    mask = np.random.rand(8) > 0.5
    # Crea un nuevo individuo a partir de los padres, utilizando la máscara para decidir de qué padre proviene cada gen
    child = np.where(mask, parent1, parent2)
    # Normaliza los genes del hijo
    child /= np.sum(child)
    return child

def verify_step(pos,matrix):
    if matrix[valor_X-1,pos[1]] != 0:
        direction = directions[np.random.choice(8, p=population[i])]
        pos = move(pos, direction)
        verify_step(pos,matrix)
    else:
        return pos
# Número de generaciones
num_generations = 50

# Número de individuos en la población
num_individuals = 50

# Numero de pasos
num_steps = 40

# Genera la población inicial
population = np.random.rand(num_individuals, 8)
population /= np.sum(population, axis=1, keepdims=True)


# Crear una figura para la animación
fig, ax = plt.subplots()

# Crear una lista para guardar las aptitudes promedio en cada generación
average_fitnesses = []

# Crear una lista para guardar las posiciones finales en cada generación
final_positions_over_generations = []

#creacion matriz
valor_X = 20
valor_Y = 20
matrix = np.zeros((20, 20))

for generation in range(num_generations):
    # Crea una lista para guardar las posiciones finales de cada individuo
    final_positions = []
    
    # Deja que cada individuo se mueva, igual que antes
    for i in range(num_individuals):
        pos = (0, 0)
        for j in range(num_steps):
            direction = directions[np.random.choice(8, p=population[i])]
            pos = move(pos, direction)
            verify_step(pos,matrix)
    
        print("paso indiviudo")
        print(i)
        print(pos)
        matrix[pos[0]-1,pos[1]] = str(i)
    print("GENERACION")
    print(generation)
    if generation % 50 == 0 and num_steps > 25:
        num_steps -= 5
    # Calcula la aptitud de cada individuo
    fitnesses = [fitness(pos) for pos in final_positions]
    
    # Agrega la aptitud promedio a la lista
    average_fitnesses.append(np.mean(fitnesses))

    # Crea una matriz para guardar las posiciones finales
    matrix = np.zeros((20, 20))
    for pos in final_positions:
        matrix[pos[1], pos[0]] += 1
    final_positions_over_generations.append(matrix)



    # Selecciona a los individuos que llegaron al extremo derecho
    best_individuals = [ind for pos, ind in zip(final_positions, population) if pos[0] == 19]

    # Si ningún individuo llegó al extremo derecho, se genera una nueva población
    
    if not best_individuals:
        population = np.random.rand(num_individuals, 8)
        population /= np.sum(population, axis=1, keepdims=True)
        continue  # Salta el resto del ciclo y vuelve al inicio
    # Si ningún individuo llegó al extremo derecho o sólo uno lo hizo, se generan individuos aleatorios
    while len(best_individuals) < 2:
        random_individual = np.random.rand(8)
        random_individual /= np.sum(random_individual)
        best_individuals.append(random_individual)
        continue


    
    # Crea la nueva población
    new_population = []
    for i in range(num_individuals):
        # Elige dos índices al azar
        parent_indices = np.random.choice(len(best_individuals), size=2, replace=False).tolist()
        parent1, parent2 = best_individuals[parent_indices[0]], best_individuals[parent_indices[1]]

        # Crea un nuevo individuo a partir de los padres
        child = reproduce(parent1, parent2)
        new_population.append(child)
    
    # Reemplaza la antigua población con la nueva
    population = np.array(new_population)


# Función de inicialización para la animación
def init():
    im.set_data(final_positions_over_generations[0])
    return [im]

# Función de animación que se ejecutará en cada cuadro
def animate(i):
    im.set_data(final_positions_over_generations[i])
    return [im]

# Crea la animación usando las funciones anteriores
im = ax.imshow(final_positions_over_generations[0], cmap='Reds', interpolation='nearest')
ani = animation.FuncAnimation(fig, animate, frames=num_generations, init_func=init, blit=True)

# Guarda la animación como un archivo .gif
ani.save('evolution.gif', writer='pillow')
plt.show()


# Dibuja la aptitud promedio en cada generación
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(average_fitnesses)
plt.xlabel('Generation')
plt.ylabel('Average Fitness')

# Dibuja la matriz de posiciones finales para la última generación
plt.subplot(1,2,2)
plt.imshow(final_positions_over_generations[-1], cmap='Reds', interpolation='nearest')
plt.title('Final Positions in Last Generation')

plt.tight_layout()
plt.show()