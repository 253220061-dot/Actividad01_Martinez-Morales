import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURACIÓN Y PARÁMETROS GENERALES
# ==============================================================================
SEED = 42
np.random.seed(SEED)

N_ASSETS = 5
ASSET_NAMES = [f"Activo_{i+1}" for i in range(N_ASSETS)]

# Parámetros del modelo matemático de inversión
LAMBDA_RISK = 0.5   # Equilibra riesgo vs. rendimiento
RHO_PENALTY = 10.0  # Factor de penalización por violación de restricciones
W_MIN, W_MAX = 0.05, 0.40  # Límites de exposición por activo [5%, 40%]

# Parámetros del algoritmo evolutivo
POP_SIZE = 30
GENERATIONS = 50
P_CROSSOVER = 0.85
P_MUTATION = 0.20
SIGMA_MUT = 0.05
ELITISM = 1

# ==============================================================================
# 2. GENERACIÓN Y VALIDACIÓN DE DATOS SINTÉTICOS DE ENTRADA
# ==============================================================================
# Rendimientos esperados por activo (mu)
MU = np.array([0.12, 0.18, 0.10, 0.15, 0.08])

# Generación de matriz de covarianzas simétrica y semidefinida positiva (Sigma)
A = np.random.randn(N_ASSETS, N_ASSETS) * 0.05
SIGMA = np.dot(A, A.T) + np.diag([0.02, 0.04, 0.015, 0.03, 0.01])

# ==============================================================================
# 3. FUNCIONES DE DECODIFICACIÓN, EVALUACIÓN Y REPARACIÓN
# ==============================================================================
def repair_and_decode(g):
    """
    Decodifica y repara el genotipo g para asegurar la suma unitaria
    y la adherencia estricta a los límites [W_MIN, W_MAX].
    """
    w = np.maximum(1e-6, g)
    w = w / np.sum(w)  # Normalización básica de presupuesto
    
    # Algoritmo de reparación iterativa de límites
    for _ in range(10):
        clipped = np.clip(w, W_MIN, W_MAX)
        excess = 1.0 - np.sum(clipped)
        if abs(excess) < 1e-5:
            w = clipped
            break
        # Redistribuir exceso entre elementos no saturados
        free_mask = (w > W_MIN) & (w < W_MAX)
        if np.sum(free_mask) == 0:
            w = clipped
            break
        clipped[free_mask] += excess * (clipped[free_mask] / np.sum(clipped[free_mask]))
        w = clipped
        
    return w / np.sum(w)

def transaction_costs(w):
    """Calcula costos proporcionales de transacción."""
    return 0.001 * np.sum(np.abs(w))

def constraint_violation(w):
    """Calcula la medida agregada de violación V(w)."""
    v_min = np.sum(np.maximum(0.0, W_MIN - w))
    v_max = np.sum(np.maximum(0.0, w - W_MAX))
    return v_min + v_max

def evaluate_portfolio(w):
    """
    Evalúa la función de costo penalizado J(w) según Ecuación (2).
    """
    risk = np.dot(w.T, np.dot(SIGMA, w))
    ret = np.dot(MU, w)
    cost = transaction_costs(w)
    viol = constraint_violation(w)
    
    J = LAMBDA_RISK * risk - (1.0 - LAMBDA_RISK) * ret + cost + RHO_PENALTY * viol
    return J, risk, ret, viol

# ==============================================================================
# 4. CICLO PRINCIPAL DEL ALGORITMO EVOLUTIVO
# ==============================================================================
def run_evolutionary_algorithm():
    # Inicialización de la población en la métrica del espacio
    pop = np.random.dirichlet(np.ones(N_ASSETS), size=POP_SIZE)
    
    best_cost_history = []
    mean_cost_history = []
    diversity_history = []
    
    for gen in range(GENERATIONS):
        # Decodificación y evaluación de la población
        decoded_pop = np.array([repair_and_decode(ind) for ind in pop])
        evals = [evaluate_portfolio(w) for w in decoded_pop]
        costs = np.array([e[0] for e in evals])
        
        # Almacenar métricas históricas
        best_idx = np.argmin(costs)
        best_cost_history.append(costs[best_idx])
        mean_cost_history.append(np.mean(costs))
        
        # Calcular diversidad genotípica media D_port (Ecuación 3)
        diffs = []
        for i in range(POP_SIZE):
            for j in range(i + 1, POP_SIZE):
                diffs.append(np.linalg.norm(decoded_pop[i] - decoded_pop[j]))
        D_port = (2.0 / (POP_SIZE * (POP_SIZE - 1))) * np.sum(diffs)
        diversity_history.append(D_port)
        
        # Estrategia de reemplazo con Elitismo Fuerte
        next_pop = [pop[best_idx].copy()]
        
        while len(next_pop) < POP_SIZE:
            # Torneo Binario (k=2)
            i1, i2 = np.random.choice(POP_SIZE, 2, replace=False)
            p1 = pop[i1] if costs[i1] < costs[i2] else pop[i2]
            
            i3, i4 = np.random.choice(POP_SIZE, 2, replace=False)
            p2 = pop[i3] if costs[i3] < costs[i4] else pop[i4]
            
            # Recombinación Aritmética
            if np.random.rand() < P_CROSSOVER:
                alpha = np.random.rand()
                child = alpha * p1 + (1.0 - alpha) * p2
            else:
                child = p1.copy()
                
            # Mutación Gaussiana
            if np.random.rand() < P_MUTATION:
                child += np.random.normal(0, SIGMA_MUT, size=N_ASSETS)
                child = np.maximum(1e-4, child)
                
            next_pop.append(child)
            
        pop = np.array(next_pop)
        
    # Extracción de la mejor solución final
    final_decoded = np.array([repair_and_decode(ind) for ind in pop])
    final_evals = [evaluate_portfolio(w) for w in final_decoded]
    final_costs = [e[0] for e in final_evals]
    best_final_idx = np.argmin(final_costs)
    
    best_w = final_decoded[best_final_idx]
    best_J, best_risk, best_ret, best_viol = final_evals[best_final_idx]
    
    return best_w, best_J, best_risk, best_ret, best_viol, best_cost_history, mean_cost_history, diversity_history

# ==============================================================================
# 5. EJECUCIÓN Y DESPLIEGUE DE RESULTADOS
# ==============================================================================
if __name__ == "__main__":
    # Ejecutar optimización
    best_w, best_J, best_risk, best_ret, best_viol, best_costs, mean_costs, diversity = run_evolutionary_algorithm()

    # Imprimir resumen de resultados en la consola
    print("\n" + "=" * 65)
    print("       RESULTADOS DE LA OPTIMIZACIÓN DE CARTERA (EJERCICIO 7.1)")
    print("=" * 65)
    print(f"Rendimiento Esperado (mu^T w) : {best_ret * 100:.2f}%")
    print(f"Riesgo / Varianza (w^T Sigma w): {best_risk:.6f} (Desv. Est.: {np.sqrt(best_risk) * 100:.2f}%)")
    print(f"Costo Penalizado J(w)          : {best_J:.6f}")
    print(f"Violación de Restricciones V(w): {best_viol:.6f}")
    print("-" * 65)
    print("Asignación Óptima de Capital por Activo:")
    for name, weight in zip(ASSET_NAMES, best_w):
        print(f"  • {name}: {weight * 100:.2f}%")
    print("=" * 65 + "\n")

    # Visualización gráfica de métricas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Gráfica 1: Convergencia de J(w)
    ax1.plot(best_costs, label="Mejor Solución", color="#8E24AA", linewidth=2)
    ax1.plot(mean_costs, label="Promedio Poblacional", color="#FF9800", linestyle="--", linewidth=1.5)
    ax1.set_title("Convergencia del Costo Penalizado J(w)")
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Costo J(w)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gráfica 2: Evolución de la Diversidad D_port
    ax2.plot(diversity, color="#00BCD4", linewidth=2)
    ax2.set_title("Evolución de la Diversidad Genotípica D_port")
    ax2.set_xlabel("Generación")
    ax2.set_ylabel("Diversidad Media por Pares")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()