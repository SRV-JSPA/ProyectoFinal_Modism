import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class ConfiguracionMonteCarlo:
    activos: List[str]
    n_simulaciones: int = 1000
    n_pasos: int = 252  
    tiempo_anos: float = 1.0
    intensidad_contagio: float = 0.15
    umbral_shock: float = 2.0
    semilla_aleatoria: Optional[int] = 42
    incluir_contagio: bool = True

@dataclass  
class ParametrosFinancieros:
    precios_iniciales: np.ndarray
    rendimientos_esperados: np.ndarray
    volatilidades: np.ndarray
    matriz_correlacion: np.ndarray
    
    def validar(self):
        n = len(self.precios_iniciales)
        assert len(self.rendimientos_esperados) == n
        assert len(self.volatilidades) == n
        assert self.matriz_correlacion.shape == (n, n)
        
        
        eigenvals = np.linalg.eigvals(self.matriz_correlacion)
        if not np.all(eigenvals >= -1e-8):
            eigenvals = np.maximum(eigenvals, 0.01)
            eigenvecs = np.linalg.eigh(self.matriz_correlacion)[1]
            self.matriz_correlacion = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            D = np.sqrt(np.diag(self.matriz_correlacion))
            self.matriz_correlacion = self.matriz_correlacion / np.outer(D, D)

class ResultadosMonteCarlo:
    def __init__(self, config: ConfiguracionMonteCarlo, parametros: ParametrosFinancieros):
        self.config = config
        self.parametros = parametros
        self.precios_simulados = None  
        self.shocks_temporales = None  
        self.eventos_contagio = None   
        self.metricas_riesgo = None    
        self.matriz_contagio = None    
    
    def get_datos_componente_2(self) -> Dict:
        if self.precios_simulados is None:
            raise ValueError("Debe ejecutar simulación primero")

        precios = self.precios_simulados
        logret = np.diff(np.log(precios), axis=1)
        R = logret.reshape(-1, logret.shape[-1])
        corr = np.corrcoef(R, rowvar=False)
        N = corr.shape[0]
        nombres = list(self.config.activos)

        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        
        W = np.abs(corr)
        adj_emp = self._crear_red_empirica(W, N)
        
        
        adj_ba = self._crear_red_ba(N)
        
        
        hubs_emp = self._identificar_hubs(adj_emp, nombres)
        
        return {
            'adj_empirica': adj_emp,
            'adj_ba': adj_ba,
            'hubs_empiricos': hubs_emp,
            'activos': nombres
        }
    
    def _crear_red_empirica(self, W, N):
        w_dis = 1.0 - W
        adj_emp = np.zeros((N, N), dtype=int)
        
        
        in_tree = np.zeros(N, dtype=bool)
        in_tree[0] = True
        
        for _ in range(N - 1):
            best_w = float('inf')
            best_i = best_j = -1
            
            for i in range(N):
                if not in_tree[i]:
                    continue
                for j in range(N):
                    if in_tree[j] or i == j:
                        continue
                    if w_dis[i, j] < best_w:
                        best_w = w_dis[i, j]
                        best_i, best_j = i, j
            
            adj_emp[best_i, best_j] = adj_emp[best_j, best_i] = 1
            in_tree[best_j] = True
        
        
        k_target = max(2, int(np.sqrt(N)))
        M_target = int(N * k_target / 2)
        
        edges_added = adj_emp.sum() // 2
        Iu, Ju = np.triu_indices(N, k=1)
        strengths = W[Iu, Ju]
        order = np.argsort(-strengths)
        
        for idx in order:
            if edges_added >= M_target:
                break
            i, j = Iu[idx], Ju[idx]
            if adj_emp[i, j] == 0:
                adj_emp[i, j] = adj_emp[j, i] = 1
                edges_added += 1
        
        return adj_emp
    
    def _crear_red_ba(self, N):
        m = max(1, int(np.sqrt(N)) // 2)
        rng = np.random.default_rng(self.config.semilla_aleatoria or 42)
        
        adj = np.zeros((N, N), dtype=int)
        deg = np.zeros(N, dtype=int)
        
        
        for i in range(m):
            for j in range(i + 1, m):
                adj[i, j] = adj[j, i] = 1
                deg[i] += 1
                deg[j] += 1
        
        
        for new_node in range(m, N):
            
            probs = deg[:new_node] / deg[:new_node].sum() if deg[:new_node].sum() > 0 else np.ones(new_node) / new_node
            targets = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
            
            for target in targets:
                adj[new_node, target] = adj[target, new_node] = 1
                deg[new_node] += 1
                deg[target] += 1
        
        return adj
    
    def _identificar_hubs(self, adj, nombres):
        grados = adj.sum(axis=1)
        top_indices = np.argsort(-grados)[:5]
        return [{'activo': nombres[i], 'grado': int(grados[i])} for i in top_indices]

    def ejecutar_modelo_SIR(self) -> Dict:
        if self.precios_simulados is None or self.shocks_temporales is None:
            raise ValueError("Debe ejecutar simulación primero")
        
        rng = np.random.default_rng(self.config.semilla_aleatoria)
        beta = self.config.intensidad_contagio
        gamma = 0.10
        umbral = self.config.umbral_shock
        
        N = len(self.config.activos)
        T = self.shocks_temporales.shape[1]
        
        
        S = np.ones((T, N), dtype=bool)
        I = np.zeros((T, N), dtype=bool)
        R = np.zeros((T, N), dtype=bool)
        
        
        vol_inicial = np.random.rand(N)
        I[0, :] = vol_inicial > 0.8
        S[0, :] = ~I[0, :]
        
        transmisiones = []
        secundarios_por_src = np.zeros(N, dtype=int)
        
        
        for t in range(1, T):
            S_prev, I_prev, R_prev = S[t-1, :].copy(), I[t-1, :].copy(), R[t-1, :].copy()
            
            
            rec = (rng.random(N) < gamma) & I_prev
            I_now = I_prev & (~rec)
            R_now = R_prev | rec
            
            
            if I_prev.any() and self.matriz_contagio is not None:
                p_ij = 1.0 - np.exp(-beta * self.matriz_contagio)
                new_inf = np.zeros(N, dtype=bool)
                
                for i in np.where(I_prev)[0]:
                    for j in np.where(S_prev)[0]:
                        if rng.random() < p_ij[i, j]:
                            new_inf[j] = True
                            transmisiones.append((t, int(i), int(j)))
                            secundarios_por_src[i] += 1
                
                I_now |= new_inf
                S_now = S_prev & (~new_inf)
            else:
                S_now = S_prev
            
            
            shocks_extremos = np.max(np.abs(self.shocks_temporales[:, t-1, :]), axis=0) > umbral
            I_now |= (shocks_extremos & S_now)
            S_now &= ~shocks_extremos
            
            S[t, :], I[t, :], R[t, :] = S_now, I_now, R_now
        
        
        orden = np.argsort(-secundarios_por_src)
        superpropagadores = [{'activo': self.config.activos[i], 'infecciones_secundarias': int(secundarios_por_src[i])} 
                           for i in orden if secundarios_por_src[i] > 0]
        
        return {
            'S': S, 'I': I, 'R': R,
            'superpropagadores': superpropagadores,
            'transmisiones': transmisiones
        }

    def preparar_datos_lstm(self, activo_objetivo: Optional[str] = None):
        if self.precios_simulados is None:
            raise ValueError("Debe ejecutar simulación primero")
        
        
        precios_promedio = np.mean(self.precios_simulados, axis=0)
        df_precios = pd.DataFrame(precios_promedio, columns=self.config.activos)
        
        
        if activo_objetivo is None:
            activo_objetivo = self.config.activos[0]
        
        
        df_rendimientos = df_precios.pct_change().fillna(0)
        df_volatilidad = df_rendimientos.rolling(21).std().fillna(0)
        
        
        precio = df_precios[activo_objetivo].values.reshape(-1, 1)
        rendimiento = df_rendimientos[activo_objetivo].values.reshape(-1, 1)
        volatilidad = df_volatilidad[activo_objetivo].values.reshape(-1, 1)
        
        dataset = np.concatenate([precio, rendimiento, volatilidad], axis=1)
        
        
        lookback = 60
        horizonte = 20
        
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(dataset)
        y_scaled = scaler_y.fit_transform(precio)
        
        
        X_seq, y_seq = [], []
        for t in range(lookback, len(dataset) - horizonte):
            X_seq.append(X_scaled[t - lookback:t, :])
            y_seq.append(y_scaled[t + horizonte - 1, 0])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq).reshape(-1, 1)
        
        
        n_train = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:n_train], X_seq[n_train:]
        y_train, y_test = y_seq[:n_train], y_seq[n_train:]
        
        return X_train, X_test, y_train, y_test, scaler_y, activo_objetivo

    def entrenar_lstm(self, activo_objetivo: Optional[str] = None, epochs: int = 10, verbose: int = 1):
        X_train, X_test, y_train, y_test, scaler_y, activo = self.preparar_datos_lstm(activo_objetivo)
        
        
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        
        model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, 
                 batch_size=32, verbose=verbose)
        
        
        y_pred = model.predict(X_test)
        y_test_inv = scaler_y.inverse_transform(y_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        
        rmse = float(np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2)))
        mae = float(np.mean(np.abs(y_test_inv - y_pred_inv)))
        
        print(f"\nLSTM entrenado para activo {activo}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        
        self.datos_lstm = {
            'y_test_inv': y_test_inv,
            'y_pred_inv': y_pred_inv,
            'activo': activo
        }
        
        return model, {'rmse': rmse, 'mae': mae}

    def graficar_predicciones_lstm(self, n_puntos: int = 100):
        if not hasattr(self, 'datos_lstm'):
            print("Primero entrena el LSTM")
            return
        
        y_test = self.datos_lstm['y_test_inv'].flatten()
        y_pred = self.datos_lstm['y_pred_inv'].flatten()
        n = min(n_puntos, len(y_test))
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_test[:n], label='Real', linewidth=2)
        plt.plot(y_pred[:n], label='Predicho', linewidth=2, linestyle='--')
        plt.title(f"Predicción LSTM - {self.datos_lstm['activo']}")
        plt.xlabel("Muestra (test)")
        plt.ylabel("Precio")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

class SimuladorMonteCarlo:
    def __init__(self, config: Optional[ConfiguracionMonteCarlo] = None):
        self.config = config
        self.parametros = None
        self.resultados = None
        
        if config and config.semilla_aleatoria:
            np.random.seed(config.semilla_aleatoria)
    
    def crear_datos_ejemplo(self) -> Tuple[ConfiguracionMonteCarlo, ParametrosFinancieros]:
        activos = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'GLD',      
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
            'JPM', 'BAC', 'WFC', 'GS', 'MS',         
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',      
            'XOM', 'CVX', 'COP', 'SLB', 'EOG'        
        ]
        
        n_activos = len(activos)
        np.random.seed(42)
        
        
        precios_iniciales = np.random.uniform(50, 500, n_activos)
        rendimientos_esperados = np.random.uniform(-0.05, 0.15, n_activos)
        volatilidades = np.random.uniform(0.15, 0.60, n_activos)
        
        
        correlacion = self._crear_matriz_correlacion_sectorial(n_activos)
        
        config = ConfiguracionMonteCarlo(activos=activos) if self.config is None else self.config
        parametros = ParametrosFinancieros(
            precios_iniciales=precios_iniciales,
            rendimientos_esperados=rendimientos_esperados,
            volatilidades=volatilidades,
            matriz_correlacion=correlacion
        )
        parametros.validar()
        
        return config, parametros
    
    def _crear_matriz_correlacion_sectorial(self, n_activos):
        correlacion = np.eye(n_activos)
        
        sectores = {
            'ETFs': range(0, 5),
            'Tech': range(5, 10),
            'Financieros': range(10, 15),
            'Salud': range(15, 20),
            'Energía': range(20, 25)
        }
        
        
        for indices in sectores.values():
            for i in indices:
                for j in indices:
                    if i != j:
                        correlacion[i, j] = np.random.uniform(0.4, 0.8)
        
        
        for i in range(n_activos):
            for j in range(n_activos):
                if correlacion[i, j] == 0:
                    correlacion[i, j] = correlacion[j, i] = np.random.uniform(-0.2, 0.3)
        
        
        eigenvals, eigenvecs = np.linalg.eigh(correlacion)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlacion = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        D = np.sqrt(np.diag(correlacion))
        correlacion = correlacion / np.outer(D, D)
        
        return correlacion
    
    def _crear_matriz_contagio(self, correlacion: np.ndarray) -> np.ndarray:
        matriz = np.abs(correlacion)
        np.fill_diagonal(matriz, 0)
        
        
        row_sums = matriz.sum(axis=1)
        row_sums[row_sums == 0] = 1
        matriz = matriz / row_sums[:, np.newaxis]
        
        return matriz
    
    def _aplicar_contagio(self, shocks: np.ndarray, matriz_contagio: np.ndarray) -> np.ndarray:
        shocks_modificados = shocks.copy()
        n_activos = len(shocks)
        
        activos_shock = np.abs(shocks) > self.config.umbral_shock
        
        for i in range(n_activos):
            if activos_shock[i]:
                for j in range(n_activos):
                    if i != j and matriz_contagio[i, j] > 0:
                        contagio = (matriz_contagio[i, j] * 
                                  self.config.intensidad_contagio * 
                                  shocks[i])
                        shocks_modificados[j] += contagio
        
        return shocks_modificados
    
    def _detectar_eventos_contagio(self, shocks_temporales: np.ndarray) -> List[Dict]:
        eventos = []
        
        for sim in range(shocks_temporales.shape[0]):
            for t in range(shocks_temporales.shape[1]):
                shocks_extremos = np.abs(shocks_temporales[sim, t, :]) > self.config.umbral_shock
                
                if np.any(shocks_extremos):
                    activos_afectados = np.where(shocks_extremos)[0]
                    eventos.append({
                        'simulacion': sim,
                        'tiempo': t,
                        'activos_origen': activos_afectados.tolist(),
                        'intensidad_maxima': np.max(np.abs(shocks_temporales[sim, t, :]))
                    })
        
        return eventos
    
    def _calcular_metricas_riesgo(self, precios: np.ndarray) -> Dict:
        pesos = np.ones(precios.shape[2]) / precios.shape[2]
        valor_portafolio = np.sum(precios * pesos, axis=2)
        valor_final = valor_portafolio[:, -1]
        valor_inicial = valor_portafolio[:, 0]
        rendimientos = (valor_final - valor_inicial) / valor_inicial
        
        return {
            'rendimiento_promedio': np.mean(rendimientos),
            'volatilidad': np.std(rendimientos),
            'var_95': np.percentile(rendimientos, 5),
            'var_99': np.percentile(rendimientos, 1),
            'cvar_95': np.mean(rendimientos[rendimientos <= np.percentile(rendimientos, 5)]),
            'prob_perdida': np.mean(rendimientos < 0),
            'sharpe_ratio': np.mean(rendimientos) / np.std(rendimientos) if np.std(rendimientos) > 0 else 0,
        }
    
    def ejecutar_simulacion(self, 
                          config: Optional[ConfiguracionMonteCarlo] = None,
                          parametros: Optional[ParametrosFinancieros] = None) -> ResultadosMonteCarlo:
        
        if config is None and parametros is None:
            config, parametros = self.crear_datos_ejemplo()
        
        self.config = config or self.config
        self.parametros = parametros
        
        print(f"Simulando {len(self.config.activos)} activos")
        print(f"{self.config.n_simulaciones:,} simulaciones x {self.config.n_pasos:,} pasos")
        
        
        self.resultados = ResultadosMonteCarlo(self.config, self.parametros)
        
        
        matriz_contagio = self._crear_matriz_contagio(self.parametros.matriz_correlacion)
        self.resultados.matriz_contagio = matriz_contagio
        
        
        dt = self.config.tiempo_anos / self.config.n_pasos
        n_activos = len(self.config.activos)
        L = cholesky(self.parametros.matriz_correlacion, lower=True)
        
        
        precios = np.zeros((self.config.n_simulaciones, self.config.n_pasos + 1, n_activos))
        shocks_temporales = np.zeros((self.config.n_simulaciones, self.config.n_pasos, n_activos))
        precios[:, 0, :] = self.parametros.precios_iniciales
        
        
        for sim in range(self.config.n_simulaciones):
            for t in range(1, self.config.n_pasos + 1):
                
                shocks_independientes = np.random.standard_normal(n_activos)
                shocks_correlacionados = L @ shocks_independientes
                
                
                if self.config.incluir_contagio:
                    shocks_finales = self._aplicar_contagio(shocks_correlacionados, matriz_contagio)
                else:
                    shocks_finales = shocks_correlacionados
                
                shocks_temporales[sim, t-1, :] = shocks_finales
                
                
                drift = (self.parametros.rendimientos_esperados - 0.5 * self.parametros.volatilidades**2) * dt
                difusion = self.parametros.volatilidades * np.sqrt(dt) * shocks_finales
                precios[sim, t, :] = precios[sim, t-1, :] * np.exp(drift + difusion)
        
        
        self.resultados.precios_simulados = precios
        self.resultados.shocks_temporales = shocks_temporales
        self.resultados.eventos_contagio = self._detectar_eventos_contagio(shocks_temporales)
        self.resultados.metricas_riesgo = self._calcular_metricas_riesgo(precios)
        
        print(f"Eventos de contagio detectados: {len(self.resultados.eventos_contagio)}")
        print(f"VaR 95%: {self.resultados.metricas_riesgo['var_95']*100:.2f}%")
        
        return self.resultados
    
    def mostrar_resumen(self):
        if self.resultados is None:
            print("No hay resultados disponibles.")
            return
        
        print("Resultados de Monte Carlo")
        print(f"\nCONFIGURACIÓN:")
        print(f"  Activos: {len(self.config.activos)}")
        print(f"  Simulaciones: {self.config.n_simulaciones:,}")
        print(f"  Horizonte: {self.config.tiempo_anos} años")
        print(f"  Contagio: {'Activado' if self.config.incluir_contagio else 'Desactivado'}")
        
        print(f"\nMétricas de riesgo:")
        metricas = self.resultados.metricas_riesgo
        for nombre, valor in metricas.items():
            if isinstance(valor, float):
                if 'ratio' in nombre.lower():
                    print(f"  {nombre.replace('_', ' ').title()}: {valor:.3f}")
                else:
                    print(f"  {nombre.replace('_', ' ').title()}: {valor*100:.2f}%")
        
        print(f"\nAnálisis de contagio:")
        print(f"  Eventos detectados: {len(self.resultados.eventos_contagio)}")
        if self.resultados.eventos_contagio:
            intensidades = [e['intensidad_maxima'] for e in self.resultados.eventos_contagio]
            print(f"  Intensidad promedio: {np.mean(intensidades):.2f}")
            print(f"  Intensidad máxima: {np.max(intensidades):.2f}")
    
    def visualizar_resultados(self):
        if self.resultados is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        precios = self.resultados.precios_simulados
        pesos = np.ones(precios.shape[2]) / precios.shape[2]
        valor_portafolio = np.sum(precios * pesos, axis=2)
        
        
        for i in range(min(100, precios.shape[0])):
            axes[0,0].plot(valor_portafolio[i], alpha=0.1, color='blue')
        valor_promedio = np.mean(valor_portafolio, axis=0)
        axes[0,0].plot(valor_promedio, 'red', linewidth=2, label='Promedio')
        axes[0,0].set_title('Evolución del Portafolio')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        
        valor_final = valor_portafolio[:, -1]
        valor_inicial = valor_portafolio[:, 0]
        rendimientos = (valor_final - valor_inicial) / valor_inicial
        
        axes[0,1].hist(rendimientos * 100, bins=50, alpha=0.7, color='skyblue')
        axes[0,1].axvline(self.resultados.metricas_riesgo['var_95'] * 100, 
                         color='red', linestyle='--', label='VaR 95%')
        axes[0,1].set_title('Distribución de Rendimientos')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        
        im = axes[1,0].imshow(self.parametros.matriz_correlacion, cmap='RdBu', vmin=-1, vmax=1)
        axes[1,0].set_title('Matriz de Correlación')
        plt.colorbar(im, ax=axes[1,0])
        
        
        if self.resultados.eventos_contagio:
            tiempos = [e['tiempo'] for e in self.resultados.eventos_contagio]
            intensidades = [e['intensidad_maxima'] for e in self.resultados.eventos_contagio]
            
            axes[1,1].scatter(tiempos, intensidades, alpha=0.6, color='orange')
            axes[1,1].axhline(y=self.config.umbral_shock, color='red', 
                             linestyle='--', label=f'Umbral ({self.config.umbral_shock})')
            axes[1,1].set_title('Eventos de Contagio')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    simulador = SimuladorMonteCarlo()
    resultados = simulador.ejecutar_simulacion()
    simulador.mostrar_resumen()
    
    
    datos_redes = resultados.get_datos_componente_2()
    print(f"\nHubs principales: {datos_redes['hubs_empiricos'][:3]}")
    
    
    datos_sir = resultados.ejecutar_modelo_SIR()
    print(f"\nSuperpropagadores: {datos_sir['superpropagadores'][:3]}")
    
    
    modelo, metricas = resultados.entrenar_lstm(epochs=10)
    
    
    simulador.visualizar_resultados()
    resultados.graficar_predicciones_lstm(n_puntos=80)