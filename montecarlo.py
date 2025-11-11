import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import cholesky
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import pickle
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ConfiguracionMonteCarlo:
    activos: List[str]
    n_simulaciones: int = 1000
    n_pasos: int = 252  
    tiempo_anos: float = 1.0
    intensidad_contagio: float = 0.15
    umbral_shock: float = 2.0  # Shocks mayores a este valor se consideran "extremos"
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
        
        # Asegurar que la matriz de correlación sea semidefinida positiva
        eigenvals = np.linalg.eigvals(self.matriz_correlacion)
        if not np.all(eigenvals >= -1e-8):
            # Ajuste simple: subir eigenvalores negativos a un mínimo pequeño
            eigenvals = np.maximum(eigenvals, 0.01)
            eigenvecs = np.linalg.eigh(self.matriz_correlacion)[1]
            self.matriz_correlacion = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            # Re-normalizar para que la diagonal sea 1
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
        self.timestamp = datetime.now().isoformat()
    
    def get_datos_componente_2(self) -> Dict:
        import numpy as np
        import math

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
        w_dis = 1.0 - W
        adj_emp = np.zeros((N, N), dtype=int)
        in_tree = np.zeros(N, dtype=bool)
        in_tree[0] = True
        for _ in range(N - 1):
            best_i = -1
            best_j = -1
            best_w = math.inf
            for i in range(N):
                if not in_tree[i]:
                    continue
                for j in range(N):
                    if in_tree[j] or i == j:
                        continue
                    ww = w_dis[i, j]
                    if ww < best_w:
                        best_w = ww
                        best_i = i
                        best_j = j
            adj_emp[best_i, best_j] = 1
            adj_emp[best_j, best_i] = 1
            in_tree[best_j] = True

        k_target = max(2, int(round(np.sqrt(N))))
        M_target = int(N * k_target / 2)

        Iu, Ju = np.triu_indices(N, k=1)
        strengths = W[Iu, Ju]
        order = np.argsort(-strengths)
        edges_added = int(adj_emp.sum() // 2)
        for idx in order:
            if edges_added >= M_target:
                break
            i, j = Iu[idx], Ju[idx]
            if adj_emp[i, j] == 1:
                continue
            adj_emp[i, j] = 1
            adj_emp[j, i] = 1
            edges_added += 1

        W_emp = W * adj_emp

        grado_emp = adj_emp.sum(axis=1)
        m_ba = max(1, int(round(grado_emp.mean() / 2.0)))
        seed = int(self.config.semilla_aleatoria or 42)

        def ba_graph(n, m, seed_val):
            rng = np.random.default_rng(seed_val)
            deg = np.zeros(n, dtype=int)
            edges = set()
            for u in range(m):
                for v in range(u + 1, m):
                    edges.add((u, v))
                    deg[u] += 1
                    deg[v] += 1
            repeated = []
            for u in range(m):
                repeated.extend([u] * deg[u])
            for new in range(m, n):
                chosen = set()
                while len(chosen) < m and len(repeated) > 0:
                    chosen.add(int(rng.choice(repeated)))
                if len(chosen) < m:
                    pool = list(range(new))
                    while len(chosen) < m:
                        chosen.add(int(rng.choice(pool)))
                for t in chosen:
                    u, v = (new, t) if new < t else (t, new)
                    if (u, v) not in edges:
                        edges.add((u, v))
                        deg[new] += 1
                        deg[t] += 1
                repeated.extend(list(chosen))
                repeated.extend([new] * deg[new])
            adj = np.zeros((n, n), dtype=int)
            for (u, v) in edges:
                adj[u, v] = 1
                adj[v, u] = 1
            return adj, sorted(list(edges))

        adj_ba, edges_ba = ba_graph(N, m_ba, seed)

        def degree(A):
            return A.sum(axis=1)

        def clustering(A):
            k = degree(A)
            A3 = A @ A @ A
            tri_i = np.diag(A3) / 2.0
            denom = k * (k - 1) / 2.0
            with np.errstate(divide='ignore', invalid='ignore'):
                c_i = np.where(denom > 0, tri_i / denom, 0.0)
            return float(np.nanmean(c_i)), c_i

        def assortativity(A):
            k = degree(A)
            iu, ju = np.triu_indices(A.shape[0], k=1)
            mask = A[iu, ju] == 1
            di, dj = k[iu][mask], k[ju][mask]
            if di.size == 0:
                return 0.0
            return float(np.corrcoef(di, dj)[0, 1])

        def eigenvector_centrality(A, iters=1000, tol=1e-9):
            v = np.ones(A.shape[0], dtype=float)
            v /= np.linalg.norm(v)
            for _ in range(iters):
                v_new = A @ v
                norm = np.linalg.norm(v_new)
                if norm == 0:
                    break
                v_new /= norm
                if np.linalg.norm(v_new - v) < tol:
                    v = v_new
                    break
                v = v_new
            return v

        k_emp = degree(adj_emp)
        k_ba = degree(adj_ba)
        c_emp_avg, c_emp = clustering(adj_emp)
        c_ba_avg, c_ba = clustering(adj_ba)
        r_emp = assortativity(adj_emp)
        r_ba = assortativity(adj_ba)
        ev_emp = eigenvector_centrality(adj_emp)
        ev_ba = eigenvector_centrality(adj_ba)

        top_emp_idx = np.argsort(-k_emp)[:5]
        top_ba_idx = np.argsort(-k_ba)[:5]
        hubs_emp = [{'activo': nombres[i], 'grado': int(k_emp[i]), 'eigencentralidad': float(ev_emp[i])} for i in top_emp_idx]
        hubs_ba = [{'activo': nombres[i], 'grado': int(k_ba[i]), 'eigencentralidad': float(ev_ba[i])} for i in top_ba_idx]

        edges_emp = []
        iu, ju = np.triu_indices(N, k=1)
        for i, j in zip(iu, ju):
            if adj_emp[i, j] == 1:
                edges_emp.append({'source': int(i), 'target': int(j), 'peso': float(W_emp[i, j])})

        sigma = np.std(R, axis=0, ddof=1)

        def _minmax(x):
            x = np.asarray(x, dtype=float)
            mn, mx = np.min(x), np.max(x)
            if mx == mn:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        cv_vec = _minmax(ev_emp) * _minmax(sigma)
        rank_idx = np.argsort(-cv_vec)
        centralidad_vol_list = [{'activo': nombres[i], 'valor': float(cv_vec[i]), 'grado': int(k_emp[i]), 'eigencentralidad': float(ev_emp[i]), 'volatilidad': float(sigma[i])} for i in rank_idx]

        return {
            'activos': nombres,
            'matriz_correlacion': corr,
            'adj_empirica': adj_emp.astype(int),
            'W_empirica': W_emp,
            'edges_empiricos': edges_emp,
            'ba_params': {'n_nodos': int(N), 'm': int(m_ba), 'seed': seed},
            'adj_ba': adj_ba.astype(int),
            'edges_ba': [{'source': int(u), 'target': int(v)} for (u, v) in edges_ba],
            'metrics': {
                'avg_degree_emp': float(k_emp.mean()),
                'avg_clustering_emp': c_emp_avg,
                'assortativity_emp': r_emp,
                'avg_degree_ba': float(k_ba.mean()),
                'avg_clustering_ba': c_ba_avg,
                'assortativity_ba': r_ba
            },
            'hubs_empiricos': hubs_emp,
            'hubs_ba': hubs_ba,
            'centralidad_volatilidad': cv_vec,
            'centralidad_volatilidad_por_activo': centralidad_vol_list
        }

    def ejecutar_modelo_SIR(self,
                            beta: float = None,
                            gamma: float = 0.10,
                            umbral_infeccion: float = None,
                            pasos: Optional[int] = None,
                            red: str = 'contagio',
                            usar_empirica_con_pesos: bool = True,
                            semilla: Optional[int] = None) -> Dict:
        import numpy as np
        if self.precios_simulados is None or self.shocks_temporales is None:
            raise ValueError("Debe ejecutar simulación primero (precios y shocks).")
        rng = np.random.default_rng(self.config.semilla_aleatoria if semilla is None else semilla)
        beta = beta if beta is not None else float(self.config.intensidad_contagio)
        umbral = umbral_infeccion if umbral_infeccion is not None else float(self.config.umbral_shock)
        N = len(self.config.activos)
        T = pasos if pasos is not None else self.shocks_temporales.shape[1]

        # --- Matriz de pesos W ---
        if red == 'contagio':
            if self.matriz_contagio is None:
                raise ValueError("self.matriz_contagio no está definida.")
            W = np.array(self.matriz_contagio, dtype=float)
        elif red == 'empirica':
            datos2 = self.get_datos_componente_2()
            adj = np.array(datos2['adj_empirica'], dtype=int)
            if usar_empirica_con_pesos:
                W_emp = np.array(datos2['W_empirica'], dtype=float)
                W = W_emp / (W_emp.sum(axis=1, keepdims=True) + 1e-12)
            else:
                row_sums = adj.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                W = adj / row_sums
        else:
            raise ValueError("red debe ser 'contagio' o 'empirica'.")

        # --- Estados S/I/R ---
        S = np.ones((T, N), dtype=bool)
        I = np.zeros((T, N), dtype=bool)
        R = np.zeros((T, N), dtype=bool)

        centralidad_vol = self.get_datos_componente_2()['centralidad_volatilidad']
        thr0 = np.percentile(centralidad_vol, 80)
        I0 = (centralidad_vol > thr0)
        I[0, :] = I0
        S[0, :] = ~I0

        transmisiones = []
        secundarios_por_src = np.zeros(N, dtype=int)

        for t in range(1, T):
            S_prev, I_prev, R_prev = S[t-1, :].copy(), I[t-1, :].copy(), R[t-1, :].copy()

            # Recuperación
            rec = (rng.random(N) < gamma) & I_prev
            I_now = I_prev & (~rec)
            R_now = R_prev | rec

            # Infección por red
            if I_prev.any():
                p_ij = 1.0 - np.exp(-beta * W)
                inf_sources = np.where(I_prev)[0]
                sus = np.where(S_prev)[0]
                new_inf = np.zeros(N, dtype=bool)
                parent_candidates = {j: [] for j in sus}
                for i in inf_sources:
                    pij_row = p_ij[i, sus]
                    hits = rng.random(len(sus)) < pij_row
                    for idx_local, hit in enumerate(hits):
                        if hit:
                            j = sus[idx_local]
                            parent_candidates[j].append(i)
                for j in sus:
                    if parent_candidates[j]:
                        new_inf[j] = True
                        src = rng.choice(parent_candidates[j])
                        transmisiones.append((t, int(src), int(j)))
                        secundarios_por_src[src] += 1
            else:
                new_inf = np.zeros(N, dtype=bool)

            shocks_t = self.shocks_temporales[:, t-1, :]  
            max_abs = np.max(np.abs(shocks_t), axis=0)
            exog = max_abs > umbral
            new_inf |= (exog & S_prev)

            # Actualiza
            I_now |= new_inf
            S_now = S_prev & (~new_inf)
            S[t, :], I[t, :], R[t, :] = S_now, I_now, R_now

        I_count = I.sum(axis=1)
        new_cases = np.zeros(T, dtype=int)
        new_cases[0] = I[0, :].sum()
        new_cases[1:] = np.maximum(0, (I[1:, :].sum(axis=1) - I[:-1, :].sum(axis=1)))
        eps = 1e-9
        Rt = (new_cases[1:] / (I_count[:-1] + eps))
        if len(Rt) >= 7:
            from numpy.lib.stride_tricks import sliding_window_view
            w = min(7, len(Rt))
            Rt_suave = sliding_window_view(Rt, w).mean(axis=-1)
            Rt_plot = np.concatenate([Rt[:w-1], Rt_suave])
        else:
            Rt_plot = Rt

        orden = np.argsort(-secundarios_por_src)
        hubs = [{'activo': self.config.activos[i],
                'infecciones_secundarias': int(secundarios_por_src[i])}
                for i in orden if secundarios_por_src[i] > 0]

        resultados_sir = {
            'S': S, 'I': I, 'R': R,
            'nuevos_casos': new_cases,
            'infectados_totales_t': I_count,
            'Rt': Rt_plot,
            'transmisiones': transmisiones,
            'superpropagadores': hubs,
            'parametros': {'beta': beta, 'gamma': gamma, 'umbral': umbral, 'red': red},
        }
        self.datos_sir = resultados_sir
        return resultados_sir

    def graficar_SIR(self):
        import matplotlib.pyplot as plt
        if not hasattr(self, 'datos_sir'):
            print("Primero ejecuta ejecutar_modelo_SIR().")
            return
        S = self.datos_sir['S'].sum(axis=1)
        I = self.datos_sir['I'].sum(axis=1)
        R = self.datos_sir['R'].sum(axis=1)
        Rt = self.datos_sir['Rt']

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(S, label='S', linewidth=2)
        ax.plot(I, label='I', linewidth=2)
        ax.plot(R, label='R', linewidth=2)
        ax.set_title('Dinámica SIR (número de activos por estado)')
        ax.set_xlabel('Tiempo (pasos)')
        ax.set_ylabel('# Activos')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(Rt, linewidth=2)
        ax2.axhline(1.0, linestyle='--')
        ax2.set_title('Rt (estimado)')
        ax2.set_xlabel('Tiempo (pasos)')
        ax2.set_ylabel('Rt')
        ax2.grid(True, alpha=0.3)
        plt.show()

    
    def get_datos_componente_3(self) -> Dict:
        if self.shocks_temporales is None:
            raise ValueError("Debe ejecutar simulación primero")

        # Estados iniciales y superpropagadores 
        centralidad_vol = self.get_datos_componente_2()['centralidad_volatilidad']
        umbral_inicial = np.percentile(centralidad_vol, 80)
        susceptibles = (centralidad_vol <= umbral_inicial).astype(float)
        infectados = (centralidad_vol > umbral_inicial).astype(float)
        recuperados = np.zeros(len(self.config.activos))

        contador_origen = {}
        for evento in self.eventos_contagio:
            for activo_idx in evento['activos_origen']:
                activo = self.config.activos[activo_idx]
                contador_origen[activo] = contador_origen.get(activo, 0) + 1
        activos_ordenados = sorted(contador_origen.items(), key=lambda x: x[1], reverse=True)
        n_super = max(1, len(activos_ordenados) // 5)
        superpropagadores_pre = [activo for activo, _ in activos_ordenados[:n_super]]

        sir = self.ejecutar_modelo_SIR(
            beta=self.config.intensidad_contagio,
            gamma=0.10,
            umbral_infeccion=self.config.umbral_shock,
            red='contagio'  # Usamos la misma red de matriz_contagio
        )

        return {
            'shocks_temporales': self.shocks_temporales,
            'eventos_contagio': self.eventos_contagio,
            'matriz_transmision': self.matriz_contagio,
            'estados_sir_iniciales': {'S': susceptibles, 'I': infectados, 'R': recuperados},
            'superpropagadores_previos': superpropagadores_pre,       # top según eventos de contagio
            'sir_resultados': sir,                                    # trayectoria SIR completa
            'parametros_sir': {
                'beta': sir['parametros']['beta'],
                'gamma': sir['parametros']['gamma'],
                'umbral_infeccion': sir['parametros']['umbral'],
                'red': sir['parametros']['red']
            }
        }

    
    def get_datos_componente_4(self) -> Dict:
        if self.precios_simulados is None:
            raise ValueError("Debe ejecutar simulación primero")
        
        # 1) Promedio de precios simulados a través de las simulaciones
        precios_promedio = np.mean(self.precios_simulados, axis=0)
        
        # 2) Construimos índice temporal
        timestamps = pd.date_range(start='2024-01-01', periods=len(precios_promedio), freq='D')
        df_precios = pd.DataFrame(precios_promedio, index=timestamps, columns=self.config.activos)
        
        # 3) Rendimientos y volatilidad simple
        df_rendimientos = df_precios.pct_change().fillna(0)
        df_volatilidad = df_rendimientos.rolling(21).std().fillna(0)
        
        # 4) Volatilidades realizadas con distintas ventanas
        ventanas = [5, 10, 21, 63]
        vol_realizadas_dict = {}
        for ventana in ventanas:
            vol_ventana = df_rendimientos.rolling(ventana).std() * np.sqrt(252)
            vol_realizadas_dict[f'vol_{ventana}d'] = vol_ventana.mean(axis=1)  
        
        df_vol_realizadas = pd.DataFrame(vol_realizadas_dict, index=timestamps)
        
        # 5) Definimos regímenes de mercado a partir de la volatilidad promedio
        vol_promedio = df_vol_realizadas.mean(axis=1)
        umbrales = [np.percentile(vol_promedio.dropna(), p) for p in [25, 75, 95]]
        
        regimenes = []
        for vol in vol_promedio:
            if pd.isna(vol):
                regimenes.append('Normal')  
            elif vol <= umbrales[0]:
                regimenes.append('Bajo_Vol')
            elif vol <= umbrales[1]:
                regimenes.append('Normal')
            elif vol <= umbrales[2]:
                regimenes.append('Alto_Vol')
            else:
                regimenes.append('Crisis')
        
        # 6) Features para modelos de ML / Deep Learning
        features_ml = self._calcular_features_ml(df_precios, df_rendimientos)
        
        return {
            'series_temporales': {
                'precios': df_precios,
                'rendimientos': df_rendimientos,
                'volatilidad': df_volatilidad,
                'timestamps': timestamps
            },
            'volatilidades_realizadas': df_vol_realizadas,
            'regimenes_mercado': pd.DataFrame({
                'regimen': regimenes,
                'volatilidad_promedio': vol_promedio
            }, index=timestamps),
            'features_ml': features_ml,
            'parametros_lstm': {
                'secuencia_lookback': 60,
                'horizonte_prediccion': 20,
                'features_input': ['precio', 'rendimiento', 'volatilidad']
            },
            'parametros_clasificador': {
                'clases_regimen': ['Bajo_Vol', 'Normal', 'Alto_Vol', 'Crisis'],
                'umbrales_volatilidad': umbrales,
                'ventana_clasificacion': 21
            }
        }
    
    def _calcular_features_ml(self, precios: pd.DataFrame, rendimientos: pd.DataFrame) -> pd.DataFrame:
        features = {}
        
        # 1) RSI promedio (sobre todos los activos)
        features['rsi_promedio'] = self._calcular_rsi(precios).mean(axis=1)
        
        # 2) Relación de medias móviles (momentum)
        ma_5 = precios.rolling(5).mean().mean(axis=1)
        ma_21 = precios.rolling(21).mean().mean(axis=1)
        features['ma_ratio'] = ma_5 / ma_21
        
        # 3) Correlación promedio rolling entre activos
        features['correlacion_promedio'] = self._correlacion_rolling(rendimientos)
        
        # 4) Dispersión cross-sectional de rendimientos
        features['dispersion'] = rendimientos.std(axis=1)
        
        # 5) Momentums a distintas ventanas
        features['momentum_5d'] = (precios / precios.shift(5) - 1).mean(axis=1)
        features['momentum_21d'] = (precios / precios.shift(21) - 1).mean(axis=1)
        
        return pd.DataFrame(features).fillna(0)
    
    def _calcular_rsi(self, precios: pd.DataFrame, periodo: int = 14) -> pd.DataFrame:
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0)
        perdida = -delta.where(delta < 0, 0)
        
        avg_ganancia = ganancia.rolling(periodo).mean()
        avg_perdida = perdida.rolling(periodo).mean()
        
        rs = avg_ganancia / avg_perdida
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _correlacion_rolling(self, rendimientos: pd.DataFrame, ventana: int = 21) -> pd.Series:
        correlaciones = []
        
        for i in range(ventana, len(rendimientos)):
            ventana_datos = rendimientos.iloc[i-ventana:i]
            matriz_corr = ventana_datos.corr()
            triu_indices = np.triu_indices_from(matriz_corr, k=1)
            corr_promedio = matriz_corr.values[triu_indices].mean()
            correlaciones.append(corr_promedio)
        
        serie_completa = [0] * ventana + correlaciones
        return pd.Series(serie_completa, index=rendimientos.index)
    
    def guardar_resultados(self, directorio: str = "./") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_completa = os.path.join(directorio, f"componente_1_resultados_{timestamp}")
        os.makedirs(ruta_completa, exist_ok=True)
        
        # 1) Guardar configuración y parámetros
        config_data = {
            'configuracion': asdict(self.config),
            'parametros': {
                'precios_iniciales': self.parametros.precios_iniciales.tolist(),
                'rendimientos_esperados': self.parametros.rendimientos_esperados.tolist(),
                'volatilidades': self.parametros.volatilidades.tolist(),
                'matriz_correlacion': self.parametros.matriz_correlacion.tolist()
            },
            'metricas_riesgo': self.metricas_riesgo,
            'timestamp': self.timestamp
        }
        
        with open(os.path.join(ruta_completa, 'configuracion.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # 2) Guardar simulaciones crudas
        datos_simulacion = {
            'precios_simulados': self.precios_simulados,
            'shocks_temporales': self.shocks_temporales,
            'eventos_contagio': self.eventos_contagio,
            'matriz_contagio': self.matriz_contagio
        }
        
        with open(os.path.join(ruta_completa, 'datos_simulacion.pkl'), 'wb') as f:
            pickle.dump(datos_simulacion, f)
        
        # 3) Guardar datos de componentes 2, 3 y 4
        for i, metodo in enumerate([self.get_datos_componente_2, self.get_datos_componente_3, self.get_datos_componente_4], 2):
            try:
                datos = metodo()
                with open(os.path.join(ruta_completa, f'datos_componente_{i}.pkl'), 'wb') as f:
                    pickle.dump(datos, f)
            except Exception as e:
                print(f"Error guardando datos componente {i}: {e}")
        
        return ruta_completa




class SimuladorMonteCarlo:
    def __init__(self, config: Optional[ConfiguracionMonteCarlo] = None):

        self.config = config
        self.parametros = None
        self.resultados = None
        
        if config and config.semilla_aleatoria:
            np.random.seed(config.semilla_aleatoria)
    
    def crear_datos_ejemplo(self) -> Tuple[ConfiguracionMonteCarlo, ParametrosFinancieros]:        
        activos = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'GLD',      # ETFs diversificados
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', # Tech / Growth
            'JPM', 'BAC', 'WFC', 'GS', 'MS',         # Bancos
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',      # Salud
            'XOM', 'CVX', 'COP', 'SLB', 'EOG'        # Energía
        ]
        
        n_activos = len(activos)
        np.random.seed(42)
        
        # Precios iniciales "realistas"
        precios_iniciales = np.random.uniform(50, 500, n_activos)
        rendimientos_esperados = np.random.uniform(-0.05, 0.15, n_activos)
        volatilidades = np.random.uniform(0.15, 0.60, n_activos)
        
        # Matriz de correlación base (bloques sectoriales)
        correlacion = np.eye(n_activos)
        
        sectores = {
            'ETFs': range(0, 5),
            'Tech': range(5, 10),
            'Financieros': range(10, 15),
            'Salud': range(15, 20),
            'Energía': range(20, 25)
        }
        
        # Correlaciones altas dentro de cada sector
        for indices in sectores.values():
            for i in indices:
                for j in indices:
                    if i != j:
                        correlacion[i, j] = np.random.uniform(0.4, 0.8)
        
        # Correlaciones cruzadas más bajas
        for i in range(n_activos):
            for j in range(n_activos):
                if correlacion[i, j] == 0:
                    correlacion[i, j] = np.random.uniform(-0.2, 0.3)
                    correlacion[j, i] = correlacion[i, j]
        
        # Proyección a matriz de correlación válida
        eigenvals, eigenvecs = np.linalg.eigh(correlacion)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlacion = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        D = np.sqrt(np.diag(correlacion))
        correlacion = correlacion / np.outer(D, D)
        
        # Config (si no se pasó una específica)
        config = ConfiguracionMonteCarlo(activos=activos) if self.config is None else self.config
        
        # Parámetros
        parametros = ParametrosFinancieros(
            precios_iniciales=precios_iniciales,
            rendimientos_esperados=rendimientos_esperados,
            volatilidades=volatilidades,
            matriz_correlacion=correlacion
        )
        parametros.validar()
        
        return config, parametros
    
    def _crear_matriz_contagio(self, correlacion: np.ndarray) -> np.ndarray:
        matriz = np.abs(correlacion)
        np.fill_diagonal(matriz, 0)
        
        # Normalizar filas para que representen "intensidad de contagio"
        row_sums = matriz.sum(axis=1)
        row_sums[row_sums == 0] = 1
        matriz = matriz / row_sums[:, np.newaxis]
        
        return matriz
    
    def _aplicar_contagio(self, shocks: np.ndarray, matriz_contagio: np.ndarray) -> np.ndarray:
        shocks_modificados = shocks.copy()
        n_activos = len(shocks)
        
        # Identificar activos con shocks extremos
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
                    
                    evento = {
                        'simulacion': sim,
                        'tiempo': t,
                        'activos_origen': activos_afectados.tolist(),
                        'intensidad_maxima': np.max(np.abs(shocks_temporales[sim, t, :])),
                        'num_activos_afectados': len(activos_afectados)
                    }
                    eventos.append(evento)
        
        return eventos
    
    def _calcular_metricas_riesgo(self, precios: np.ndarray, pesos: np.ndarray = None) -> Dict:
        if pesos is None:
            pesos = np.ones(precios.shape[2]) / precios.shape[2]
        
        # Valor del portafolio en cada simulación y tiempo
        valor_portafolio = np.sum(precios * pesos, axis=2)
        valor_final = valor_portafolio[:, -1]
        valor_inicial = valor_portafolio[:, 0]
        
        # Rendimientos de cada simulación
        rendimientos = (valor_final - valor_inicial) / valor_inicial
        
        # Cálculo de métricas básicas de riesgo
        return {
            'rendimiento_promedio': np.mean(rendimientos),
            'volatilidad': np.std(rendimientos),
            'var_95': np.percentile(rendimientos, 5),
            'var_99': np.percentile(rendimientos, 1),
            'cvar_95': np.mean(rendimientos[rendimientos <= np.percentile(rendimientos, 5)]),
            'prob_perdida': np.mean(rendimientos < 0),
            'sharpe_ratio': np.mean(rendimientos) / np.std(rendimientos) if np.std(rendimientos) > 0 else 0,
            'max_drawdown': self._calcular_max_drawdown(valor_portafolio)
        }
    
    def _calcular_max_drawdown(self, valor_portafolio: np.ndarray) -> float:
        drawdowns = []
        
        for sim in range(valor_portafolio.shape[0]):
            valores = valor_portafolio[sim, :]
            peak = np.maximum.accumulate(valores)
            drawdown = (valores - peak) / peak
            drawdowns.append(np.min(drawdown))
        
        return np.mean(drawdowns)
    
    def ejecutar_simulacion(self, 
                          config: Optional[ConfiguracionMonteCarlo] = None,
                          parametros: Optional[ParametrosFinancieros] = None,
                          usar_datos_ejemplo: bool = True) -> ResultadosMonteCarlo:
      
        # Si no se pasa nada, crear datos de ejemplo
        if config is None and parametros is None and usar_datos_ejemplo:
            config, parametros = self.crear_datos_ejemplo()
        elif config is not None and parametros is None:
            raise ValueError("Se necesitan parametros para la simulacion")
        
        self.config = config or self.config
        self.parametros = parametros
        
        if self.config is None:
            raise ValueError("Debe proporcionar configuración")
        
        print(f"Simulando {len(self.config.activos)} activos")
        print(f"{self.config.n_simulaciones:,} simulaciones x {self.config.n_pasos:,} pasos")
        
        # Crear objeto Resultados
        self.resultados = ResultadosMonteCarlo(self.config, self.parametros)
        
        # Matriz de contagio a partir de la correlación
        matriz_contagio = self._crear_matriz_contagio(self.parametros.matriz_correlacion)
        self.resultados.matriz_contagio = matriz_contagio
        
        # Parámetros de tiempo
        dt = self.config.tiempo_anos / self.config.n_pasos
        n_activos = len(self.config.activos)
        
        # Descomposición de Cholesky
        L = cholesky(self.parametros.matriz_correlacion, lower=True)
        
        # Arreglos para precios y shocks
        precios = np.zeros((self.config.n_simulaciones, self.config.n_pasos + 1, n_activos))
        shocks_temporales = np.zeros((self.config.n_simulaciones, self.config.n_pasos, n_activos))
        
        # Condiciones iniciales de precios
        precios[:, 0, :] = self.parametros.precios_iniciales
        
        # Simulación
        for sim in range(self.config.n_simulaciones):
            for t in range(1, self.config.n_pasos + 1):
                # Shocks independientes
                shocks_independientes = np.random.standard_normal(n_activos)
                
                # Aplicar correlación
                shocks_correlacionados = L @ shocks_independientes
                
                # Aplicar contagio si está activado
                if self.config.incluir_contagio:
                    shocks_finales = self._aplicar_contagio(shocks_correlacionados, matriz_contagio)
                else:
                    shocks_finales = shocks_correlacionados
                
                # Guardar shocks
                shocks_temporales[sim, t-1, :] = shocks_finales
                
                # Actualizar precios con GBM
                drift = (self.parametros.rendimientos_esperados - 0.5 * self.parametros.volatilidades**2) * dt
                difusion = self.parametros.volatilidades * np.sqrt(dt) * shocks_finales
                
                precios[sim, t, :] = precios[sim, t-1, :] * np.exp(drift + difusion)
        
        # Guardar resultados base
        self.resultados.precios_simulados = precios
        self.resultados.shocks_temporales = shocks_temporales
        
        # Detectar eventos de contagio
        self.resultados.eventos_contagio = self._detectar_eventos_contagio(shocks_temporales)
        
        # Calcular métricas de riesgo
        self.resultados.metricas_riesgo = self._calcular_metricas_riesgo(precios)
        
        print(f"Eventos de contagio detectados: {len(self.resultados.eventos_contagio)}")
        print(f"VaR 95%: {self.resultados.metricas_riesgo['var_95']*100:.2f}%")
        
        return self.resultados
    
    def mostrar_resumen(self):
        if self.resultados is None:
            print("No hay resultados disponibles.")
            return
        
        print("Resultados obtenidos de Monte Carlo")
        
        print(f"\nCONFIGURACIÓN:")
        print(f"  Activos: {len(self.config.activos)}")
        print(f"  Simulaciones: {self.config.n_simulaciones:,}")
        print(f"  Horizonte: {self.config.tiempo_anos} años")
        print(f"  Contagio: {'Activado' if self.config.incluir_contagio else 'Desactivado'}")
        
        print(f"\nMétricas de reisgo:")
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
            print("Hubo un problema con la visualización de los resultados")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1) Trayectorias del portafolio
        precios = self.resultados.precios_simulados
        pesos = np.ones(precios.shape[2]) / precios.shape[2]
        valor_portafolio = np.sum(precios * pesos, axis=2)
        
        for i in range(min(100, precios.shape[0])):
            axes[0,0].plot(valor_portafolio[i], alpha=0.1, color='blue')
        
        valor_promedio = np.mean(valor_portafolio, axis=0)
        axes[0,0].plot(valor_promedio, 'red', linewidth=2, label='Promedio')
        axes[0,0].set_title('Evolución del Portafolio')
        axes[0,0].set_xlabel('Días')
        axes[0,0].set_ylabel('Valor')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2) Distribución de rendimientos finales
        valor_final = valor_portafolio[:, -1]
        valor_inicial = valor_portafolio[:, 0]
        rendimientos = (valor_final - valor_inicial) / valor_inicial
        
        axes[0,1].hist(rendimientos * 100, bins=50, alpha=0.7, color='skyblue')
        axes[0,1].axvline(self.resultados.metricas_riesgo['var_95'] * 100, 
                         color='red', linestyle='--', label='VaR 95%')
        axes[0,1].set_title('Distribución de Rendimientos')
        axes[0,1].set_xlabel('Rendimiento (%)')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3) Matriz de correlación
        corr = self.parametros.matriz_correlacion
        im = axes[1,0].imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
        axes[1,0].set_title('Matriz de Correlación')
        plt.colorbar(im, ax=axes[1,0])
        
        # 4) Eventos de contagio
        if self.resultados.eventos_contagio:
            tiempos = [e['tiempo'] for e in self.resultados.eventos_contagio]
            intensidades = [e['intensidad_maxima'] for e in self.resultados.eventos_contagio]
            
            axes[1,1].scatter(tiempos, intensidades, alpha=0.6, color='orange')
            axes[1,1].axhline(y=self.config.umbral_shock, color='red', 
                             linestyle='--', label=f'Umbral ({self.config.umbral_shock})')
            axes[1,1].set_title('Eventos de Contagio')
            axes[1,1].set_xlabel('Tiempo (días)')
            axes[1,1].set_ylabel('Intensidad')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def ejecutar_simulacion_rapida(n_simulaciones: int = 1000, 
                              n_activos: int = 25,
                              incluir_contagio: bool = True) -> ResultadosMonteCarlo:
    simulador = SimuladorMonteCarlo()
    
    # Crear datos de ejemplo
    config, parametros = simulador.crear_datos_ejemplo()
    config.n_simulaciones = n_simulaciones
    config.incluir_contagio = incluir_contagio
    
    # Ajustar número de activos si se desea menos
    if n_activos < 25:
        config.activos = config.activos[:n_activos]
        parametros.precios_iniciales = parametros.precios_iniciales[:n_activos]
        parametros.rendimientos_esperados = parametros.rendimientos_esperados[:n_activos]
        parametros.volatilidades = parametros.volatilidades[:n_activos]
        parametros.matriz_correlacion = parametros.matriz_correlacion[:n_activos, :n_activos]
    
    # Ejecutar simulación
    resultados = simulador.ejecutar_simulacion(config=config, parametros=parametros, usar_datos_ejemplo=False)
    
    return resultados

def cargar_resultados(directorio: str) -> ResultadosMonteCarlo:
    # 1) Cargar configuración y parámetros
    with open(os.path.join(directorio, 'configuracion.json'), 'r') as f:
        config_data = json.load(f)
    
    config = ConfiguracionMonteCarlo(**config_data['configuracion'])
    
    parametros = ParametrosFinancieros(
        precios_iniciales=np.array(config_data['parametros']['precios_iniciales']),
        rendimientos_esperados=np.array(config_data['parametros']['rendimientos_esperados']),
        volatilidades=np.array(config_data['parametros']['volatilidades']),
        matriz_correlacion=np.array(config_data['parametros']['matriz_correlacion'])
    )
    
    # 2) Inicializar objeto de resultados
    resultados = ResultadosMonteCarlo(config, parametros)
    resultados.metricas_riesgo = config_data['metricas_riesgo']
    
    # 3) Cargar simulaciones
    with open(os.path.join(directorio, 'datos_simulacion.pkl'), 'rb') as f:
        datos = pickle.load(f)
        resultados.precios_simulados = datos['precios_simulados']
        resultados.shocks_temporales = datos['shocks_temporales']
        resultados.eventos_contagio = datos['eventos_contagio']
        resultados.matriz_contagio = datos['matriz_contagio']
    
    return resultados


if __name__ == "__main__":
      
    simulador = SimuladorMonteCarlo()
    
    resultados = simulador.ejecutar_simulacion()
    
    simulador.mostrar_resumen()
    
    datos_redes = resultados.get_datos_componente_2()
    
    datos_contagio = resultados.get_datos_componente_3()
    
    datos_ia = resultados.get_datos_componente_4()
    
    directorio = resultados.guardar_resultados()
    
    simulador.visualizar_resultados()
